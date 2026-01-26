import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import tqdm 

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path, instantiate
import logging

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from utils.dataset import TrajDataset
from model.new_model import PolytopeConstrainedFlowModel
from model.safe_flow_sampler import SafeFlowSampler
from utils.utils import ot_minibatch_coupling
from utils.eval import evaluate_dismatch_metrics, evaluate_trajectory_quality
from utils.logger import flatten_metrics, save_csv_native

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("abspath", lambda x: to_absolute_path(x))

def train_worker(cfg: DictConfig):

    device = cfg.train.device
    writer = SummaryWriter(log_dir=".")
    
    log.info(f"Training Config:\n{OmegaConf.to_yaml(cfg)}")

    if hasattr(cfg.dataset, 'file_path'):
        cfg.dataset.file_path = to_absolute_path(cfg.dataset.file_path)

    log.info(f"Instantiating Dataset: {cfg.dataset._target_}")
    dataset = instantiate(cfg.dataset)
    
    log.info(f"Instantiating Env: {cfg.env._target_}")
    env = instantiate(cfg.env)

    log.info(f"Instantiating DataLoader: {cfg.train_dataloader._target_}")
    dataloader = instantiate(cfg.train_dataloader, dataset=dataset)
    val_loader = instantiate(cfg.val_dataloader, dataset=dataset)


    log.info(f"Instantiating Model: {cfg.model._target_}")
    model = instantiate(cfg.model)
    model.to(device) 


    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    model.train()
    t_iter = tqdm.trange(cfg.train.iteration, desc="Training")
    data_iter = iter(dataloader)
    steps = cfg.train.steps

    for i in t_iter:
        try:
            batch_data_dict = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_data_dict = next(data_iter)

        if cfg.train.method == "discrete":
            loss = train_batch_poly_flow(model, dataset, batch_data_dict, optimizer, scheduler, device, cfg)
        elif cfg.train.method == "flow":
            loss = train_batch_flow_matching(model, dataset, batch_data_dict, optimizer, scheduler, device, cfg)
        elif cfg.train.method == "diffusion":
            raise NotImplementedError("Diffusion method not implemented yet.")
        else:
            raise ValueError(f"Unknown training method: {cfg.train.method}")

        # Logging
        if i % cfg.train.log_freq == 0:
            writer.add_scalar('Train/Loss', loss.item(), i)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], i)
            log.debug(f"Iter {i}: Loss {loss.item():.6f}")
            t_iter.set_postfix(loss=f"{loss.item():.4f}")        

        # Evaluation
        if i % cfg.train.eval_freq == 0 and i > 0:
            run_eval(model, val_loader, dataset, env, writer, i, cfg, device, log)

        # Save
        if i % cfg.train.save_freq == 0 and i > 0:
            torch.save(model.state_dict(), f'model_iter_{i}.pt')

    torch.save(model.state_dict(), f'model_final.pt')

    seq_length = dataset.seq_length
    x_dim = dataset.x_dim
    eval_samples = cfg.eval.eval_samples
    generated_traj, total_time, avg_per_step_time = sample_worker(cfg, model, dataset, env, n_samples=eval_samples) # (n_samples, seq_length*x_dim)
    generated_traj = generated_traj[-1].reshape(eval_samples, seq_length, x_dim)
    true_traj = dataset.sample_traj_data(n_sample=eval_samples)
    true_traj = true_traj.reshape(eval_samples, seq_length, x_dim)
    env.plot_trajectory_comparison(
        true_trajs=true_traj, 
        gene_trajs=generated_traj, 
        plot_ellips=cfg.eval.plot_ellips,
        max_plot=cfg.eval.max_plot_traj,
        save_path=f"final_traj_compare.png"
    )

    check_horizon = [0, seq_length // 2, seq_length - 1]
    eval_metrics = evaluate_dismatch_metrics(
        generated_traj, true_traj, check_horizon_list=check_horizon, max_samples=1000
    )
    traj_quality_metrics = evaluate_trajectory_quality(
        generated_traj, env.safety_check
    )
    log_dict = {}
    for key, value in eval_metrics.items():
        log_dict[key] = value
    for key, value in traj_quality_metrics.items():
        log_dict[key] = value
    log_dict['TotalTime'] = total_time
    log_dict['AvgStepTime'] = avg_per_step_time
    log_dict = flatten_metrics(log_dict, check_horizon)
    save_csv_native(log_dict, save_path="final_eval_metrics.csv")

    writer.close()

def train_batch_poly_flow(model, dataset, batch_data_dict, optimizer, scheduler, device, cfg):

    batch_traj = batch_data_dict['traj'].float().to(device) # [batch_size, seq_length*x_dim]
    batch_A = batch_data_dict['A'].float().to(device) # [batch_size, seq_length, num_cons, x_dim]
    batch_b = batch_data_dict['b'].float().to(device) # [batch_size, seq_length, num_cons]

    x1 = batch_traj
    x0, _, _ = dataset.generate_prior_data(batch_size=x1.shape[0], A=batch_A, b=batch_b)
    x0 = x0.float().to(device)
    
    if cfg.train.use_ot:
        x0, x1 = ot_minibatch_coupling(x0, x1)

    # Flow Matching Loss Calculation
    steps = cfg.train.steps
    k = torch.randint(0, steps, (x1.size(0),), device=device)
    t_curr = (k.float() / steps).unsqueeze(1)       # [B, 1]
    t_next = ((k.float() + 1) / steps).unsqueeze(1) # [B, 1]
    
    x_curr = (1 - t_curr) * x0 + t_curr * x1
    x_next = (1 - t_next) * x0 + t_next * x1
    target_delta = x_next - x_curr
    
    pred_delta, _, _ = model(x_curr, t_curr.squeeze(-1), batch_A, batch_b)
    loss = F.mse_loss(pred_delta, target_delta)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return loss

def train_batch_flow_matching(model, dataset, batch_data_dict, optimizer, scheduler, device, cfg):

    batch_traj = batch_data_dict['traj'].float().to(device) # [batch_size, seq_length*x_dim]
    batch_A = batch_data_dict['A'].float().to(device) # [batch_size, seq_length, num_cons, x_dim]
    batch_b = batch_data_dict['b'].float().to(device) # [batch_size, seq_length, num_cons]

    batch_size = batch_traj.shape[0]
    seq_length = dataset.seq_length
    x_dim = dataset.x_dim

    x1 = batch_traj.reshape(batch_size, seq_length, x_dim)

    # t ~ U(0, 1)
    t = torch.rand(batch_size, device=device)
    x0 = torch.randn_like(x1)
    xt = (1 - t.view(batch_size, 1, 1)) * x0 + t.view(batch_size, 1, 1) * x1
    # Conditional Vector Field
    ut = x1 - x0
    vt = model(xt, t)
    loss = F.mse_loss(vt, ut)

    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return loss

def run_eval(model, val_loader, dataset, env, writer, step_i, cfg, device, log):
    model.eval()
    with torch.no_grad():
        try:
            val_batch = next(iter(val_loader))
        except StopIteration:
            val_batch = next(iter(val_loader))
        
        val_A = val_batch['A'].float().to(device)
        val_b = val_batch['b'].float().to(device)
        true_traj = val_batch['traj'].float().to(device) # (B, seq_len*x_dim)

        x_dim = dataset.x_dim 
        B = true_traj.shape[0]
        true_traj = true_traj.view(B, -1, x_dim)

        true_traj_np = true_traj.cpu().numpy()

        #  (Steps+1, B, seq_len*x_dim)
        sampled_traj_np, total_time, avg_per_step_time = sample_worker(cfg, model, dataset, env, n_samples=B)
        sampled_traj_np = sampled_traj_np[-1] #  (B, seq_len*x_dim)
        sampled_traj_np = sampled_traj_np.reshape(B, -1, x_dim) # (B, T, D)
        

        max_seq = dataset.seq_length
        check_horizon = [0, max_seq // 2, max_seq - 1]
        
        eval_metrics = evaluate_dismatch_metrics(
            sampled_traj_np, true_traj_np, check_horizon_list=check_horizon, max_samples=500
        )
        for t_idx in check_horizon:
            writer.add_scalar(f'Eval/MMD_t{t_idx}', eval_metrics['mmd'][check_horizon.index(t_idx)], step_i)
            writer.add_scalar(f'Eval/Wasserstein_t{t_idx}', eval_metrics['wasserstein'][check_horizon.index(t_idx)], step_i)
            writer.add_scalar(f'Eval/KL_t{t_idx}', eval_metrics['kl'][check_horizon.index(t_idx)], step_i)
                
        traj_quality_metrics = evaluate_trajectory_quality(
            sampled_traj_np, env.safety_check
        )
        for key, value in traj_quality_metrics.items():
            writer.add_scalar(f'Eval/{key}', value, step_i)
        
        writer.add_scalar('Eval/TotalTime', total_time, step_i)
        writer.add_scalar('Eval/AvgStepTime', avg_per_step_time, step_i)

        log.info(f"Eval Iter {step_i}: "
                f"{'MMD='}{np.mean(eval_metrics['mmd']):8.4f} "
                f"{'W2='}{np.mean(eval_metrics['wasserstein']):8.4f} "
                f"{'KL='}{np.mean(eval_metrics['kl']):8.4f} "
                f"{'R='}{traj_quality_metrics['safety_ratio']:8.4f} "
                f"{'CURVE='}{traj_quality_metrics['curvature_smoothness']:8.4f} "
                f"{'ACC='}{traj_quality_metrics['acc_smoothness']:8.4f} "
                f"{'TotalTime='}{total_time:8.4f}s "
                f"{'AvgStepTime='}{avg_per_step_time*1000:8.4f}ms "
                )

    model.train()    

@hydra.main(config_path="config", config_name="train_oneray_lmaze.yaml")
def main(cfg: DictConfig):

    if "seed" in cfg.train:
        seed = cfg.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        log.info(f"Set random seed to: {seed}")

    train_worker(cfg)


@torch.no_grad()
def sample_worker(cfg: DictConfig, model, dataset: TrajDataset, env, n_samples=10):
    device = cfg.sample.device
    steps = cfg.sample.steps
    method = cfg.sample.method
    projection = cfg.sample.get("projection", "none")

    if method == "discrete":
        sampled_traj, total_time, avg_per_step_time = sample_discrete_delta(model, dataset, env, n_samples=n_samples, steps=steps)
    elif method == "flow":
        sampled_traj, total_time, avg_per_step_time = sample_flow_matching(model, dataset, env, n_samples=n_samples, steps=steps, projection=projection)
    elif method == "diffusion":
        sampled_traj, total_time, avg_per_step_time = sample_diffusion_model(model, dataset, env,n_samples=n_samples, steps=steps, projection=projection)
    elif method == 'safeflow':
        sampled_traj, total_time, avg_per_step_time = sample_safeflow(model, dataset, env, n_samples=n_samples, steps=steps, cfg=cfg)
    
    return sampled_traj, total_time, avg_per_step_time


@torch.no_grad()
def sample_safeflow(model, dataset: TrajDataset, env, n_samples=10, steps=10, projection="none", cfg=None):

    device = next(model.parameters()).device
    model.eval()

    obstacles = env.maze_obs.get_ellips_list()

    if cfg is not None:
        safe_sampler = SafeFlowSampler(model, obstacles=obstacles, device=device,
                    clip_u=cfg.sample.clip_u, clip_grad=cfg.sample.clip_grad, slack_penalty=cfg.sample.slack_penalty)

    else:
        safe_sampler = SafeFlowSampler(model, obstacles=obstacles, device=device)

    if device.type == 'cuda':
        torch.cuda.synchronize() # 
    start_time = time.time()

    # (n_samples, seq_length, x_dim)
    gene_traj = safe_sampler.sample(n_samples=n_samples, horizon=dataset.seq_length, steps=steps,
                                    use_cbf=True, use_closed_form=False)
    # gene_traj = safe_sampler.sample_rk4(n_samples=n_samples, horizon=dataset.seq_length, steps=steps,
    #                                 use_cbf=True, use_closed_form=False)
    traj_history = [gene_traj.cpu().numpy().reshape(n_samples, -1)]

    if device.type == 'cuda':
        torch.cuda.synchronize() 
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_step = total_time / steps

    return np.array(traj_history), total_time, avg_time_per_step

@torch.no_grad()
def sample_flow_matching(model, dataset: TrajDataset, env, n_samples=10, steps=10, projection="none"):

    device = next(model.parameters()).device
    model.eval()

    seq_len = dataset.seq_length
    x_dim = dataset.x_dim

    x = torch.randn((n_samples, seq_len, x_dim), device=device)
    traj_history = [x.cpu().numpy().reshape(n_samples, -1)]

    if device.type == 'cuda':
        torch.cuda.synchronize() 
    start_time = time.time()

    dt = 1.0 / steps
    for i in range(steps):
        t = torch.ones(n_samples, device=device) * (i / steps)
        v = model(x, t)
        x_new = x + v * dt  # Euler step

        if projection == "truncation":
            x_proj = env.Shield(x, x_new, t)
        elif projection == "none":
            x_proj = x_new
        elif projection == "classifier_guidance":
            x_proj = env.GD(x, x_new, t)
        else:
            raise ValueError(f"Unknown projection method: {projection}")
        
        x = x_proj

        traj_history.append(x.cpu().numpy().reshape(n_samples, -1))

    if device.type == 'cuda':
        torch.cuda.synchronize() 
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_step = total_time / steps

    return np.array(traj_history), total_time, avg_time_per_step


@torch.no_grad()
def sample_diffusion_model(model, dataset: TrajDataset, env, n_samples=10, steps=10, projection="none"):

    return None

@torch.no_grad()
def sample_discrete_delta(model: PolytopeConstrainedFlowModel, dataset: TrajDataset, env, n_samples=10, steps=10):
    """
    Sample process: x_{k+1} = x_k + Model(x_k, t_k)
    """
    device = next(model.parameters()).device
    model.eval()
    
    # x [B, seq_length*x_dim]
    # A [B, seq_length, num_cons, x_dim]
    # b [B, seq_length, num_cons]
    x, A, b = dataset.generate_prior_data(batch_size=n_samples)
    x, A, b = x.float().to(device), A.float().to(device), b.float().to(device)
    
    traj_history = [x.cpu().numpy()]
    
    print(f"Sampling with Delta Prediction ({steps} steps)...")
    
    if device.type == 'cuda':
        torch.cuda.synchronize() 
    start_time = time.time()

    for k in range(steps):
        t_curr = torch.ones(n_samples, device=device) * (k / steps)
        
        pred_delta, _, _ = model(x, t_curr, A, b)
        
        x = x + pred_delta
        
        traj_history.append(x.cpu().numpy())
        
    if device.type == 'cuda':
        torch.cuda.synchronize() 
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_step = total_time / steps
    
    return np.array(traj_history), total_time, avg_time_per_step

if __name__ == "__main__":
    main()
