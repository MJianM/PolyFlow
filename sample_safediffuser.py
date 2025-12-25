import torch
import numpy as np
import math

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path, instantiate
import logging

from utils.eval import evaluate_dismatch_metrics, evaluate_trajectory_quality
from utils.logger import flatten_metrics, save_csv_native

# 获取 Hydra 提供的 logger
log = logging.getLogger(__name__)

def sample_and_eval(cfg: DictConfig):
    # 读取环境配置并初始化环境

    device = cfg.device
    
    log.info(f"Sampling Config:\n{OmegaConf.to_yaml(cfg)}")

    if hasattr(cfg, 'file_path'):
        cfg.file_path = to_absolute_path(cfg.file_path)
    if hasattr(cfg.sample, 'load_model_path'):
        cfg.sample.load_model_path = to_absolute_path(cfg.sample.load_model_path)

    log.info(f"Instantiating Dataset: {cfg.dataset._target_}")
    dataset = instantiate(cfg.dataset)
    
    log.info(f"Instantiating Env: {cfg.env._target_}")
    env = instantiate(cfg.env)

    # log.info(f"Instantiating DataLoader: {cfg.train_dataloader._target_}")
    val_loader = instantiate(cfg.val_dataloader, dataset=dataset)

    diffusion = instantiate(cfg.model, ellips_list=env.maze_obs.get_ellips_list()).to(device)
    # CRITICAL: Set normalization parameters for the safety check.
    # The 'invariance' method in diffusion.py relies on self.norm_mins/maxs 
    # to normalize coordinates for obstacle checking.
    # Note: dataset.normalizer is a DatasetNormalizer, we need the specific normalizer for observations
    diffusion.norm_mins = torch.from_numpy(dataset.normalizer.normalizers['observations'].mins).to(device).float()
    diffusion.norm_maxs = torch.from_numpy(dataset.normalizer.normalizers['observations'].maxs).to(device).float()


    log.info(f"Loading Model...")
    data = torch.load(cfg.sample.load_model_path, map_location=device)
    diffusion.load_state_dict(data['model'])

    policy = instantiate(cfg.eval.policy, diffusion_model=diffusion, normalizer=dataset.normalizer)
    cond = None
    action, samples, diffusion_paths, elbo, total_time, avg_per_step_time = policy(cond, batch_size=cfg.eval.eval_samples)
    # array: (batch_size, horizon, x_dim)
    gene_traj = samples.observations

    batch = next(iter(val_loader))
    true_joint_normed = batch.trajectories # (batch_size, horizon, act_dim + x_dim)
    true_traj_normed = true_joint_normed[:, :, dataset.action_dim:]
    true_traj = policy.normalizer.unnormalize(true_traj_normed, 'observations')
    true_traj = true_traj.cpu().numpy()

    max_seq = cfg.max_seq
    x_dim = cfg.x_dim

    check_horizon = [0, max_seq // 2, max_seq - 1]
    
    eval_metrics = evaluate_dismatch_metrics(
        gene_traj, true_traj, check_horizon_list=check_horizon, max_samples=500
    )

    traj_quality_metrics = evaluate_trajectory_quality(
        gene_traj, env.safety_check
    )

    env.plot_trajectory_comparison(
        true_trajs=true_traj, 
        gene_trajs=gene_traj, 
        plot_ellips=cfg.eval.plot_ellips,
        max_plot=cfg.eval.max_plot_traj,
        save_path=f"final_traj_compare.png"
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

    # 保存轨迹
    np.savez(
        file="sampled_traj.npz",
        generated_traj=gene_traj,
        true_traj=true_traj,
        seq_length=max_seq,
        x_dim=x_dim
    )

@hydra.main(config_path="config", config_name="sample_safediffuser_maze2d.yaml")
def main(cfg: DictConfig):


    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    log.info(f"Set random seed to: {seed}")

    sample_and_eval(cfg)


if __name__=="__main__":
    main()