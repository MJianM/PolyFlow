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
from pyvirtualdisplay import Display

from src.utils.eval import evaluate_dismatch_metrics, evaluate_trajectory_quality
from src.utils.logger import flatten_metrics, save_csv_native
from src.utils.arrays import apply_dict, set_all_seed

# 获取 Hydra 提供的 logger
log = logging.getLogger(__name__)

def train_worker(cfg: DictConfig):
    # 读取环境配置并初始化环境

    device = cfg.device
    # Hydra 会自动切换工作目录到 outputs/..., 所以 log_dir 设置为当前目录即可
    # 这样 tensorboard 文件会保存在对应的 output 文件夹下
    writer = SummaryWriter(log_dir=".")
    
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # if hasattr(cfg, 'file_path'):
    #     cfg.file_path = to_absolute_path(cfg.file_path)

    log.info(f"Instantiating Dataset: {cfg.dataset._target_}")
    if hasattr(cfg, 'file_path'):
        cfg.file_path = to_absolute_path(cfg.file_path)
    dataset = instantiate(cfg.dataset)
    
    # log.info(f"Instantiating Env Handler: {cfg.env._target_}")
    # env_handler = instantiate(cfg.env)

    log.info(f"Instantiating Eval DataLoader: {cfg.val_dataloader._target_}")
    val_loader = instantiate(cfg.val_dataloader, dataset=dataset)

    log.info(f"Instantiating Backbone: {cfg.backbone._target_}")
    backbone = instantiate(cfg.backbone)

    log.info(f"Instantiating Diffusion: {cfg.algorithm._target_}")
    algo = instantiate(cfg.algorithm, model=backbone).to(device)
    # CRITICAL: Set normalization parameters for the safety check.
    # The 'invariance' method in diffusion.py relies on self.norm_mins/maxs 
    # to normalize coordinates for obstacle checking.
    # Note: dataset.normalizer is a DatasetNormalizer, we need the specific normalizer for observations
    if cfg.dataset.normalizer == 'GaussianNormalizer':
        algo.means = torch.from_numpy(dataset.normalizer.normalizers['observations'].means).to(device).float()
        algo.stds = torch.from_numpy(dataset.normalizer.normalizers['observations'].stds).to(device).float()
        algo.act_means = torch.from_numpy(dataset.normalizer.normalizers['actions'].means).to(device).float()
        algo.act_stds = torch.from_numpy(dataset.normalizer.normalizers['actions'].stds).to(device).float()
    else:
        algo.norm_mins = torch.from_numpy(dataset.normalizer.normalizers['observations'].mins).to(device).float()
        algo.norm_maxs = torch.from_numpy(dataset.normalizer.normalizers['observations'].maxs).to(device).float()
        algo.act_norm_mins = torch.from_numpy(dataset.normalizer.normalizers['actions'].mins).to(device).float()
        algo.act_norm_maxs = torch.from_numpy(dataset.normalizer.normalizers['actions'].maxs).to(device).float()    


    log.info(f"Instantiating Trainer: {cfg.trainer._target_}")
    trainer = instantiate(
        cfg.trainer, 
        diffusion_model=algo, 
        dataset=dataset,
        renderer=None,
        results_folder=".",
    )

    trainer.train(n_train_steps=cfg.iteration, 
                  use_cosine_scheduler=True, writer=writer,
                  use_grad_clip=True, grad_clip_norm=1.0)
    log.info("Training completed.")
    trainer.save("final")
    
    # 评估阶段
    log.info("Starting evaluation...")


    policy = instantiate(cfg.policy, guide=None, diffusion_model=algo, normalizer=dataset.normalizer, dataset=dataset)

    algo.eval() # 确保 eval 模式

    batch = next(iter(val_loader))
    true_joint_normed = batch.trajectories # [B, H, A+O]
    true_cond_normed = batch.conditions  # {0: [B, O]}
    true_traj_normed = true_joint_normed[:, :, dataset.action_dim:]
    true_traj = policy.normalizer.unnormalize(true_traj_normed, 'observations')
    true_act_traj_normed = true_joint_normed[:, :, :dataset.action_dim]
    true_act_traj = policy.normalizer.unnormalize(true_act_traj_normed, 'actions')
    
    true_cond = apply_dict(policy.normalizer.unnormalize, true_cond_normed, 'observations')
    true_A_normed = batch.A
    true_b_normed = batch.b
    true_A, true_b = dataset.unnormalize_constraints_tensor(true_A_normed, true_b_normed)
    true_contact = batch.contact
    batch_size = true_joint_normed.shape[0]



    # === 新增：Warm-up (预热) ===
    log.info("Running warm-up pass...")
    with torch.no_grad():
        # 用同样的 batch 跑一次，不计入时间
        # 这一步会触发 CuDNN benchmark, Kernel loading, Allocator setup
        policy(true_cond, A_0=true_A[:, 0:1], b_0=true_b[:, 0:1], contact_0=true_contact[:, 0:1], batch_size=batch_size)
    # === 正式测量 ===
    log.info("Running benchmark pass...")

    # action: [B, act_dim]
    # trajectories.actions [B, H, A]
    # trajectories.observations [B, H, O]
    # trajectories.values [B]
    # diffusion_obs [B, diffusion_steps, H, O]
    action, trajectories, diffusion_obs, _, total_time, avg_per_step_time = policy(true_cond, A_0=true_A[:, 0:1], b_0=true_b[:, 0:1], contact_0=true_contact[:, 0:1], batch_size=batch_size)
    
    # 检验与真实数据的匹配程度
    horizon = cfg.horizon
    check_horizon = [i for i in range(0, horizon)]
    eval_metrics = evaluate_dismatch_metrics(
        sampled_traj=trajectories.actions, true_traj=true_act_traj, check_horizon_list=check_horizon, max_samples=1000
    )

    # 检查与真实动作的误差
    action_tensor = torch.from_numpy(action).to(true_act_traj.device)
    print(action_tensor.shape)
    mse_error = torch.nn.functional.mse_loss(action_tensor, true_act_traj[:, 0])

    # # 检查是否满足约束
    # # (batch, 4, num_cons)
    # print("true A:", true_A[:, 0])
    # lhs = torch.sum(true_A[:, 0] * action_tensor.reshape(batch_size, 4, 1, 3), dim=-1) - true_b[:, 0]
    # flag = torch.all(lhs < 1e-2)


    # log.info(
    #         f"{'MMD='}{np.mean(eval_metrics['mmd']):8.4f} "
    #         f"{'W2='}{np.mean(eval_metrics['wasserstein']):8.4f} "
    #         f"{'KL='}{np.mean(eval_metrics['kl']):8.4f} "
    #         f"{'TotalTime='}{total_time:8.4f}s "
    #         f"{'AvgStepTime='}{avg_per_step_time*1000:8.4f}ms "
    #         f"Mse Error: {mse_error:8.4f} "
    #         f"Safe Flag: {flag} "
    # )


    # 判断安全性
    #  lhs 的形状是 (Batch, 4, Num_Cons)
    lhs = torch.sum(true_A[:, 0] * action_tensor.reshape(batch_size, 4, 1, 3), dim=-1) - true_b[:, 0]
    flag = torch.all(lhs < 1e-3, dim=-1, keepdim=False) # (batch, 4)
    final_flag = flag[true_contact[:, 0] > 0] # (n,)
    bool_flag = torch.all(final_flag)

    # # 3. 【新增】找到最大违背值及其索引
    # max_violation_val, max_idx_flat = torch.max(lhs.view(-1), dim=0)
    # max_violation_val = max_violation_val.item()

    # # 获取最大值对应的多维索引 (Batch_idx, Agent_idx, Cons_idx)
    # # 使用 nonzero 获取坐标，取第一个结果以防有多个相同的最大值
    # max_indices = (lhs == max_violation_val).nonzero(as_tuple=False)[0]
    # b_idx, agent_idx, cons_idx = max_indices[0].item(), max_indices[1].item(), max_indices[2].item()

    # # 4. 【新增】提取对应的 A, x, b
    # # 根据 lhs 的计算公式反推对应的数据切片
    # # x: 来自 action_tensor
    # violation_x = action_tensor[b_idx, 3*agent_idx:3*agent_idx+3].detach().cpu().numpy()

    # violation_A = true_A[b_idx, 0, agent_idx, :].detach().cpu().numpy() 

    # violation_b = true_b[b_idx, 0, agent_idx, :].detach().cpu().numpy()

    # 5. 修改日志输出
    log.info(
        f"{'MMD='}{np.mean(eval_metrics['mmd']):8.4f} "
        f"{'W2='}{np.mean(eval_metrics['wasserstein']):8.4f} "
        f"{'KL='}{np.mean(eval_metrics['kl']):8.4f} "
        f"{'TotalTime='}{total_time:8.4f}s "
        f"{'AvgStepTime='}{avg_per_step_time*1000:8.4f}ms "
        f"Mse Error: {mse_error:8.4f} "
        f"Safe Flag: {bool_flag} "
    )

    # print("max violation val:", max_violation_val)
    # print("b=", b_idx)
    # print("leg=", agent_idx)
    # print("c=", cons_idx)
    # print("A= ", violation_A)
    # print("b= ", violation_b)
    # print("x= ", violation_x)


    obs_traj_list, obs_expand_traj_list, ret_list, rollout_metrics = [], [], [], {}

    # 这两个是变长的列表
    # 必须显式转为 object 数组，否则 np.savez 尝试自动堆叠会失败
    obs_traj_arr = np.array(obs_traj_list, dtype=object)
    obs_expand_traj_arr = np.array(obs_expand_traj_list, dtype=object)
    np.savez(
        "final_traj.npz",
        obs_traj_list=obs_traj_arr, # [(episode_length1, obs_dim),...]
        obs_expand_traj_list=obs_expand_traj_arr, # [(episode_length1, obs_dim+1),...]
        ret_list=ret_list, # [float,]
        true_traj=true_traj, # (batch, horizon, obs_dim)
        true_act_traj=true_act_traj,
        gene_traj=trajectories.observations, # (batch, horizon, obs_dim)
        gene_act_traj=trajectories.actions,  # (batch, horizon, act_dim)
    )

    # data = np.load("final_traj.npz", allow_pickle=True)


    log_dict = {}
    for key, value in eval_metrics.items():
        log_dict[key] = np.mean(value)

    log_dict['TotalTime'] = total_time
    log_dict['AvgStepTime'] = avg_per_step_time
    log_dict = flatten_metrics(log_dict, check_horizon)
    save_csv_native(log_dict, save_path="final_eval_metrics.csv")



@hydra.main(config_path="config", config_name="train_polyflow_go2.yaml")
def main(cfg: DictConfig):

    with Display(visible=0, size=(1024, 768), backend="xvfb") as disp:
        if "seed" in cfg:
            seed = cfg.seed
            set_all_seed(seed)
            log.info(f"Set random seed to: {seed}")

        train_worker(cfg)


if __name__ == "__main__":

    main()

