
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


from utils.eval import evaluate_dismatch_metrics, evaluate_trajectory_quality
from utils.logger import flatten_metrics, save_csv_native

# 获取 Hydra 提供的 logger
log = logging.getLogger(__name__)
# --- 注册自定义解析器用于处理文件路径 ---
OmegaConf.register_new_resolver("abspath", lambda x: to_absolute_path(x))


def train_worker(cfg: DictConfig):
    # 读取环境配置并初始化环境

    device = cfg.train.device
    # Hydra 会自动切换工作目录到 outputs/..., 所以 log_dir 设置为当前目录即可
    # 这样 tensorboard 文件会保存在对应的 output 文件夹下
    writer = SummaryWriter(log_dir=".")
    
    log.info(f"Training Config:\n{OmegaConf.to_yaml(cfg)}")

    if hasattr(cfg, 'file_path'):
        cfg.file_path = to_absolute_path(cfg.file_path)

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


    log.info(f"Instantiating Trainer: {cfg.train.trainer._target_}")
    trainer = instantiate(
        cfg.train.trainer, 
        diffusion_model=diffusion, 
        dataset=dataset,
        renderer=None,
        results_folder=".",
    )

    trainer.train(n_train_steps=cfg.train.iteration)
    log.info("Training completed.")
    trainer.save("final")
    
    # 评估阶段
    log.info("Starting evaluation...")


    policy = instantiate(cfg.eval.policy, diffusion_model=diffusion, normalizer=dataset.normalizer)
    cond = None
    action, samples, diffusion_paths, elbo, total_time, avg_per_step_time = policy(cond, batch_size=cfg.eval.eval_samples)
    # (batch_size, horizon, x_dim)
    gene_traj = samples.observations


    batch = next(iter(val_loader))
    true_joint_normed = batch.trajectories # (batch_size, horizon, act_dim + x_dim)
    true_traj_normed = true_joint_normed[:, :, dataset.action_dim:]
    true_traj = policy.normalizer.unnormalize(true_traj_normed, 'observations')
    true_traj = true_traj.cpu().numpy()

    max_seq = cfg.max_seq

    check_horizon = [0, max_seq // 2, max_seq - 1]
    
    eval_metrics = evaluate_dismatch_metrics(
        gene_traj, true_traj, check_horizon_list=check_horizon, max_samples=500
    )

    traj_quality_metrics = evaluate_trajectory_quality(
        gene_traj, env.safety_check
    )

    log.info(
            f"{'MMD='}{np.mean(eval_metrics['mmd']):8.4f} "
            f"{'W2='}{np.mean(eval_metrics['wasserstein']):8.4f} "
            f"{'KL='}{np.mean(eval_metrics['kl']):8.4f} "
            f"{'R='}{traj_quality_metrics['safety_ratio']:8.4f} "
            f"{'CURVE='}{traj_quality_metrics['curvature_smoothness']:8.4f} "
            f"{'ACC='}{traj_quality_metrics['acc_smoothness']:8.4f} "
            f"{'TotalTime='}{total_time:8.4f}s "
            f"{'AvgStepTime='}{avg_per_step_time*1000:8.4f}ms "
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


@hydra.main(config_path="config", config_name="train_safediffuser_unet_maze2d.yaml")
def main(cfg: DictConfig):

    if "seed" in cfg.train:
        seed = cfg.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        log.info(f"Set random seed to: {seed}")

    train_worker(cfg)


if __name__ == "__main__":
    main()