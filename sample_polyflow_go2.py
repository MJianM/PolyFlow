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


class Go2PolyFlowModelPolicy:
    def __init__(self, cfg: DictConfig):

        # 读取环境配置并初始化环境

        device = cfg.device

        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")


        log.info(f"Instantiating Dataset: {cfg.dataset._target_}")
        if hasattr(cfg, 'file_path'):
            cfg.file_path = to_absolute_path(cfg.file_path)
        dataset = instantiate(cfg.dataset)

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

        # 加载模型
        if hasattr(cfg.eval, 'load_model_path'):
            cfg.eval.load_model_path = to_absolute_path(cfg.eval.load_model_path)
        log.info(f"Load model from {cfg.eval.load_model_path}")
        load_data = torch.load(cfg.eval.load_model_path, map_location=device, weights_only=False)
        if cfg.eval.load_ema:
            algo.load_state_dict(load_data['ema'])
        else:
            algo.load_state_dict(load_data['model'])

        
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
        true_vertex_normed = batch.vertex # (B, H, 4, 3)
        true_vertex = policy.normalizer.unnormalize(true_vertex_normed.reshape(-1, 12), 'actions')
        true_vertex = true_vertex.reshape(-1, cfg.horizon, 4, 3)


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
            policy(true_cond, A_0=true_A[:, 0:1], b_0=true_b[:, 0:1], contact_0=true_contact[:, 0:1], vertex_0=true_vertex[:, 0:1], batch_size=batch_size)
        # === 正式测量 ===
        log.info("Running benchmark pass...")


        self.policy = policy
        self.dataset = dataset
        self.val_loader = val_loader

    def __call__(self, obs_0, A_0, b_0, contact_0, h, qpos, qvel, default_q, kp, kd):
        """
        obs_0: (B, obs_dim) torch.Tensor 当前时刻的obs
        A_0: (B, 4, num_cons, 3) torch.Tensor 当前时刻的线性不等式约束矩阵
        b_0: (B, 4, num_cons) torch.Tensor 当前时刻的线性不等式约束向量
        contact_0: (B, 4,) torch.Tensor 当前时刻的接触状态
        h: (B, 4, 3) torch.Tensor 当前时刻四条腿的非线性项
        qpos: (B, 12) torch.Tensor
        qvel: (B, 12) torch.Tensor
        default_q: (12, ) torch.Tensor
        kp: (12, ) torch.Tensor
        kd: (12, ) torch.Tensor

        return:
            action: (B, act_dim) numpy.array
        """

        batch_size = obs_0.shape[0]

        with torch.no_grad():

            cond = {0: obs_0}
            A_input = A_0.unsqueeze(1)  # (B, 1, 4, num_cons, 3)
            b_input = b_0.unsqueeze(1)  # (B, 1, 4, num_cons)
            contact_input = contact_0.unsqueeze(1)  # (B, 1, 4)

            # 计算vertex
            tau_bias = kp.unsqueeze(0) * (default_q.unsqueeze(0) - qpos) - kd.unsqueeze(0) * qvel  # (B, 12)
            vertex = (h.reshape(batch_size, -1) - tau_bias) / kp.unsqueeze(0) # (B, 12)
            vertex_input = vertex.reshape(batch_size, 1, 4, 3) # (B, 1, 4, 3)


            # action: [B, act_dim]
            # trajectories.actions [B, H, A]
            # trajectories.observations [B, H, O]
            # trajectories.values [B]
            # diffusion_obs [B, diffusion_steps, H, O]
            action, trajectories, diffusion_obs, _, total_time, avg_per_step_time = self.policy(
                cond={0: obs_0.cpu()}, A_0=A_input.cpu(), b_0=b_input.cpu(), contact_0=contact_input.cpu(), 
                vertex_0=vertex_input.cpu(), batch_size=batch_size
            )


        return action / self.dataset.action_scale
        



@hydra.main(config_path="config", config_name="train_polyflow_go2.yaml")
def main(cfg: DictConfig):

    with Display(visible=0, size=(1024, 768), backend="xvfb") as disp:
        if "seed" in cfg:
            seed = cfg.seed
            set_all_seed(seed)
            log.info(f"Set random seed to: {seed}")

        policy = Go2PolyFlowModelPolicy(cfg)




if __name__ == "__main__":

    main()

