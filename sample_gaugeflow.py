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
from train_polyflow import run_eval, run_halfcheetah_eval

# 获取 Hydra 提供的 logger
log = logging.getLogger(__name__)

def train_worker(cfg: DictConfig):
    # 读取环境配置并初始化环境

    device = cfg.device
    # Hydra 会自动切换工作目录到 outputs/..., 所以 log_dir 设置为当前目录即可
    # 这样 tensorboard 文件会保存在对应的 output 文件夹下
    writer = SummaryWriter(log_dir=".")
    
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")


    log.info(f"Instantiating Dataset: {cfg.dataset._target_}")
    dataset = instantiate(cfg.dataset)
    
    normed_single_A = dataset.norm_A
    normed_single_b = dataset.norm_b
    normed_center_point = dataset.center

    log.info(f"Instantiating Env Handler: {cfg.env._target_}")
    env_handler = instantiate(cfg.env)

    log.info(f"Instantiating Eval DataLoader: {cfg.val_dataloader._target_}")
    val_loader = instantiate(cfg.val_dataloader, dataset=dataset)

    log.info(f"Instantiating Backbone: {cfg.backbone._target_}")
    backbone = instantiate(cfg.backbone)

    log.info(f"Instantiating Diffusion: {cfg.algorithm._target_}")
    algo = instantiate(cfg.algorithm, model=backbone,
                normed_single_A=normed_single_A, normed_single_b=normed_single_b,
                normed_center_point=normed_center_point).to(device)
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
    algo.load_state_dict(load_data['ema'])

    
    # 评估阶段
    log.info("Starting evaluation...")

    if 'HalfCheetah' in cfg.env._target_:
        run_halfcheetah_eval(
            cfg=cfg, guide=None, algo=algo, dataset=dataset, val_loader=val_loader,
            env_handler=env_handler, log=log
        )  
    else:      
        run_eval(
            cfg=cfg, guide=None, algo=algo, dataset=dataset, val_loader=val_loader,
            env_handler=env_handler, log=log
        )

@hydra.main(config_path="config", config_name="sample_safeflow_hoppercpx.yaml")
def main(cfg: DictConfig):

    with Display(visible=0, size=(1024, 768), backend="xvfb") as disp:
        if "seed" in cfg:
            seed = cfg.seed
            set_all_seed(seed)
            log.info(f"Set random seed to: {seed}")

        train_worker(cfg)


if __name__ == "__main__":

    main()


    # log.info(f"Instantiating Policy: {cfg.policy._target_}")
    # policy = instantiate(cfg.policy, guide=None, diffusion_model=diffusion, normalizer=dataset.normalizer)
    
    # batch = next(iter(val_loader))
    # true_joint_normed = batch.trajectories # [B, H, A+O]
    # true_cond_normed = batch.conditions  # {0: [B, O]}
    # true_traj_normed = true_joint_normed[:, :, dataset.action_dim:]
    # true_traj = policy.normalizer.unnormalize(true_traj_normed, 'observations')
    # true_cond = apply_dict(policy.normalizer.unnormalize, true_cond_normed, 'observations')
    # batch_size = true_joint_normed.shape[0]

    # # action: [B, act_dim]
    # # trajectories.actions [B, H, A]
    # # trajectories.observations [B, H, O]
    # # trajectories.values [B]
    # # diffusion_obs [B, diffusion_steps, H, O]
    # action, trajectories, diffusion_obs, _, total_time, avg_per_step_time = policy(true_cond, batch_size)
    
    # # 检验与真实数据的匹配程度
    # horizon = cfg.horizon
    # check_horizon = [1, horizon // 2, horizon - 1]
    # eval_metrics = evaluate_dismatch_metrics(
    #     sampled_traj=trajectories.observations, true_traj=true_traj, check_horizon_list=check_horizon, max_samples=1000
    # )
    # # 检验生成轨迹质量
    # traj_quality_metrics = evaluate_trajectory_quality(
    #     trajectories=trajectories.observations, safety_check_fn=env_handler.safety_check,
    #     check_index_list=cfg.eval.check_index_list # 躯干高度，角度，三个关节角度
    # )

    # log.info(
    #         f"{'MMD='}{np.mean(eval_metrics['mmd']):8.4f} "
    #         f"{'W2='}{np.mean(eval_metrics['wasserstein']):8.4f} "
    #         f"{'KL='}{np.mean(eval_metrics['kl']):8.4f} "
    #         f"{'R='}{traj_quality_metrics['safety_ratio']:8.4f} "
    #         f"{'CURVE='}{traj_quality_metrics['curvature_smoothness']:8.4f} "
    #         f"{'ACC='}{traj_quality_metrics['acc_smoothness']:8.4f} "
    #         f"{'TotalTime='}{total_time:8.4f}s "
    #         f"{'AvgStepTime='}{avg_per_step_time*1000:8.4f}ms "
    # )

    # # 检查 rollout
    # obs_traj_list, obs_expand_traj_list, ret_list, rollout_metrics = env_handler.rollout(
    #     policy, n_episodes=cfg.eval.n_episodes, seed=cfg.eval.seed,
    #     is_video=cfg.eval.is_video, video_episodes=cfg.eval.video_episodes)

    # log.info(
    #         f"{'RetMean='}{np.mean(rollout_metrics['ret_mean']):8.4f} "
    #         f"{'RetStd='}{np.mean(rollout_metrics['ret_std']):8.4f} "
    #         f"{'Safety='}{np.mean(rollout_metrics['safety_ratio']):8.4f} "
    # )
    # # 绘制轨迹图
    # env_handler.plot_expand_trajectory(
    #     traj_expand_list=obs_expand_traj_list, plot_height_limit=True,
    #     max_plot=2, save_path="rollout_result.png"
    # )


    # # 这两个是变长的列表
    # # 必须显式转为 object 数组，否则 np.savez 尝试自动堆叠会失败
    # obs_traj_arr = np.array(obs_traj_list, dtype=object)
    # obs_expand_traj_arr = np.array(obs_expand_traj_list, dtype=object)
    # np.savez(
    #     "final_traj.npz",
    #     obs_traj_list=obs_traj_arr, # [(episode_length1, obs_dim),...]
    #     obs_expand_traj_list=obs_expand_traj_arr, # [(episode_length1, obs_dim+1),...]
    #     ret_list=ret_list, # [float,]
    #     true_traj=true_traj, # (batch, horizon, obs_dim)
    #     gene_traj=trajectories.observations, # (batch, horizon, obs_dim)
    #     gene_act_traj=trajectories.actions,  # (batch, horizon, act_dim)
    # )

    # # data = np.load("final_traj.npz", allow_pickle=True)
    # # obs_list = data['obs_traj_list'].tolist() # 还原成原来的列表结构


    # log_dict = {}
    # for key, value in eval_metrics.items():
    #     log_dict[key] = value
    # for key, value in traj_quality_metrics.items():
    #     log_dict[key] = value
    # for key, value in rollout_metrics.items():
    #     log_dict[key] = value
    # log_dict['TotalTime'] = total_time
    # log_dict['AvgStepTime'] = avg_per_step_time
    # log_dict = flatten_metrics(log_dict, check_horizon)
    # save_csv_native(log_dict, save_path="final_eval_metrics.csv")
