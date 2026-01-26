import torch
import numpy as np
import math

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
from utils.eval import evaluate_dismatch_metrics, evaluate_trajectory_quality
from utils.logger import flatten_metrics, save_csv_native
from train import sample_worker

log = logging.getLogger(__name__)
# OmegaConf.register_new_resolver("abspath", lambda x: to_absolute_path(x))

def sample_and_eval(cfg: DictConfig):

    device = cfg.device
    
    log.info(f"Sampling Config:\n{OmegaConf.to_yaml(cfg)}")

    if hasattr(cfg.dataset, 'file_path'):
        cfg.dataset.file_path = to_absolute_path(cfg.dataset.file_path)
    if hasattr(cfg.sample, 'load_model_path'):
        cfg.sample.load_model_path = to_absolute_path(cfg.sample.load_model_path)

    log.info(f"Instantiating Dataset: {cfg.dataset._target_}")
    dataset = instantiate(cfg.dataset)
    
    log.info(f"Instantiating Env: {cfg.env._target_}")
    env = instantiate(cfg.env)

    log.info(f"Instantiating Model: {cfg.model._target_}")
    model = instantiate(cfg.model)
    model.to(device) 

    load_data = torch.load(cfg.sample.load_model_path, map_location=device)
    model.load_state_dict(load_data)

    model.eval()

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

    np.savez(
        file="sampled_traj.npz",
        generated_traj=generated_traj,
        true_traj=true_traj,
        seq_length=seq_length,
        x_dim=x_dim
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


@hydra.main(config_path="config", config_name="sample_safeflow_maze2d.yaml")
def main(cfg: DictConfig):


    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    log.info(f"Set random seed to: {seed}")

    sample_and_eval(cfg)


if __name__=="__main__":
    main()