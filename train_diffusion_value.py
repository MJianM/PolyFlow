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

from src.utils.eval import evaluate_dismatch_metrics, evaluate_trajectory_quality
from src.utils.logger import flatten_metrics, save_csv_native
from src.utils.arrays import apply_dict, set_all_seed
from src.utils.video import virtual_display


log = logging.getLogger(__name__)

def train_worker(cfg: DictConfig):


    device = cfg.device

    writer = SummaryWriter(log_dir=".")
    
    log.info(f"Training Config:\n{OmegaConf.to_yaml(cfg)}")

    if hasattr(cfg, 'file_path'):
        cfg.file_path = to_absolute_path(cfg.file_path)

    log.info(f"Instantiating Dataset: {cfg.dataset._target_}")
    dataset = instantiate(cfg.dataset)
    
    log.info(f"Instantiating Env Handler: {cfg.env._target_}")
    env_handler = instantiate(cfg.env)

    # log.info(f"Instantiating Eval DataLoader: {cfg.val_dataloader._target_}")
    # val_loader = instantiate(cfg.val_dataloader, dataset=dataset)

    log.info(f"Instantiating Backbone: {cfg.backbone._target_}")
    backbone = instantiate(cfg.backbone)

    log.info(f"Instantiating Diffusion: {cfg.algorithm._target_}")
    diffusion = instantiate(cfg.algorithm, model=backbone).to(device)
    # CRITICAL: Set normalization parameters for the safety check.
    if cfg.dataset.normalizer == 'GaussianNormalizer':
        diffusion.means = torch.from_numpy(dataset.normalizer.normalizers['observations'].means).to(device).float()
        diffusion.stds = torch.from_numpy(dataset.normalizer.normalizers['observations'].stds).to(device).float()
        diffusion.act_means = torch.from_numpy(dataset.normalizer.normalizers['actions'].means).to(device).float()
        diffusion.act_stds = torch.from_numpy(dataset.normalizer.normalizers['actions'].stds).to(device).float()
    else:
        diffusion.norm_mins = torch.from_numpy(dataset.normalizer.normalizers['observations'].mins).to(device).float()
        diffusion.norm_maxs = torch.from_numpy(dataset.normalizer.normalizers['observations'].maxs).to(device).float()
        diffusion.act_norm_mins = torch.from_numpy(dataset.normalizer.normalizers['actions'].mins).to(device).float()
        diffusion.act_norm_maxs = torch.from_numpy(dataset.normalizer.normalizers['actions'].maxs).to(device).float()    


    log.info(f"Instantiating Trainer: {cfg.trainer._target_}")
    trainer = instantiate(
        cfg.trainer, 
        diffusion_model=diffusion, 
        dataset=dataset,
        renderer=None,
        results_folder=".",
    )

    trainer.train(n_train_steps=cfg.iteration)
    log.info("Training completed.")
    trainer.save("final")
    


@hydra.main(config_path="config", config_name="train_diffusion_value_hopper.yaml")
def main(cfg: DictConfig):

    if "seed" in cfg:
        seed = cfg.seed
        set_all_seed(seed)
        log.info(f"Set random seed to: {seed}")

    train_worker(cfg)


if __name__ == "__main__":

    with virtual_display():
        main()