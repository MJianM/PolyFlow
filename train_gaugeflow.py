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


log = logging.getLogger(__name__)

def train_worker(cfg: DictConfig):


    device = cfg.device

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

    trainer.train(n_train_steps=cfg.iteration, use_cosine_scheduler=True, writer=writer)
    log.info("Training completed.")
    trainer.save("final")
    

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



@hydra.main(config_path="config", config_name="train_gaugeflow_hoppercpx.yaml")
def main(cfg: DictConfig):

    with Display(visible=0, size=(1024, 768), backend="xvfb") as disp:
        if "seed" in cfg:
            seed = cfg.seed
            set_all_seed(seed)
            log.info(f"Set random seed to: {seed}")

        train_worker(cfg)


if __name__ == "__main__":

    main()