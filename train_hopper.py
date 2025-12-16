import numpy as np
import torch

from utils import set_all_seed
from dataset import ConstantConsTrajDataset
from train import train_discrete_delta, sample_discrete_delta
from visual import plot_results, plot_simple_loss

def train():

    seed = 42
    iters = 10000
    file_path = "xxx"

    set_all_seed(seed)

    dataset = ConstantConsTrajDataset(file_path=file_path, seq_length=200) # 把轨迹降采样到200

    model, loss_history = train_discrete_delta(
        dataset=dataset, iteration=iters, batch_size=100, lr=1e-4, steps=10, use_ot=True
    )
    
    plot_simple_loss(loss_history, save_path='train_loss.png')
    torch.save(model, "final_model.pt")

    generated_traj = sample_discrete_delta(model, dataset, n_samples=10, steps=10)[-1] # (n_samples, seq_length*x_dim)

    # eva
    #TODO
    #使用gym mujoco render功能，不进行仿真step,只做可视化
    #TODO
    #把高度曲线画出来