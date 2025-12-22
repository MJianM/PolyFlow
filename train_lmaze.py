import numpy as np
import torch

from utils import set_all_seed
from dataset import ConstantConsTrajDataset
from train import train_discrete_delta, sample_discrete_delta
from visual import plot_results, plot_simple_loss

def train():

    seed = 42
    iters = 10000
    file_path = "L_maze_traj_data.npz"
    device = "cuda:0"
    steps = 10

    set_all_seed(seed)

    dataset = ConstantConsTrajDataset(file_path=file_path, seq_length=None) # 

    model, loss_history = train_discrete_delta(
        dataset=dataset, iteration=iters, batch_size=100, lr=1e-4, steps=steps, use_ot=True, use_one_ray=True,
        device=device
    )
    
    plot_simple_loss(loss_history, save_path=f'L_maze_train_loss_one2_{steps}.png')
    torch.save(model, "L_maze_final_model.pt")

    generated_traj = sample_discrete_delta(model, dataset, n_samples=2, steps=steps)[-1] # (n_samples, seq_length*x_dim)

    true_traj = dataset.sample_traj_data(n_sample=2)

    plot_results(true_traj=true_traj, gene_traj=generated_traj, fig_name=f'L_maze_result_one2_{steps}.png')

if __name__=="__main__":

    train()
