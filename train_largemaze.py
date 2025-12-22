import numpy as np
import torch

from utils import set_all_seed
from dataset import BoxConsTrajDataset
from train import train_discrete_delta, sample_discrete_delta
from visual import plot_simple_loss, plot_trajectory_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import gymnasium as gym
import gymnasium_robotics


def train():

    seed = 42
    iters = 15000
    file_path = "large_maze_traj_data_expand_02.npz"
    device = "cuda:0"
    steps = 10

    set_all_seed(seed)


    # maze2d 轨迹长度300

    dataset = BoxConsTrajDataset(file_path=file_path, seq_length=None) # 

    model, loss_history = train_discrete_delta(
        dataset=dataset, iteration=iters, batch_size=100, lr=1e-4, steps=steps, use_ot=True, use_one_ray=True,
        device=device
    )
    
    plot_simple_loss(loss_history, save_path=f'large_maze_expand02_train_loss_one_{steps}.png')
    torch.save(model, "large_maze_expand02_final_model.pt")

    generated_traj = sample_discrete_delta(model, dataset, n_samples=5, steps=steps)[-1] # (n_samples, seq_length*x_dim)

    true_traj = dataset.sample_traj_data(n_sample=2) # (n_samples, seq_length*x_dim)


    seq_length = dataset.seq_length
    x_dim = dataset.x_dim

    LARGE_MAZE_TRUE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    env = gym.make('PointMaze_Large-v3', maze_map=LARGE_MAZE_TRUE, continuing_task=False, reset_target=False, max_episode_steps=1000,
                render_mode='rgb_array')
    env_maze = env.unwrapped.maze

    plot_trajectory_comparison(
        env_maze=env_maze, 
        true_trajs=true_traj.reshape(-1, seq_length, 2), 
        gene_trajs=generated_traj.reshape(-1, seq_length, 2), 
        obs_expand_dis=0.2,
        max_plot=100, 
        save_path=f"large_maze_expand02_result_one_{steps}.png")


if __name__=="__main__":

    train()
