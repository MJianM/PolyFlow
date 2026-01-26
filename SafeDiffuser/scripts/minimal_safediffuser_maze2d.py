from typing import List
from collections import namedtuple
import torch
import numpy as np
import math
from pathlib import Path
import os
import sys
parent_parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_parent_dir))
import diffuser.utils as utils
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.models.temporal import TemporalUnet
from diffuser.models.diffusion import GaussianDiffusion, SafeGaussianDiffusion
from diffuser.guides.policies import Policy
import gymnasium as gym
import gymnasium_robotics

Batch = namedtuple('Batch', 'trajectories conditions')

Rectangle = namedtuple('Rectangle', ['r_min', 'c_min', 'r_max', 'c_max'])

class MazeObs:
    def __init__(self, 
                 maze, 
                 rect_list: List[Rectangle],
                 obs_expand_dis = 0.2,
                 ellips_n = 4,
                 alpha: float = 0.5):

        assert alpha >= 0 and alpha <= 1

        self.alpha = alpha
        self.rect_list = rect_list
        self.maze = maze
        self.obs_expand_dis = obs_expand_dis
        self.ellips_n = ellips_n

        # CBF form: (x-x_c)^2/a^2 + (y-y_c)^2/b^2 -1 >= 0
        self.ellips_list = self.create_ellips_list()

    def create_ellips_list(self):

        ellips_list = []
        for rect in self.rect_list:
            p_min = self.maze.cell_rowcol_to_xy(np.array([rect.r_min, rect.c_min]))
            p_max = self.maze.cell_rowcol_to_xy(np.array([rect.r_max, rect.c_max]))

            half_scale = self.maze.maze_size_scaling * 0.5
            x_min, x_max = p_min[0] - half_scale - self.obs_expand_dis, p_max[0] + half_scale + self.obs_expand_dis
            y_min, y_max = p_max[1] - half_scale - self.obs_expand_dis, p_min[1] + half_scale + self.obs_expand_dis

            x_center = (x_min + x_max) * 0.5
            y_center = (y_min + y_max) * 0.5

            x_length = x_max - x_min
            y_length = y_max - y_min

            a_in = x_length * 0.5
            b_in = y_length * 0.5
            a_out = x_length * 0.5 * math.pow(2, 1.0 / self.ellips_n)
            b_out = y_length * 0.5 * math.pow(2, 1.0 / self.ellips_n)
            a = a_in + (a_out - a_in) * self.alpha
            b = b_in + (b_out - b_in) * self.alpha

            ellips_list.append([x_center, y_center, a, b, self.ellips_n])

        return ellips_list

    def get_ellips_list(self):

        return self.ellips_list

# -----------------------------------------------------------------------------
# 2. Custom Dataset Class for .npz files
# -----------------------------------------------------------------------------
class NpzGoalDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, action_dim, obs_dim, horizon, normalizer='LimitsNormalizer', max_n_episodes=10000):
        self.action_dim = action_dim
        self.observation_dim = obs_dim
        self.horizon = horizon
        
        # Load data from .npz file
        if os.path.exists(data_path):
            print(f"Loading data from {data_path}")
            with np.load(data_path) as data:
                # Expected shape: (num_traj, seq_length, action_dim + obs_dim)
                self.trajectories = data['traj']
        else:
            print(f"Warning: {data_path} not found. Generating dummy data for demonstration.")
            # Dummy data generation
            self.trajectories = np.random.randn(100, 1000, action_dim + obs_dim).astype(np.float32)

        if len(self.trajectories) > max_n_episodes:
            self.trajectories = self.trajectories[:max_n_episodes]

        # Split into actions and observations
        # Assuming the format is [actions, observations] in the last dimension
        self.actions = self.trajectories[:, :, :action_dim]
        self.observations = self.trajectories[:, :, action_dim:]
        
        # Flatten for normalization (Normalizer expects [N, dim])
        if action_dim == 0:
            self.actions_flat = self.actions.flatten()
        else:
            self.actions_flat = self.actions.reshape(-1, action_dim)
        self.observations_flat = self.observations.reshape(-1, obs_dim)
        
        # Initialize Normalizer
        data_dict = {
            'actions': self.actions_flat,
            'observations': self.observations_flat
        }
        self.normalizer = DatasetNormalizer(data_dict, normalizer)
    
        # Normalize data and reshape back to (num_traj, seq_length, dim)
        self.normed_actions = self.normalizer.normalize(self.actions_flat, 'actions').reshape(self.actions.shape)
        self.normed_observations = self.normalizer.normalize(self.observations_flat, 'observations').reshape(self.observations.shape)
        
        # Reconstruct trajectories: [actions, observations]
        self.normed_trajectories = np.concatenate([self.normed_actions, self.normed_observations], axis=2)
    
        # Create indices for sampling sub-trajectories of length `horizon`
        self.indices = []
        num_traj, seq_len, _ = self.trajectories.shape
        for i in range(num_traj):
            max_start = seq_len - self.horizon
            if max_start >= 0:
                for start in range(max_start + 1):
                    end = start + self.horizon
                    self.indices.append((i, start, end))
        
        print(f"Dataset loaded. Total samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def get_conditions(self, observations):
        # Condition on start (t=0) and goal (t=horizon-1)
        # observations shape: (horizon, obs_dim)
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }
        # return None

    def __getitem__(self, idx):
        traj_idx, start, end = self.indices[idx]
        
        # Get trajectory segment
        segment = self.normed_trajectories[traj_idx, start:end]
        
        # Extract observations for conditioning (observations are at the end)
        observations = segment[:, self.action_dim:]
        
        conditions = self.get_conditions(observations)
        
        # Return Batch namedtuple (trajectories, conditions)
        return Batch(segment, conditions)


def main():

    # Load environment for testing
    # Use gymnasium.robotics standard interface
    # Note: Requires 'gymnasium-robotics' installed
    # We use PointMaze_Large-v3 as the equivalent to maze2d-large-v1
    LARGE_MAZE =   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 'g', 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 'r', 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    gym.register_envs(gymnasium_robotics)
    env = gym.make('PointMaze_Large-v3', maze_map=LARGE_MAZE, continuing_task=False, reset_target=False, max_episode_steps=1000,
                render_mode='rgb_array')
    maze = env.unwrapped.maze

    # (r_min, c_mn, r_max, c_max)
    rect_list = [
        Rectangle(2, 2, 2, 3),
        Rectangle(1, 5, 2, 5),
        Rectangle(4, 4, 4, 5),
        Rectangle(5, 5, 6, 5),
        Rectangle(3, 7, 4, 7),
        Rectangle(4, 8, 4, 9),
        Rectangle(6, 7, 7, 7),
        Rectangle(6, 9, 6, 10),
    ]
    obs_expand_dis = 0.2

    maze_obs = MazeObs(
        maze=maze, rect_list=rect_list, alpha=0.5, obs_expand_dis=obs_expand_dis, ellips_n=4,
    )

    # -----------------------------------------------------------------------------
    # 3. Configuration
    # -----------------------------------------------------------------------------
    config = {
        'dataset': 'maze2d-large-v1', # Environment name for renderer
        'data_path': 'data.npz',      # Path to custom .npz dataset
        'action_dim': 0,              # Action dimension for maze2d
        'obs_dim': 4,                 # Observation dimension for maze2d
        'horizon': 300,              # Trajectory length
        'n_diffusion_steps': 256,    # Number of denoising steps
        'action_weight': 0,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),      # U-Net dimension multipliers
        'batch_size': 32,
        'learning_rate': 2e-4,
        'n_train_steps': 200,        # Reduced for minimal demo (use ~2e6 for real training)
        'savepath': 'logs/minimal_safediffuser_maze2d',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'safe_method': 'ReS',     # none, truncate, classifier_guidance, RoS, RoS_cf, ReS, TVS
        'use_condition': False,
    }
    
    print(f"Running SafeDiffuser minimal implementation on {config['device']}...")
    if not os.path.exists(config['savepath']):
        os.makedirs(config['savepath'])

    # -----------------------------------------------------------------------------
    # 4. Dataset & Model Setup
    # -----------------------------------------------------------------------------
    print("Loading dataset and environment...")
    
    # Use the new NpzGoalDataset
    dataset = NpzGoalDataset(
        data_path=config['data_path'],
        action_dim=config['action_dim'],
        obs_dim=config['obs_dim'],
        horizon=config['horizon'],
        normalizer='LimitsNormalizer',
    )
    
    observation_dim = config['obs_dim']
    action_dim = config['action_dim']

    print("Initializing models...")
    # Temporal U-Net model (the noise predictor)
    model = TemporalUnet(
        horizon=config['horizon'],
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        dim_mults=config['dim_mults'],
    )
    model.to(config['device'])

    # SafeGaussianDiffusion (The diffusion process wrapper)
    diffusion = SafeGaussianDiffusion(
        model=model,
        horizon=config['horizon'],
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=config['n_diffusion_steps'],
        loss_type='l2',
        clip_denoised=True,
        predict_epsilon=config['predict_epsilon'],
        action_weight=config['action_weight'],
        loss_weights=config['loss_weights'],
        loss_discount=config['loss_discount'],
        ellips_list=maze_obs.get_ellips_list(),
        safe_method=config['safe_method']
    )
    diffusion.to(config['device'])

    # CRITICAL: Set normalization parameters for the safety check.
    # The 'invariance' method in diffusion.py relies on self.norm_mins/maxs 
    # to normalize coordinates for obstacle checking.
    # Note: dataset.normalizer is a DatasetNormalizer, we need the specific normalizer for observations
    diffusion.norm_mins = torch.from_numpy(dataset.normalizer.normalizers['observations'].mins).to(config['device']).float()
    diffusion.norm_maxs = torch.from_numpy(dataset.normalizer.normalizers['observations'].maxs).to(config['device']).float()

    # -----------------------------------------------------------------------------
    # 5. Training Loop
    # -----------------------------------------------------------------------------
    print(f"Starting training for {config['n_train_steps']} steps...")
    trainer = utils.Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        renderer=None,
        use_condition=config['use_condition'],
        train_batch_size=config['batch_size'],
        train_lr=config['learning_rate'],
        results_folder=config['savepath'],
        save_freq=5000,
        log_freq=1000,
        label_freq=5000,
    )
    
    # Run the training loop
    trainer.train(n_train_steps=config['n_train_steps'])
    print("Training complete.")

    # -----------------------------------------------------------------------------
    # 6. Safe Sampling / Planning
    # -----------------------------------------------------------------------------
    print("Starting safe sampling (planning)...")
    

    
    # Policy wrapper: handles conditioning and calling the diffusion model
    policy = Policy(diffusion, dataset.normalizer)
    
    # Set up a specific test case (Start -> Target)
    # Using coordinates from plan_maze2d.py for consistency
    obs, info = env.reset()
    start_pos = obs['observation'][:2]
    start_vel = obs['observation'][2:4]
    # env.unwrapped.set_state(start_pos, start_vel)
    
    target = obs['desired_goal']
    print(f"Planning from {start_pos} to {target}")

    # Define conditions: Fix start (t=0) and Goal (t=horizon-1)
    # cond = {
    #     0: np.concatenate([start_pos, start_vel]),
    #     config['horizon'] - 1: np.array([*target, 0, 0]),
    # }
    cond = None
    
    # Run sampling
    # Note: batch_size=1 is required for the current 'invariance' implementation
    action, samples, diffusion_paths, elbo = policy(cond, batch_size=1)
    
    # -----------------------------------------------------------------------------
    # 7. Save Results
    # -----------------------------------------------------------------------------
    # samples.observations is [batch, horizon, obs_dim]
    trajectory = samples.observations
    
    print(f"Plan generated. Trajectory shape: {trajectory.shape}")
    
    # Save visualization
    # TODO

if __name__ == "__main__":
    main()