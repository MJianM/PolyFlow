import torch
import numpy as np
import os
from collections import namedtuple
from .normalization import DatasetNormalizer

# from collections import namedtuple
# import numpy as np
# import torch
# import pdb

# from .preprocessing import get_preprocess_fn
# from .d4rl import load_environment, sequence_dataset
# from .normalization import DatasetNormalizer
# from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
# ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

# class SequenceDataset(torch.utils.data.Dataset):

#     def __init__(self, env='hopper-medium-replay', horizon=64,
#         normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
#         max_n_episodes=10000, termination_penalty=0, use_padding=True):
#         self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)

#         self.env = env = load_environment(env)

#         self.horizon = horizon

#         self.max_path_length = max_path_length

#         self.use_padding = use_padding

#         itr = sequence_dataset(env, self.preprocess_fn)

#         fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)

#         for i, episode in enumerate(itr):
#             fields.add_path(episode)

#         fields.finalize()

#         self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])

#         # self.indices: np.ndarray, shape (num_samples, 3)
#         self.indices = self.make_indices(fields.path_lengths, horizon)

#         self.observation_dim = fields.observations.shape[-1]
#         self.action_dim = fields.actions.shape[-1]

#         # self.fields: ReplayBuffer
#         self.fields = fields
#         self.n_episodes = fields.n_episodes
#         self.path_lengths = fields.path_lengths

#         self.normalize()

#         print(fields)
#         # shapes = {key: val.shape for key, val in self.fields.items()}
#         # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

#     def normalize(self, keys=['observations', 'actions']):
#         '''
#             normalize fields that will be predicted by the diffusion model
#         '''
#         for key in keys:
#             # [N * T, dim]
#             # array shape: (n_episodes * max_path_length, dim)
#             array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)

#             # normed shape: (n_episodes * max_path_length, dim)
#             normed = self.normalizer(array, key)

#             # self.fields[f'normed_{key}'] shape: (n_episodes, max_path_length, dim)
#             self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

#     def make_indices(self, path_lengths, horizon):
#         '''
#             makes indices for sampling from dataset;
#             each index maps to a datapoint
#         '''
#         indices = []
#         for i, path_length in enumerate(path_lengths):
#             max_start = min(path_length - 1, self.max_path_length - horizon)
#             if not self.use_padding:
#                 max_start = min(max_start, path_length - horizon)
#             for start in range(max_start):
#                 end = start + horizon
#                 # (episode_index, start_timestep, end_timestep)
#                 indices.append((i, start, end))
#         # indices shape: (num_samples, 3)
#         indices = np.array(indices)
#         return indices

#     def get_conditions(self, observations):
#         '''
#             condition on current observation for planning
#         '''
#         return {0: observations[0]}

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx, eps=1e-4):
#         # path_ind: int (episode index), start: int, end: int
#         path_ind, start, end = self.indices[idx]

#         # observations shape: (horizon, observation_dim)
#         observations = self.fields.normed_observations[path_ind, start:end]
#         # actions shape: (horizon, action_dim)
#         actions = self.fields.normed_actions[path_ind, start:end]

#         # conditions: dict {timestep: observation}
#         conditions = self.get_conditions(observations)
#         # trajectories shape: (horizon, action_dim + observation_dim)
#         trajectories = np.concatenate([actions, observations], axis=-1)
#         batch = Batch(trajectories, conditions)
#         return batch

# class GoalDataset(SequenceDataset):

#     def get_conditions(self, observations):
#         '''
#             condition on both the current observation and the last observation in the plan
#         '''
#         return {
#             0: observations[0],
#             self.horizon - 1: observations[-1],
#         }

# class ValueDataset(SequenceDataset):
#     '''
#         adds a value field to the datapoints for training the value function
#     '''

#     def __init__(self, *args, discount=0.99, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.discount = discount
#         # self.discounts shape: (max_path_length, 1)
#         self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

#     def __getitem__(self, idx):
#         # batch: Batch(trajectories, conditions)
#         batch = super().__getitem__(idx)
#         path_ind, start, end = self.indices[idx]
#         # rewards shape: (length_of_segment, 1)
#         rewards = self.fields['rewards'][path_ind, start:]
#         # discounts shape: (length_of_segment, 1)
#         discounts = self.discounts[:len(rewards)]
#         # value shape: (1,)
#         value = (discounts * rewards).sum()
#         value = np.array([value], dtype=np.float32)
#         value_batch = ValueBatch(*batch, value)
#         return value_batch


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
                self.trajectories = data['traj_dataset']
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