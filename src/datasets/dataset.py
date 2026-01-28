from collections import namedtuple
import numpy as np
import copy
import torch
import pdb
import minari

from .preprocessing import get_preprocess_fn
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class MinariSequenceDataset(torch.utils.data.Dataset):
    """
    MinariSequenceDataset
    """
    def __init__(self, env=['mujoco/hopper/medium-v0'], horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        """
        MinariSequenceDataset

        参数:
            env (list[str]): Minari 数据集名称的列表 (例如 ['mujoco/hopper/medium-v0', 'mujoco/hopper/expert-v0'])。
            horizon (int): 采样的轨迹片段长度 (T)。
            normalizer (str): 归一化器类型。
            preprocess_fns (list): 预处理函数列表。
            max_path_length (int): 最大路径长度。
            max_n_episodes (int): 加载的最大 episode 总数量 (所有数据集之和)。
            termination_penalty (float): 提前终止的惩罚值。
            use_padding (bool): 是否使用填充。
            seed (int): 随机种子。
        """

        if isinstance(env, str):
            env_list = [env]
        else:
            env_list = env
        
        self.env_list = env_list
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding


        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        total_episodes_loaded = 0 


        for env_name in env_list:
            if total_episodes_loaded >= max_n_episodes:
                print(f'[ datasets/sequence ] Reached max total episodes: {max_n_episodes}')
                break
            
            print(f'[ datasets/sequence ] Loading dataset: {env_name}')


            current_preprocess_fn = get_preprocess_fn(preprocess_fns, env_name)


            minari_dataset = minari.load_dataset(env_name)
            

            itr = minari_dataset.iterate_episodes()


            for i, episode in enumerate(itr):

                if total_episodes_loaded >= max_n_episodes:
                    break


                assert episode.observations.shape[0] == episode.actions.shape[0] + 1, \
                    f'Observations length {episode.observations.shape[0]} != Actions length {episode.actions.shape[0]} + 1'
                
                path = {
                    'observations': episode.observations[:-1],
                    'actions': episode.actions,
                    'rewards': episode.rewards.reshape(-1, 1),
                    'terminals': episode.terminations.reshape(-1, 1),
                    'timeouts': episode.truncations.reshape(-1, 1),
                }
                

                path = current_preprocess_fn(path)
                

                fields.add_path(path)
                
                total_episodes_loaded += 1
                
        fields.finalize()
        print(f'[ datasets/sequence ] Total episodes loaded: {total_episodes_loaded}')


        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        

        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        

        self.normalize()

        print(fields)

    def normalize(self, keys=['observations', 'actions']):

        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
             (path_ind, start, end)
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):

            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
            
            参数:
                observations: [horizon, observation_dim] 
            
            返回:
                dict: {0: observation[0]}
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        """
        
        参数:
            idx (int)
            
        返回:
            batch (Batch): 包含 trajectories (动作+观测) 和 conditions 的命名元组。
                - trajectories: [horizon, action_dim + observation_dim]
                - conditions: dict {timestep: observation}
        """
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)

        if torch.is_tensor(actions):
            # GPU/Tensor 
            trajectories = torch.cat([actions, observations], dim=-1)
        else:
            # CPU/Numpy 
            trajectories = np.concatenate([actions, observations], axis=-1)

        batch = Batch(trajectories, conditions)
        return batch


class ValueDataset(MinariSequenceDataset):

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        # [1, gamma, gamma^2, ..., gamma^max_path_length]
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):

        batch = super().__getitem__(idx)
        

        path_ind, start, end = self.indices[idx]
        

        rewards = self.fields['rewards'][path_ind, start:]
        

        discounts = self.discounts[:len(rewards)]
        
        # Discounted Return: Sum(gamma^k * r_{t+k})
        value = (discounts * rewards).sum()
        
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        

        # trajectories: [horizon, transition_dim]
        # conditions: dict
        # values: [1]
        value_batch = ValueBatch(*batch, value)
        return value_batch
    

if __name__ == '__main__':

    value_dataset = ValueDataset(
        env='mujoco/walker2d/medium-v0',
        horizon=16,
        max_n_episodes=50,
        normed=True
    )

    batch = value_dataset[0]
    print(f'Batch trajectories shape: {batch.trajectories.shape}')
    print(f'Batch conditions keys: {batch.conditions}')
    print(f'Batch value shape: {batch.values}')



