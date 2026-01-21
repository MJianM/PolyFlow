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
    MinariSequenceDataset 用于从 Minari 数据集加载和处理序列数据。
    它从 minari 加载数据，存储在 ReplayBuffer 中，并支持按固定 horizon 采样轨迹片段。
    """
    def __init__(self, env=['mujoco/hopper/medium-v0'], horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        """
        初始化 MinariSequenceDataset。

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
        # 兼容性处理：如果用户传的是单个字符串，转为列表
        if isinstance(env, str):
            env_list = [env]
        else:
            env_list = env
        
        self.env_list = env_list
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        # 初始化 ReplayBuffer (只初始化一次，用于存放所有环境的数据)
        # 注意：ReplayBuffer 通常会在第一次 add_path 时确定维度，因此请确保 env_list 中的所有环境维度一致
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        total_episodes_loaded = 0 # 全局计数器

        # --- 1. 遍历数据集列表 ---
        for env_name in env_list:
            if total_episodes_loaded >= max_n_episodes:
                print(f'[ datasets/sequence ] Reached max total episodes: {max_n_episodes}')
                break
            
            print(f'[ datasets/sequence ] Loading dataset: {env_name}')

            # 针对当前环境获取预处理函数 (假设 get_preprocess_fn 依赖环境名)
            current_preprocess_fn = get_preprocess_fn(preprocess_fns, env_name)

            # 加载 Minari 数据集
            minari_dataset = minari.load_dataset(env_name)
            
            # 从 Minari 数据集创建迭代器
            itr = minari_dataset.iterate_episodes()

            # --- 2. 遍历当前数据集的 episodes ---
            for i, episode in enumerate(itr):
                # 检查全局总数限制
                if total_episodes_loaded >= max_n_episodes:
                    break

                # 将 Minari episode 格式转换为 ReplayBuffer 期望的字典格式
                # 注意：Minari 使用 'terminations' 和 'truncations'，我们将其映射到 'terminals' 和 'timeouts'
                # minari 中 observations 序列长度比 actions 多 1
                assert episode.observations.shape[0] == episode.actions.shape[0] + 1, \
                    f'Observations length {episode.observations.shape[0]} != Actions length {episode.actions.shape[0]} + 1'
                
                path = {
                    'observations': episode.observations[:-1],
                    'actions': episode.actions,
                    'rewards': episode.rewards.reshape(-1, 1),
                    'terminals': episode.terminations.reshape(-1, 1),
                    'timeouts': episode.truncations.reshape(-1, 1),
                }
                
                # 使用当前环境对应的预处理函数
                path = current_preprocess_fn(path)
                
                # 添加到统一的 Buffer 中
                fields.add_path(path)
                
                total_episodes_loaded += 1
                
        fields.finalize()
        print(f'[ datasets/sequence ] Total episodes loaded: {total_episodes_loaded}')

        # 初始化归一化器 (基于合并后的数据)
        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        
        # 创建采样索引
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        
        # 对所有数据进行归一化
        self.normalize()

        print(fields)

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
            对观测和动作进行归一化，并将结果存储在 fields 中 (例如 'normed_observations')。
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
            创建用于采样的索引列表。每个索引是一个元组 (path_ind, start, end)。
            
            参数:
                path_lengths (np.array): 每个 episode 的长度。
                horizon (int): 采样的时间步长。
            
            返回:
                indices (np.array): 形状为 [N, 3] 的数组，包含所有有效的采样片段索引。
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            # 计算该 episode 允许的最大起始步
            # self.max_path_length - horizon 确保不会超出 buffer 的固定大小
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                # 如果不使用填充，起始步必须保证剩余长度至少为 horizon
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
            获取条件数据。默认情况下，仅以当前观测 (t=0) 作为条件。
            
            参数:
                observations: [horizon, observation_dim] 的观测序列。
            
            返回:
                dict: {0: observation[0]}
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        """
        获取一个数据样本。
        
        参数:
            idx (int): 样本索引。
            
        返回:
            batch (Batch): 包含 trajectories (动作+观测) 和 conditions 的命名元组。
                - trajectories: [horizon, action_dim + observation_dim]
                - conditions: dict {timestep: observation}
        """
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)

        # ---------------- [修改开始] ----------------
        # 自动判断使用 torch.cat 还是 np.concatenate
        if torch.is_tensor(actions):
            # GPU/Tensor 模式
            trajectories = torch.cat([actions, observations], dim=-1)
        else:
            # CPU/Numpy 模式
            trajectories = np.concatenate([actions, observations], axis=-1)
        # ---------------- [修改结束] ----------------

        batch = Batch(trajectories, conditions)
        return batch


class ValueDataset(MinariSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
        ValueDataset 用于为训练 Value Function 提供数据。
        它不仅提供轨迹片段，还计算该片段起始状态的折扣回报（Discounted Return）作为标签。
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        # 预计算折扣因子序列 [1, gamma, gamma^2, ..., gamma^max_path_length]
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        # 如果需要归一化价值，则计算数据集中的最大和最小价值
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
        # 调用父类获取基础的轨迹数据 (trajectories) 和条件 (conditions)
        # batch 类型为 Batch(trajectories, conditions)
        batch = super().__getitem__(idx)
        
        # 获取当前样本在原始 buffer 中的索引信息
        # path_ind: 第几条轨迹, start: 轨迹片段的起始步, end: 轨迹片段的结束步
        path_ind, start, end = self.indices[idx]
        
        # 获取从当前 start 时刻开始，直到整条轨迹结束的所有奖励
        # 注意：这里不仅仅是 horizon 长度内的奖励，而是直到 episode 结束，用于计算真实的 Return
        rewards = self.fields['rewards'][path_ind, start:]
        
        # 获取对应的折扣因子序列，长度与剩余奖励长度一致
        discounts = self.discounts[:len(rewards)]
        
        # 计算折扣回报 (Discounted Return): Sum(gamma^k * r_{t+k})
        value = (discounts * rewards).sum()
        
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        
        # 返回包含 value 的增强 Batch
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

    # # ----------------------------------------------------
    # # MinariSequenceDataset 验证代码
    # # ----------------------------------------------------
    # print("\n\n----------- Testing MinariSequenceDataset -----------")
    # try:
    #     # 1. 初始化 MinariSequenceDataset
    #     print("\n[1] Initializing MinariSequenceDataset with 'pointmaze-umaze-v1'...")
    #     # 为了快速测试，我们只加载少量 episodes
    #     minari_dataset = MinariSequenceDataset(
    #         env='mujoco/walker2d/medium-v0',
    #         horizon=16,
    #         max_n_episodes=50
    #     )
    #     print("      Initialization successful.")

    #     # 2. 检查数据集属性
    #     print("\n[2] Checking dataset properties...")
    #     print(f"      - Number of samples: {len(minari_dataset)}")
    #     print(f"      - Observation dimension: {minari_dataset.observation_dim}")
    #     print(f"      - Action dimension: {minari_dataset.action_dim}")
    #     assert len(minari_dataset) > 0, "Dataset should not be empty."

    #     # 3. 获取单个样本并验证其形状
    #     print("\n[3] Getting a single sample (dataset[0])...")
    #     sample_batch = minari_dataset[0]
    #     trajectories = sample_batch.trajectories
    #     conditions = sample_batch.conditions
    #     print(f"      - Trajectories shape: {trajectories.shape}")
    #     print(f"      - Conditions keys: {conditions.keys()}")
    #     print(f"      - Condition at t=0 shape: {conditions[0].shape}")

    #     # 验证形状是否符合预期
    #     expected_traj_shape = (minari_dataset.horizon, minari_dataset.action_dim + minari_dataset.observation_dim)
    #     assert trajectories.shape == expected_traj_shape, f"Expected trajectory shape {expected_traj_shape}, but got {trajectories.shape}"
    #     assert conditions[0].shape == (minari_dataset.observation_dim,), f"Expected condition shape {(minari_dataset.observation_dim,)}, but got {conditions[0].shape}"
    #     print("      Sample shapes are correct.")

    #     # 4. 使用 DataLoader 进行批量加载测试
    #     print("\n[4] Testing with torch.utils.data.DataLoader...")
    #     data_loader = torch.utils.data.DataLoader(minari_dataset, batch_size=4, shuffle=True)
    #     batch_from_loader = next(iter(data_loader))
    #     loader_trajectories = batch_from_loader.trajectories
    #     loader_conditions_t0 = batch_from_loader.conditions[0]

    #     print(f"      - Batch trajectories shape: {loader_trajectories.shape}")
    #     print(f"      - Batch conditions at t=0 shape: {loader_conditions_t0.shape}")
    #     assert loader_trajectories.shape[0] == 4, "Batch size should be 4."
    #     print("      DataLoader batch shapes are correct.")

    #     print("\n----------- MinariSequenceDataset validation successful! -----------")

    # except ImportError:
    #     print("\n[ERROR] Could not run validation. Please install minari: `pip install minari`")
    # except Exception as e:
    #     print(f"\n[ERROR] An unexpected error occurred during validation: {e}")

