# from collections import namedtuple
# import numpy as np
# import torch
# import pdb

# from .preprocessing import get_preprocess_fn
# from .d4rl import load_environment, sequence_dataset
# from .normalization import DatasetNormalizer
# from .buffer import ReplayBuffer

# # 定义 Batch 命名元组，包含轨迹和条件
# Batch = namedtuple('Batch', 'trajectories conditions')
# # 定义 ValueBatch 命名元组，包含轨迹、条件和价值
# ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

# class SequenceDataset(torch.utils.data.Dataset):

#     def __init__(self, env='hopper-medium-replay', horizon=64,
#         normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
#         max_n_episodes=10000, termination_penalty=0, use_padding=True):
#         # 获取预处理函数
#         # self.preprocess_fn: function, 用于处理环境返回的数据
#         self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
#         # 加载环境
#         # self.env: gym.Env, 强化学习环境
#         self.env = env = load_environment(env)
#         # 设置时间视界（horizon），即每次采样的序列长度
#         # self.horizon: int, 扩散模型生成或采样的轨迹长度
#         self.horizon = horizon
#         # 设置最大路径长度
#         # self.max_path_length: int, 数据集中轨迹的最大长度
#         self.max_path_length = max_path_length
#         # 是否使用填充
#         # self.use_padding: bool, 是否允许采样超出实际轨迹长度（使用填充）
#         self.use_padding = use_padding
#         # 获取序列数据集迭代器
#         # itr: generator, 产生每条轨迹的数据字典
#         itr = sequence_dataset(env, self.preprocess_fn)

#         # 初始化回放缓冲区
#         # fields: ReplayBuffer 对象, 存储所有episode的数据
#         # fields.observations shape: (max_n_episodes, max_path_length, observation_dim)
#         fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
#         # 将每一集数据添加到缓冲区
#         for i, episode in enumerate(itr):
#             fields.add_path(episode)
#         # 完成缓冲区构建
#         fields.finalize()

#         # 初始化数据集归一化器
#         # self.normalizer: DatasetNormalizer, 用于归一化和反归一化数据
#         self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
#         # 创建索引，用于从数据集中采样
#         # self.indices: np.ndarray, shape (num_samples, 3), 每一行是 (episode_idx, start_idx, end_idx)
#         self.indices = self.make_indices(fields.path_lengths, horizon)

#         # 获取观测维度和动作维度
#         self.observation_dim = fields.observations.shape[-1]
#         self.action_dim = fields.actions.shape[-1]
#         # 保存字段数据
#         # self.fields: ReplayBuffer, 包含原始数据和归一化后的数据
#         self.fields = fields
#         self.n_episodes = fields.n_episodes
#         self.path_lengths = fields.path_lengths
#         # 对数据进行归一化
#         self.normalize()

#         print(fields)
#         # shapes = {key: val.shape for key, val in self.fields.items()}
#         # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

#     def normalize(self, keys=['observations', 'actions']):
#         '''
#             normalize fields that will be predicted by the diffusion model
#             归一化将被扩散模型预测的字段
#         '''
#         for key in keys:
#             # 将数据重塑为二维数组 [N * T, dim]
#             # array shape: (n_episodes * max_path_length, dim)
#             array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
#             # 使用归一化器进行归一化
#             # normed shape: (n_episodes * max_path_length, dim)
#             normed = self.normalizer(array, key)
#             # 将归一化后的数据重塑回三维数组 [N, T, dim] 并保存
#             # self.fields[f'normed_{key}'] shape: (n_episodes, max_path_length, dim)
#             self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

#     def make_indices(self, path_lengths, horizon):
#         '''
#             makes indices for sampling from dataset;
#             each index maps to a datapoint
#             创建用于从数据集采样的索引；每个索引映射到一个数据点
#         '''
#         indices = []
#         for i, path_length in enumerate(path_lengths):
#             # 计算最大起始索引
#             # max_start: int, 轨迹中最后一个可以作为起点的索引
#             max_start = min(path_length - 1, self.max_path_length - horizon)
#             if not self.use_padding:
#                 max_start = min(max_start, path_length - horizon)
#             # 生成每个可能的起始点的索引
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
#             基于当前观测进行规划的条件
#         '''
#         # 默认只以第一个观测作为条件
#         # 返回 dict: {timestep: observation}
#         return {0: observations[0]}

#     def __len__(self):
#         # 返回数据集大小
#         return len(self.indices)

#     def __getitem__(self, idx, eps=1e-4):
#         # 获取索引对应的路径索引、起始和结束位置
#         # path_ind: int (episode index), start: int, end: int
#         path_ind, start, end = self.indices[idx]

#         # 获取归一化后的观测和动作片段
#         # observations shape: (horizon, observation_dim)
#         observations = self.fields.normed_observations[path_ind, start:end]
#         # actions shape: (horizon, action_dim)
#         actions = self.fields.normed_actions[path_ind, start:end]

#         # 获取条件
#         # conditions: dict {timestep: observation}
#         conditions = self.get_conditions(observations)
#         # 将动作和观测拼接成轨迹 [horizon, action_dim + obs_dim]
#         # trajectories shape: (horizon, action_dim + observation_dim)
#         trajectories = np.concatenate([actions, observations], axis=-1)
#         # 创建 Batch 对象
#         batch = Batch(trajectories, conditions)
#         return batch

# class GoalDataset(SequenceDataset):

#     def get_conditions(self, observations):
#         '''
#             condition on both the current observation and the last observation in the plan
#             基于当前观测和计划中的最后一个观测作为条件
#         '''
#         # 以第一个和最后一个观测作为条件
#         # 返回 dict: {timestep: observation}
#         return {
#             0: observations[0],
#             self.horizon - 1: observations[-1],
#         }

# class ValueDataset(SequenceDataset):
#     '''
#         adds a value field to the datapoints for training the value function
#         为数据点添加价值字段，用于训练价值函数
#     '''

#     def __init__(self, *args, discount=0.99, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.discount = discount
#         # 预计算折扣因子
#         # self.discounts shape: (max_path_length, 1)
#         self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

#     def __getitem__(self, idx):
#         # 获取基础 batch
#         # batch: Batch(trajectories, conditions)
#         batch = super().__getitem__(idx)
#         path_ind, start, end = self.indices[idx]
#         # 获取从当前步开始的奖励
#         # rewards shape: (length_of_segment, 1)
#         rewards = self.fields['rewards'][path_ind, start:]
#         # 获取对应的折扣因子
#         # discounts shape: (length_of_segment, 1)
#         discounts = self.discounts[:len(rewards)]
#         # 计算折扣回报（价值）
#         # value shape: (1,)
#         value = (discounts * rewards).sum()
#         value = np.array([value], dtype=np.float32)
#         # 创建 ValueBatch 对象
#         value_batch = ValueBatch(*batch, value)
#         return value_batch
