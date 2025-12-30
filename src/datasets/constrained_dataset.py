from collections import namedtuple
import numpy as np
import torch
import pdb
import minari

from .preprocessing import get_preprocess_fn
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

from .dataset import MinariSequenceDataset
from src.utils.chebyshev_center import chebyshev_center_lp, uniform_sample_in_ball

# 定义包含约束信息的 Batch
ConstrainedBatch = namedtuple('ConstrainedBatch', 'trajectories conditions A b')

class ConstrainedMinariDataset(MinariSequenceDataset):
    """
    MinariSequenceDataset 的子类，支持多面体约束 A x <= b。
    会自动将物理空间的约束 A, b 转换为归一化空间的约束，以匹配归一化后的观测数据。
    """

    def __init__(self, obs_constrained_idx, single_A, single_b, *args, **kwargs):
        """
        参数:
            obs_constrained_idx (list/array): 观测向量中参与约束的维度索引 (例如 [0, 1] 表示 x, y)。
            single_A (np.ndarray): 约束矩阵 A，形状 (num_cons, len(obs_constrained_idx))。
            single_b (np.ndarray): 约束边界 b，形状 (num_cons,)。
            *args, **kwargs: 传递给 MinariSequenceDataset 的其他参数。
        """
        super().__init__(*args, **kwargs)

        self.obs_constrained_idx = np.array(obs_constrained_idx, dtype=int)
        
        # 存储原始约束 (物理空间)
        self.raw_A = np.array(single_A, dtype=np.float32)
        self.raw_b = np.array(single_b, dtype=np.float32)
        
        # 验证维度
        assert self.raw_A.shape[1] == len(self.obs_constrained_idx), \
            f"A matrix columns ({self.raw_A.shape[1]}) must match constrained dims ({len(self.obs_constrained_idx)})"

        # 计算归一化后的约束 (Norm Space)
        self.norm_A, self.norm_b = self._normalize_constraints()

        # 预计算 chebyshev center
        # 1. 计算 Chebyshev 中心和半径 (基于归一化后的约束)
        # center: (sub_dim,), radius: scalar
        self.center, self.radius = chebyshev_center_lp(self.norm_A, self.norm_b)

        print("---------------------")
        print(f"norm A:\n {self.norm_A}")
        print(f"norm b:\n {self.norm_b}")
        obs_normalizer = self.normalizer.normalizers['observations']
        obs_means = obs_normalizer.means[self.obs_constrained_idx]
        obs_stds = obs_normalizer.stds[self.obs_constrained_idx]
        print(f"obs means: {obs_means}")
        print(f"obs stds: {obs_stds}")
        print(f"chebyshev center:\n {self.center}")
        print(f"chebyshev radius: {self.radius}")
        print("-------------------")

    def _normalize_constraints(self):
        """
        根据数据集的 Normalizer 参数，将约束投影到归一化空间，并对 A 的行向量进行归一化。
        
        数学推导原理:
        1. 空间变换:
           原始约束: A_raw * x_raw <= b_raw
           反归一化公式: x_raw = x_norm * scale + offset
           代入得: A_raw * (x_norm * scale + offset) <= b_raw
                 => (A_raw * scale) * x_norm <= b_raw - A_raw * offset
           
           令 A_temp = A_raw * scale, b_temp = b_raw - A_raw * offset
        
        2. 向量归一化 (Row Normalization):
           为了数值稳定性和几何直观性，我们将 A_temp 的每一行除以其 L2 范数。
           令 L_i = ||A_temp[i]||_2
           则 norm_A[i] = A_temp[i] / L_i
              norm_b[i] = b_temp[i] / L_i
        
        return:
            norm_A: (num_cons, len(idx))
            norm_b: (num_cons,)
        """
        # 获取观测的 normalizer (DatasetNormalizer 实例)
        obs_normalizer = self.normalizer.normalizers['observations']
        
        if hasattr(obs_normalizer, "means"):
            # === GaussianNormalizer (均值方差归一化) ===
            obs_means = obs_normalizer.means[self.obs_constrained_idx]
            obs_stds = obs_normalizer.stds[self.obs_constrained_idx]
            
            scale = obs_stds
            offset = obs_means
            
        else:
            # === LimitsNormalizer (归一化到 [-1, 1]) ===
            obs_mins = obs_normalizer.mins[self.obs_constrained_idx]
            obs_maxs = obs_normalizer.maxs[self.obs_constrained_idx]
            
            scale = (obs_maxs - obs_mins) / 2.0
            offset = (obs_maxs + obs_mins) / 2.0

        # 1. 计算投影到归一化空间的约束 (中间结果)
        # norm_A_temp = A_raw * scale
        norm_A_temp = self.raw_A * scale[None, :]
        
        # norm_b_temp = b_raw - A_raw @ offset
        norm_b_temp = self.raw_b - (self.raw_A @ offset)
        
        # 2. 对行向量进行 L2 归一化
        # 计算每一行的 L2 范数: shape (num_cons, 1)
        row_norms = np.linalg.norm(norm_A_temp, ord=2, axis=1, keepdims=True)
        
        # 防止除以 0 (虽然实际约束中法向量不应为 0)
        row_norms[row_norms < 1e-8] = 1.0
        
        # 执行归一化: A / ||A||, b / ||A||
        norm_A = norm_A_temp / row_norms
        norm_b = norm_b_temp / row_norms.flatten() # 展平以匹配 b 的维度

        return norm_A.astype(np.float32), norm_b.astype(np.float32)

    def __getitem__(self, idx):
        """
        返回包含约束的 Batch。
        注意：A 和 b 会被复制扩展到 horizon 长度，以适应序列模型输入。
        """
        # 获取基础的 batch (trajectories, conditions)
        # trajectories shape: [horizon, action_dim + observation_dim]
        base_batch = super().__getitem__(idx)
        
        horizon = base_batch.trajectories.shape[0]
        
        # 扩展 A 和 b 到时间维度: [horizon, num_cons, dim_sub]
        A_seq = np.tile(self.norm_A[None, :, :], (horizon, 1, 1))
        b_seq = np.tile(self.norm_b[None, :], (horizon, 1))
        
        # 转换为 Tensor (如果 dataloader 不自动做的话，这里保持 numpy 也可以，通常 dataloader 会处理)
        # 这里为了和 base_batch 一致，保持 numpy
        return ConstrainedBatch(
            trajectories=base_batch.trajectories,
            conditions=base_batch.conditions,
            A=A_seq,
            b=b_seq
        )

    def generate_prior_data(self, batch_size, device="cuda:0"):
        """
        生成满足归一化约束的可行域初始数据分布。
        生成的形状适配 Flow Matching 的 Flatten 输入。
        
        Returns:
            sample_batch: Tensor (B, horizon * full_x_dim) 
                对于约束受限部分，在对应的chebyshev球中均匀采样；对于无约束部分，在标准正态分布中采样
                由于我们的约束不随序列变化，所以他们的chebyshev球是一样的
            A_batch: Tensor (B, horizon, num_cons, sub_dim)
            b_batch: Tensor (B, horizon, num_cons)
        """
        # 1. 计算 Chebyshev 中心和半径 (基于归一化后的约束)
        # center: (sub_dim,), radius: scalar
        if self.center is None or self.radius is None:
            center, radius = chebyshev_center_lp(self.norm_A, self.norm_b)
        else:
            center = self.center
            radius = self.radius
        center = center.to(device)
        
        # 2. 准备全维度的随机噪声容器 (Standard Normal Distribution)
        # trajectories 结构为: [actions, observations]
        full_dim = self.action_dim + self.observation_dim
        
        # 先生成全维度的标准高斯噪声 (B, T, D)
        # 对于无约束的动作和观测维度，这就是它们的初始分布
        sample_trajs = torch.randn(batch_size, self.horizon, full_dim, device=device)
        
        # 3. 对受限维度进行覆盖采样 (Uniform inside Chebyshev Ball)
        # 计算需要采样的总点数: Batch * Time
        total_points = batch_size * self.horizon
        
        # 在球内均匀采样 (返回 numpy array)
        # constrained_samples shape: (B*T, sub_dim)
        constrained_samples = uniform_sample_in_ball(center, radius * 0.95, num_samples=total_points)
        
        # 转换为 Tensor 并 reshape 为 (B, T, sub_dim)
        constrained_samples = constrained_samples.view(batch_size, self.horizon, -1).to(device)
        
        # 4. 将受限样本填入全维度轨迹中
        # 这里的关键是确定受限维度在 [actions, observations] 中的全局索引
        # self.obs_constrained_idx 是相对于 observations 的索引
        # 因此全局索引需要加上 action_dim
        global_constrained_idxs = self.obs_constrained_idx + self.action_dim
        
        # 执行替换
        sample_trajs[:, :, global_constrained_idxs] = constrained_samples
        
        # 5. 轨迹以适配 Flow Matching 输入 [B, T, D]
        sample_batch = sample_trajs.reshape(batch_size, self.horizon, full_dim)
        
        # 6. 构建 A_batch 和 b_batch
        # 约束在时间维和Batch维是共享/重复的
        
        # A_batch: [B, T, M, sub_dim]
        # self.norm_A shape: (M, sub_dim) -> (1, 1, M, sub_dim) -> repeat
        A_batch = torch.from_numpy(self.norm_A).float().unsqueeze(0).unsqueeze(0)
        A_batch = A_batch.repeat(batch_size, self.horizon, 1, 1).to(device)
        
        # b_batch: [B, T, M]
        # self.norm_b shape: (M,) -> (1, 1, M) -> repeat
        b_batch = torch.from_numpy(self.norm_b).float().unsqueeze(0).unsqueeze(0)
        b_batch = b_batch.repeat(batch_size, self.horizon, 1).to(device)
        
        return sample_batch, A_batch, b_batch
    



class BoxConstrainedMinariDataset(ConstrainedMinariDataset):
    """
    专门处理 Box 形式约束的 Dataset 子类。
    利用 Box 约束的解析性质加速先验数据生成 (无需 Linear Programming)。
    """

    def generate_prior_data(self, batch_size, device="cuda:0"):
        """
        生成满足归一化约束的可行域初始数据分布。
        生成的形状适配 Flow Matching 的 Flatten 输入。
        
        约束形式是box形式，不需要调用LP来计算切比雪夫球，可以直接得到解析解。
        
        假设约束 single_b_raw 的排列格式为: [x0_max, -x0_min, x1_max, -x1_min, ..., xm_max, -xm_min]
        对应的 norm_A 结构为对角块状 (每两行对应一个维度)。

        Returns:
            sample_batch: Tensor (B, horizon * full_x_dim) 
                对于约束受限部分，在对应的chebyshev球中均匀采样；对于无约束部分，在标准正态分布中采样
                由于我们的约束不随序列变化，所以他们的chebyshev球是一样的
            A_batch: Tensor (B, horizon, num_cons, sub_dim)
            b_batch: Tensor (B, horizon, num_cons)
        """
        # 1. 解析归一化后的约束边界 (Analytical Solution)
        # -----------------------------------------------------------
        # norm_A: (num_cons, sub_dim), norm_b: (num_cons,)
        # num_cons 应该是 2 * sub_dim
        # 排列假设: 
        # Row 2k:   a_k * x_k <= b_2k   (Upper bound constraint)
        # Row 2k+1: -a_k * x_k <= b_2k+1 (Lower bound constraint -> x_k >= -b_2k+1 / a_k)
        
        sub_dim = self.norm_A.shape[1]
        assert self.norm_A.shape[0] == 2 * sub_dim, "Box constraints must have 2 * dim rows"

        # 提取系数 (scale)
        # norm_A[2*k, k] 是第 k 个维度的系数
        # diag_indices = (np.arange(sub_dim) * 2, np.arange(sub_dim))
        # scales = self.norm_A[diag_indices] # (sub_dim,)
        
        # 更稳健的方式：提取每一对约束对应的系数绝对值 (避免符号问题)
        # reshape A 到 (sub_dim, 2, sub_dim) -> 取出对应维度的列
        # 这里假设 A 的结构是稀疏对角的，第 k 对约束只作用于第 k 个维度
        scales = np.abs(self.norm_A[np.arange(0, 2*sub_dim, 2), np.arange(sub_dim)])

        # 提取边界值 b
        b_reshaped = self.norm_b.reshape(sub_dim, 2)
        b_upper = b_reshaped[:, 0] # b_2k
        b_lower = b_reshaped[:, 1] # b_2k+1

        # 计算归一化空间下的 Upper (u) and Lower (l) bounds
        # constraint: scale * x <= b_upper  => x <= b_upper / scale
        # constraint: -scale * x <= b_lower => x >= -b_lower / scale
        
        if sub_dim == 1:
            b_upper = b_upper.reshape(sub_dim,)
            b_lower = b_lower.reshape(sub_dim,)
            scales = scales.reshape(sub_dim,)

        u = b_upper / (scales + 1e-8)
        l = -b_lower / (scales + 1e-8)

        # 2. 计算 Chebyshev 球参数
        # -----------------------------------------------------------
        # 中心 c = (u + l) / 2
        center = (u + l) / 2.0
        
        # 半径 r = min( (u - l) / 2 )
        # 每一维的半宽
        half_widths = (u - l) / 2.0
        # Chebyshev 半径是所有维度中最小的半宽 (对应最大内切 L2 球)
        radius = np.min(half_widths)
        
        # 确保半径非负 (处理数值误差)
        radius = max(0.0, radius)

        # 3. 采样 (Sampling)
        # -----------------------------------------------------------
        # 全维度噪声容器 [actions, observations]
        full_dim = self.action_dim + self.observation_dim
        sample_trajs = torch.randn(batch_size, self.horizon, full_dim, device=device) # 标准正态分布作为默认值
        
        # 在受限维度的 L2 球内均匀采样
        total_points = batch_size * self.horizon
        
        # 使用工具函数进行采样 (返回 numpy)
        # constrained_samples: (total_points, sub_dim)
        center = torch.from_numpy(center).to(device)
        constrained_samples = uniform_sample_in_ball(center, radius * 0.95, num_samples=total_points, device=device, dtype=sample_trajs.dtype)
        
        # 转 Tensor 并整形
        constrained_samples = constrained_samples.view(batch_size, self.horizon, -1)
        
        # 4. 填充回全维度轨迹
        # -----------------------------------------------------------
        # 计算全局索引 (trajectory = concat[action, observation])
        global_constrained_idxs = self.obs_constrained_idx + self.action_dim
        
        # 替换受限部分的数据
        sample_trajs[:, :, global_constrained_idxs] = constrained_samples
        
        #  [B, T, D]
        sample_batch = sample_trajs.reshape(batch_size, self.horizon, full_dim)

        # 5. 构建 A_batch, b_batch
        # -----------------------------------------------------------
        # 扩展到 Batch 和 Time 维度
        A_batch = torch.from_numpy(self.norm_A).float().unsqueeze(0).unsqueeze(0)
        A_batch = A_batch.repeat(batch_size, self.horizon, 1, 1).to(device) # (B, T, M, D_sub)
        
        b_batch = torch.from_numpy(self.norm_b).float().unsqueeze(0).unsqueeze(0)
        b_batch = b_batch.repeat(batch_size, self.horizon, 1).to(device)    # (B, T, M)

        return sample_batch, A_batch, b_batch