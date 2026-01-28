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


ConstrainedBatch = namedtuple('ConstrainedBatch', 'trajectories conditions A b')

class ConstrainedMinariDataset(MinariSequenceDataset):
    """
    MinariSequenceDataset 的子类，支持多面体约束 A x <= b。
    """

    def __init__(self, full_constrained_idx, single_A, single_b, *args, **kwargs):
        """
        参数:
            obs_constrained_idx (list/array): 观测向量中参与约束的维度索引 (例如 [0, 1] 表示 x, y)。
            single_A (np.ndarray): 约束矩阵 A，形状 (num_cons, len(obs_constrained_idx))。
            single_b (np.ndarray): 约束边界 b，形状 (num_cons,)。
        """
        super().__init__(*args, **kwargs)

        self.full_constrained_idx = np.array(full_constrained_idx, dtype=int)
        

        self.raw_A = np.array(single_A, dtype=np.float32)
        self.raw_b = np.array(single_b, dtype=np.float32)
        

        assert self.raw_A.shape[1] == len(self.full_constrained_idx), \
            f"A matrix columns ({self.raw_A.shape[1]}) must match constrained dims ({len(self.full_constrained_idx)})"


        print("---------------------")
        self.norm_A, self.norm_b = self._normalize_constraints()

        self.center, self.radius = chebyshev_center_lp(self.norm_A, self.norm_b)

        print(f"norm A:\n {self.norm_A}")
        print(f"norm b:\n {self.norm_b}")
        print(f"chebyshev center:\n {self.center}")
        print(f"chebyshev radius: {self.radius}")
        print("-------------------")

    def _normalize_constraints(self):


        obs_normalizer = self.normalizer.normalizers['observations']
        act_normalizer = self.normalizer.normalizers['actions']
        
        if hasattr(obs_normalizer, "means"):
            # === GaussianNormalizer ===
            full_means = np.concatenate([act_normalizer.means, obs_normalizer.means])
            full_stds = np.concatenate([act_normalizer.stds, obs_normalizer.stds])
            obs_means = full_means[self.full_constrained_idx]
            obs_stds = full_stds[self.full_constrained_idx]
            print(f"means: {obs_means}")
            print(f"stds: {obs_stds}")
            
            scale = obs_stds
            offset = obs_means
            
        else:
            # === LimitsNormalizer ===
            full_mins = np.concatenate([act_normalizer.mins, obs_normalizer.mins])
            full_maxs = np.concatenate([act_normalizer.maxs, obs_normalizer.maxs])
            obs_mins = full_mins[self.full_constrained_idx]
            obs_maxs = full_maxs[self.full_constrained_idx]
            
            scale = (obs_maxs - obs_mins) / 2.0
            offset = (obs_maxs + obs_mins) / 2.0

        # norm_A_temp = A_raw * scale
        norm_A_temp = self.raw_A * scale[None, :]
        
        # norm_b_temp = b_raw - A_raw @ offset
        norm_b_temp = self.raw_b - (self.raw_A @ offset)
        
        row_norms = np.linalg.norm(norm_A_temp, ord=2, axis=1, keepdims=True)
        
        row_norms[row_norms < 1e-8] = 1.0
        
        # A / ||A||, b / ||A||
        norm_A = norm_A_temp / row_norms
        norm_b = norm_b_temp / row_norms.flatten() 

        return norm_A.astype(np.float32), norm_b.astype(np.float32)

    def __getitem__(self, idx):

        base_batch = super().__getitem__(idx)
        horizon = base_batch.trajectories.shape[0]

        if torch.is_tensor(self.norm_A):
            # === GPU / Tensor ===
            A_seq = self.norm_A.unsqueeze(0).repeat(horizon, 1, 1)
            # b_seq: [horizon, num_cons]
            b_seq = self.norm_b.unsqueeze(0).repeat(horizon, 1)
        else:
            # === CPU / Numpy ===
            A_seq = np.tile(self.norm_A[None, :, :], (horizon, 1, 1))
            b_seq = np.tile(self.norm_b[None, :], (horizon, 1))

        return ConstrainedBatch(
            trajectories=base_batch.trajectories,
            conditions=base_batch.conditions,
            A=A_seq,
            b=b_seq
        )

    def generate_prior_data(self, batch_size, device="cuda:0"):

        # center: (sub_dim,), radius: scalar
        if torch.is_tensor(self.center):
            center = self.center.to(device)
        else:
            center = torch.from_numpy(self.center).float().to(device)
            
        radius = self.radius # scalar
        
        # trajectories: [actions, observations]
        full_dim = self.action_dim + self.observation_dim
        

        sample_trajs = torch.randn(batch_size, self.horizon, full_dim, device=device)
        

        total_points = batch_size * self.horizon
        
        # constrained_samples shape: (B*T, sub_dim)
        constrained_samples = uniform_sample_in_ball(center, radius * 0.95, num_samples=total_points)
        
        constrained_samples = constrained_samples.view(batch_size, self.horizon, -1).to(device)
        
        if torch.is_tensor(self.full_constrained_idx):
            idxs = (self.full_constrained_idx).long().to(device)
        else:
             idxs = self.full_constrained_idx 


        sample_trajs[:, :, idxs] = constrained_samples
        
        sample_batch = sample_trajs.reshape(batch_size, self.horizon, full_dim)
        
        if torch.is_tensor(self.norm_A):
            raw_A_tensor = self.norm_A.to(device)
            raw_b_tensor = self.norm_b.to(device)
        else:
            raw_A_tensor = torch.from_numpy(self.norm_A).float().to(device)
            raw_b_tensor = torch.from_numpy(self.norm_b).float().to(device)

        # A_batch: [B, T, M, sub_dim]
        A_batch = raw_A_tensor.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.horizon, 1, 1)
        
        # b_batch: [B, T, M]
        b_batch = raw_b_tensor.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.horizon, 1)
        
        return sample_batch, A_batch, b_batch
    



class BoxConstrainedMinariDataset(ConstrainedMinariDataset):

    def generate_prior_data(self, batch_size, device="cuda:0"):


        sub_dim = self.norm_A.shape[1]
        assert self.norm_A.shape[0] == 2 * sub_dim, "Box constraints must have 2 * dim rows"


        scales = np.abs(self.norm_A[np.arange(0, 2*sub_dim, 2), np.arange(sub_dim)])


        b_reshaped = self.norm_b.reshape(sub_dim, 2)
        b_upper = b_reshaped[:, 0] # b_2k
        b_lower = b_reshaped[:, 1] # b_2k+1

        # constraint: scale * x <= b_upper  => x <= b_upper / scale
        # constraint: -scale * x <= b_lower => x >= -b_lower / scale
        
        if sub_dim == 1:
            b_upper = b_upper.reshape(sub_dim,)
            b_lower = b_lower.reshape(sub_dim,)
            scales = scales.reshape(sub_dim,)

        u = b_upper / (scales + 1e-8)
        l = -b_lower / (scales + 1e-8)


        # c = (u + l) / 2
        center = (u + l) / 2.0
        
        # r = min( (u - l) / 2 )
        half_widths = (u - l) / 2.0

        radius = np.min(half_widths)
        
        radius = max(0.0, radius)

        full_dim = self.action_dim + self.observation_dim
        sample_trajs = torch.randn(batch_size, self.horizon, full_dim, device=device) # 标准正态分布作为默认值
        
        total_points = batch_size * self.horizon
        
        # constrained_samples: (total_points, sub_dim)
        center = torch.from_numpy(center).to(device)
        constrained_samples = uniform_sample_in_ball(center, radius * 0.95, num_samples=total_points, device=device, dtype=sample_trajs.dtype)
        
        constrained_samples = constrained_samples.view(batch_size, self.horizon, -1)
        

        global_constrained_idxs = self.full_constrained_idx 
        

        sample_trajs[:, :, global_constrained_idxs] = constrained_samples
        
        #  [B, T, D]
        sample_batch = sample_trajs.reshape(batch_size, self.horizon, full_dim)


        A_batch = torch.from_numpy(self.norm_A).float().unsqueeze(0).unsqueeze(0)
        A_batch = A_batch.repeat(batch_size, self.horizon, 1, 1).to(device) # (B, T, M, D_sub)
        
        b_batch = torch.from_numpy(self.norm_b).float().unsqueeze(0).unsqueeze(0)
        b_batch = b_batch.repeat(batch_size, self.horizon, 1).to(device)    # (B, T, M)

        return sample_batch, A_batch, b_batch