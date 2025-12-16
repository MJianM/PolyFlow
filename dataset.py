from torch.utils.data import Dataset
import torch
import numpy as np

from chebyshev_center import chebyshev_center_lp, uniform_sample_in_ball



class TrajDataset(Dataset):
    def __init__(self, file_path: str, seq_length=None):
        """
        Docstring for __init__
        
        :param self: Description
        :param file_path: Description
        :type file_path: str
        :param seq_length: 是否对序列进行降采样，seq_length表示降采样后的序列长度
        """
        super().__init__()

        self.file_path = file_path
        data = np.load(file_path)

        self.traj_dataset = data['traj_dataset'] # (batch_size, seq_length, x_dim)
        self.single_A = data['single_A'] # (batch_size, seq_length, num_cons, x_dim)
        self.single_b = data['single_b'] # (batch_size, seq_length, num_cons)

        if seq_length is None:
            self.seq_length = self.traj_dataset.shape[1]
        else:
            length = self.traj_dataset.shape[1]
            indices = np.linspace(0, length - 1, seq_length, dtype=int)
            # 对每个数据集进行降采样
            self.traj_dataset = self.traj_dataset[:, indices, :]  # (batch_size, seq2, x_dim)
            self.single_A = self.single_A[:, indices, :, :]      # (batch_size, seq2, num_cons, x_dim)
            self.single_b = self.single_b[:, indices, :]         # (batch_size, seq2, num_cons)
            self.seq_length = seq_length

        self.num_traj = self.traj_dataset.shape[0]
        self.num_cons = self.single_A.shape[2]
        self.x_dim = self.traj_dataset.shape[-1]

    def __getitem__(self, index):
        return {
            'traj': self.traj_dataset[index].flatten(), # Flow Matching 输入通常展平
            'A': self.single_A[index], # (seq_length, num_cons, x_dim)
            'b': self.single_b[index], # (seq_length, num_cons)
        }
    
    def __len__(self):
        return self.traj_dataset.shape[0]
    
    def generate_prior_data(self, batch_size, A=None, b=None):
        """
        用 Chebyshev 中心生成可行域内的随机初始点
        
        :param self: Description
        :param batch_size: Description
        :param A: tensor (batch_size, seq_length, num_cons, x_dim)
        :param b: tensor (batch_size, seq_length, num_cons)
        """

        if (A is not None) and (b is not None):
            sample_list = []
            for idx in range(batch_size):
                sample_points_list = []
                for i in range(self.seq_length):
                    curA = A[idx][i] # (num_cons, x_dim)
                    curb = b[idx][i] # (num_cons,)
                    center, radius = chebyshev_center_lp(curA, curb)
                    sample_points = uniform_sample_in_ball(center, radius, num_samples=1)
                    sample_points_list.append(sample_points)
                sample = torch.concatenate(sample_points_list, dim=-1) # (1, seq_length*x_dim)
                sample_list.append(sample)  
            sample_batch = torch.concatenate(sample_list, dim=0) # (batch, seq_length*x_dim)

            return sample_batch, A, b
        
        else:
            # 如果没有A, b输入，那么从数据集中随机选取约束
            indices = np.random.choice(self.num_traj, batch_size, replace=True)
            sample_list = []
            for idx in indices:
                sample_points_list = []
                for i in range(self.seq_length):
                    A = self.single_A[idx][i] # (num_cons, x_dim)
                    b = self.single_b[idx][i] # (num_cons,)
                    center, radius = chebyshev_center_lp(A, b)
                    sample_points = uniform_sample_in_ball(center, radius, num_samples=1)
                    sample_points_list.append(sample_points)
                sample = torch.concatenate(sample_points_list, dim=-1) # (1, seq_length*x_dim)
                sample_list.append(sample)
            sample_batch = torch.concatenate(sample_list, dim=0) # (batch, seq_length*x_dim)
            A_batch = torch.from_numpy(self.single_A[indices]) # (batch, seq_length, num_cons, x_dim)
            b_batch = torch.from_numpy(self.single_b[indices]) # (batch, seq_length, num_cons)

            return sample_batch, A_batch, b_batch

    def sample_traj_data(self, n_sample):
        """
        从数据集中采样轨迹
        
        :param self: Description
        :param n_sample: Description
        :return ndarray (n_sample, seq_length*x_dim)
        """
        total_trajectories = self.traj_dataset.shape[0]
        if n_sample > total_trajectories: n_sample = total_trajectories
        random_indices = np.random.choice(total_trajectories, size=n_sample, replace=False)
        return self.traj_dataset[random_indices].reshape(n_sample, -1)


class ConstantConsTrajDataset(TrajDataset):
    """
    假设不同数据之间的约束 A (seq_length, num_cons, x_dim) 是固定的
    也就是说，数据分布在同一个约束表示内
    """
    def __init__(self, file_path, seq_length=None):
        super().__init__(file_path, seq_length)


    def generate_prior_data(self, batch_size, A=None, b=None):
        """
        每个batch的 A b 都相同
        
        :param self: Description
        :param batch_size: Description
        :param A: tensor (batch_size, seq_length, num_cons, x_dim)
        :param b: tensor (batch_size, seq_length, num_cons)
        """

        sample_points_list = []
        for i in range(self.seq_length):
            A = self.single_A[0][i] # (4, 2)
            b = self.single_b[0][i] # (4, )
            center, radius = chebyshev_center_lp(A, b)
            # 缩小一点 radius 保证在内部
            sample_points = uniform_sample_in_ball(center, radius, num_samples=batch_size)
            sample_points_list.append(sample_points)
        sample_batch = torch.concatenate(sample_points_list, dim=-1) # (batch, 2T)

        A_batch = torch.from_numpy(self.single_A[0]).unsqueeze(0).repeat(batch_size, 1, 1, 1) # [B, T, 4, 2]
        b_batch = torch.from_numpy(self.single_b[0]).unsqueeze(0).repeat(batch_size, 1, 1)    # [B, T, 4]

        return sample_batch, A_batch, b_batch 