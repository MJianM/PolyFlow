from collections import namedtuple
import numpy as np
import copy
import torch
import pdb
from qpth.qp import QPFunction, QPSolvers

from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from src.utils.chebyshev_center import chebyshev_center_lp, uniform_sample_in_ball


ConstrainedContactBatch = namedtuple('ConstrainedContactBatch', 'trajectories conditions A b contact')


class LeggedRobotDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, file_path, horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=5000, termination_penalty=0, use_padding=False, seed=None):
        """
        初始化 MinariSequenceDataset。

        参数:
            file_path: 文件位置
            horizon (int): 采样的轨迹片段长度 (T)。
            normalizer (str): 归一化器类型。
            preprocess_fns (list): 预处理函数列表。
            max_path_length (int): 最大路径长度。
            max_n_episodes (int): 加载的最大 episode 总数量 (所有数据集之和)。
            termination_penalty (float): 提前终止的惩罚值。
            use_padding (bool): 是否使用填充。
            seed (int): 随机种子。
        """

        
        self.file_path = file_path
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        # 初始化 ReplayBuffer (只初始化一次，用于存放所有环境的数据)
        # 注意：ReplayBuffer 通常会在第一次 add_path 时确定维度，因此请确保 env_list 中的所有环境维度一致
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        try:
            npz_data = np.load(file_path, allow_pickle=True)
        except:
            print(f"[LeggedRobotDataset] Failed to load file from {file_path}")
            raise ValueError
        

        obs_traj = npz_data['obs_traj'] # (batch, horizon, obs_dim)
        act_traj = npz_data['act_traj'] # (batch, horizon, act_dim)
        u_traj = npz_data['u']   # (batch, horizon,)
        A_traj = npz_data['A']   # (batch, horizon, 4, num_cons, 3)
        b_traj = npz_data['b']   # (batch, horizon, 4, num_cons,)
        contact_mask_traj = npz_data['contact_mask'] # (batch, horizon, 4)
        kp = npz_data['kp']      # (12, )
        kd = npz_data['kd']      # (12, )
        default_q = npz_data['defaut_q'] # (12,)

        batch_size = obs_traj.shape[0]
        horizon_length = obs_traj.shape[1]
        self.kp = kp
        self.kd = kd
        self.default_q = default_q

        """
        act 关节顺序 FL-hip FL-thigh FL-calf, FR, RL, RR,
        act_dim=12
        """

        """
        obs 观测顺序, obs_dim=12+36+1=49
        base_lin_vel (3,)
        base_ang_vel (3,)
        projected_gravity (3,)
        commands (3,)
        dof_pos - default_dof_pos (12, )
        dof_vel (12, )
        actions 表示上一步的policy输出 (12, )
        u (1,)
        """

        """
        trajectory 是将每个时刻的 act + obs 拼接起来 (batch, horizon, act_dim+obs_dim)

        condition 定义为一个字典  {0: (batch, obs_dim)}
            表示的是 horizon=0 时刻的obs

        A: (batch, horizon, 4, num_cons, 3) 
        b: (batch, horizon, 4, num_cons,)
        表示的是不同腿的act受到的约束: A * x \leq b
        腿的顺序与 act 一致，腿内的关节顺序与 act 一致
            
        """

        total_episodes_loaded = 0

        for i in range(batch_size):
            obs = obs_traj[i]
            act = act_traj[i]
            u = u_traj[i]
            A = A_traj[i]
            b = b_traj[i]
            contact = contact_mask_traj[i]

            unreal_mask = np.all(obs == 0, axis=-1) & np.all(act == 0, axis=-1)
            if np.any(unreal_mask):
                unreal_idx = np.argwhere(unreal_mask)[0][0] # 取第一个不成立的索引
                obs = obs[:unreal_idx]
                act = act[:unreal_idx]
                u = u[:unreal_idx]
                A = A[:unreal_idx]
                b = b[:unreal_idx]
                contact = contact[:unreal_idx]
                terminals = np.zeros((unreal_idx,1))
                terminals[unreal_idx-1][0] = 1.0 # 取最后一个状态为terminal
            else:
                terminals = np.zeros((horizon_length, 1))


            path = {
                'observations': obs,
                'actions': act,
                'u': u.reshape(-1, 1),
                'contact': contact,
                'terminals': terminals,
            }

            # 添加到统一的 Buffer 中
            fields.add_path(path)
            
            total_episodes_loaded += 1

        # A, b 独立于 fields 存储
        self.raw_A = A_traj # (batch, horizon, 4, num_cons, 3)
        self.raw_b = b_traj # (batch, horizon, 4, num_cons,)


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
        
        # 对obs和act数据进行归一化
        self.normalize()

        self._init_constraint_normalization_params()

        # 对 A, b 进行归一化
        self.normed_A, self.normed_b = self.normalize_constraints(self.raw_A, self.raw_b)
        print(f"[Dataset] Constraints normalized using vectorized implementation.")
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
        A = self.normed_A[path_ind, start:end]
        b = self.normed_b[path_ind, start:end]
        contact = self.fields.contact[path_ind, start:end]
        u = self.fields.u[path_ind, start:end]

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

        batch = ConstrainedContactBatch(
            trajectories=trajectories,
            conditions=conditions,
            A=A,
            b=b,
            contact=contact
        )
        return batch

    def _init_constraint_normalization_params(self):
        """
        [内部方法] 预计算用于约束归一化的 scale 和 offset 参数。
        将结果存储为形状 (1, 1, 4, 1, 3) 的属性，以便直接广播。
        """
        obs_normalizer = self.normalizer.normalizers['observations']
        act_normalizer = self.normalizer.normalizers['actions']

        # 获取统计量 (逻辑与之前一致)
        if hasattr(obs_normalizer, "means"):
            # === GaussianNormalizer ===
            full_means = np.concatenate([act_normalizer.means, obs_normalizer.means])
            full_stds = np.concatenate([act_normalizer.stds, obs_normalizer.stds])
            stats_mean = full_means[:12]
            stats_std = full_stds[:12]
            
            scale = stats_std
            offset = stats_mean
        else:
            # === LimitsNormalizer ===
            full_mins = np.concatenate([act_normalizer.mins, obs_normalizer.mins])
            full_maxs = np.concatenate([act_normalizer.maxs, obs_normalizer.maxs])
            stats_min = full_mins[:12]
            stats_max = full_maxs[:12]
            
            scale = (stats_max - stats_min) / 2.0
            offset = (stats_max + stats_min) / 2.0

        # 重塑为 (4, 3) -> 4条腿, 3关节
        scale_reshaped = scale.reshape(4, 3)
        offset_reshaped = offset.reshape(4, 3)

        # 扩展维度以支持广播: (Batch, Horizon, Legs, Num_Cons, Joints)
        # Target shape: (1, 1, 4, 1, 3)
        self.cons_scale = scale_reshaped[None, None, :, None, :].astype(np.float32)
        self.cons_offset = offset_reshaped[None, None, :, None, :].astype(np.float32)

    def normalize_constraints(self, A, b):
        """
        向量化版本的约束归一化函数。
        
        参数:
            A: (batch, horizon, 4, num_cons, 3)
            b: (batch, horizon, 4, num_cons)
        """
        # 1. 线性变换 (使用预计算的参数)
        # A_temp = A_raw * scale
        norm_A_temp = A * self.cons_scale

        # b_temp = b_raw - A_raw @ offset
        # sum(axis=-1) 等价于点积
        offset_term = np.sum(A * self.cons_offset, axis=-1)
        norm_b_temp = b - offset_term

        # 2. 行向量 L2 归一化 (Row Normalization)
        # 计算每一行的 L2 范数, axis=-1 为关节维度 (x,y,z)
        row_norms = np.linalg.norm(norm_A_temp, ord=2, axis=-1, keepdims=True)

        # 防止除以 0 (处理 padding 和无效约束)
        row_norms[row_norms < 1e-8] = 1.0

        # 3. 执行除法
        normed_A = norm_A_temp / row_norms
        normed_b = norm_b_temp / row_norms[..., 0] # 移除最后一个维度以匹配 b

        return normed_A.astype(np.float32), normed_b.astype(np.float32)
        
    def unnormalize_constraints(self, normed_A, normed_b):
        """
        将归一化空间中的约束还原回物理空间。
        
        数学推导:
        归一化空间约束: A_norm * x_norm <= b_norm
        代入 x_norm = (x_raw - offset) / scale:
            A_norm * ((x_raw - offset) / scale) <= b_norm
        
        展开:
            (A_norm / scale) * x_raw - (A_norm / scale) * offset <= b_norm
            (A_norm / scale) * x_raw <= b_norm + (A_norm / scale) * offset
            
        因此:
            A_rec = A_norm / scale
            b_rec = b_norm + A_rec @ offset
        
        参数:
            normed_A: (batch, horizon, 4, num_cons, 3)
            normed_b: (batch, horizon, 4, num_cons)
        """
        
        # 1. 恢复 A (A_rec = A_norm / scale)
        # 注意：这里恢复的 A 模长可能与原始 A 不同（因为 normalize 过程中除以了 row_norms），
        # 但它们定义的几何约束边界是完全一致的。
        # scale 形状 (1, 1, 4, 1, 3)，直接广播除法
        
        # 防止 scale 为 0 (理论上 std 或 range 不应为 0，加 eps 增加鲁棒性)
        safe_scale = self.cons_scale + 1e-8
        recovered_A = normed_A / safe_scale

        # 2. 恢复 b (b_rec = b_norm + A_rec @ offset)
        # 计算 A_rec @ offset (点积)
        offset_term = np.sum(recovered_A * self.cons_offset, axis=-1)
        
        recovered_b = normed_b + offset_term

        return recovered_A.astype(np.float32), recovered_b.astype(np.float32)


    def normalize_constraints_tensor(self, A, b):
        """
        [PyTorch 版本] 向量化版本的约束归一化函数。
        适用于在模型 forward/loss 计算中使用，支持自动微分。
        
        参数:
            A: Tensor (batch, horizon, 4, num_cons, 3)
            b: Tensor (batch, horizon, 4, num_cons)
        """
        # 0. 准备参数：将 NumPy 参数转为 Tensor，并移动到正确设备
        # 使用 as_tensor 避免不必要的内存复制（如果源已经是 tensor）
        scale = torch.as_tensor(self.cons_scale, device=A.device, dtype=A.dtype)
        offset = torch.as_tensor(self.cons_offset, device=A.device, dtype=A.dtype)

        # 1. 线性变换
        # A_temp = A_raw * scale
        norm_A_temp = A * scale

        # b_temp = b_raw - A_raw @ offset
        # torch.sum(..., dim=-1) 等价于点积
        offset_term = torch.sum(A * offset, dim=-1)
        norm_b_temp = b - offset_term

        # 2. 行向量 L2 归一化 (Row Normalization)
        # torch.linalg.norm 是新版 PyTorch 推荐的范数计算方式
        # dim=-1 表示沿着关节维度 (x,y,z) 计算
        row_norms = torch.linalg.norm(norm_A_temp, ord=2, dim=-1, keepdim=True)

        # 防止除以 0 (处理 padding 和无效约束)
        # 使用 mask 填充，避免梯度出现 NaN
        # 注意：这里会产生原位修改 (in-place)，如果不希望影响反向传播图的某些分支，可用 torch.where
        row_norms[row_norms < 1e-8] = 1.0

        # 3. 执行除法
        normed_A = norm_A_temp / row_norms
        normed_b = norm_b_temp / row_norms.squeeze(-1) # 移除最后一个维度以匹配 b

        return normed_A, normed_b

    def unnormalize_constraints_tensor(self, normed_A, normed_b):
        """
        [PyTorch 版本] 将归一化空间中的约束还原回物理空间。
        
        参数:
            normed_A: Tensor (batch, horizon, 4, num_cons, 3)
            normed_b: Tensor (batch, horizon, 4, num_cons)
        """
        # 0. 准备参数
        scale = torch.as_tensor(self.cons_scale, device=normed_A.device, dtype=normed_A.dtype)
        offset = torch.as_tensor(self.cons_offset, device=normed_A.device, dtype=normed_A.dtype)

        # 1. 恢复 A (A_rec = A_norm / scale)
        safe_scale = scale + 1e-8
        recovered_A = normed_A / safe_scale

        # 2. 恢复 b (b_rec = b_norm + A_rec @ offset)
        # 计算 A_rec @ offset (点积)
        offset_term = torch.sum(recovered_A * offset, dim=-1)
        
        recovered_b = normed_b + offset_term

        return recovered_A, recovered_b


    # def generate_prior_data(self, batch_size, A_0, b_0, contact_0, device="cuda:0"):
    #     """
    #     生成满足归一化约束的可行域初始数据分布。
    #     A_0: Tensor (batch, 1, 4, num_cons, 3)
    #     b_0: Tensor (batch, 1, 4, num_cons, )
    #     contact_0: (batch, 1, 4)

    #     Returns:
    #         sample_batch: Tensor (B, horizon, full_dim) 
    #             对于约束受限部分，在对应的chebyshev球中均匀采样；对于无约束部分，在标准正态分布中采样
    #             contact_0 = 1 对应的腿部关节受约束，contact_0 = 0 对应的腿部关节不受约束

    #         A_batch: Tensor (B, horizon, 4, num_cons, 3)
    #         b_batch: Tensor (B, horizon, 4, num_cons)
    #         contact_batch: Tensor (B, horizon, 4)
    #     """

    #     sample_batch = torch.randn(batch_size, self.horizon, self.action_dim+self.observation_dim, device=device)
    #     num_cons = A_0.shape[-2]
    #     A_out = torch.zeros(batch_size, self.horizon, 4, num_cons, 3, device=device)
    #     b_out = torch.zeros(batch_size, self.horizon, 4, num_cons, device=device)
    #     contact_out = torch.zeros(batch_size, self.horizon, 4, device=device)

    #     for i in range(batch_size):
    #         A_cur = A_0[i][0] # (4, num_cons, 3)
    #         b_cur = b_0[i][0] # (4, num_cons)
    #         contact_cur = contact_0[i][0] # (4)
    #         for j in range(4):
    #             if contact_cur[j] > 0:
    #                 cen, r = chebyshev_center_lp(A=A_cur[j], b=b_cur[j])
    #                 sample_cur = uniform_sample_in_ball(cen, r, num_samples=1, device=device, dtype=A_0.dtype)
    #                 sample_cur = sample_cur.flatten()
    #                 sample_batch[i, 0, 3*j:3*j+3] = sample_cur

    #     A_out[:, 0:1] = A_0
    #     b_out[:, 0:1] = b_0
    #     contact_out[:, 0:1] = contact_0

    #     return sample_batch, A_out, b_out, contact_out


    def generate_prior_data(self, batch_size, A_0, b_0, contact_0, device="cuda:0"):
        """
        使用 qpth 在 GPU 上批量求解 LP (Chebyshev Center) 并采样。
        """
        # ------------------------------------------------------------------
        # 1. 初始化容器
        # ------------------------------------------------------------------
        sample_batch = torch.randn(batch_size, self.horizon, self.action_dim + self.observation_dim, device=device)
        
        # 填充 metadata
        A_out = torch.zeros(batch_size, self.horizon, 4, A_0.shape[-2], 3, device=device)
        b_out = torch.zeros(batch_size, self.horizon, 4, A_0.shape[-2], device=device)
        contact_out = torch.zeros(batch_size, self.horizon, 4, device=device)
        
        A_out[:, 0:1] = A_0
        b_out[:, 0:1] = b_0
        contact_out[:, 0:1] = contact_0

        # ------------------------------------------------------------------
        # 2. 数据准备：Flatten & Filter
        # ------------------------------------------------------------------
        num_cons = A_0.shape[-2]
        
        # 展平 Batch 和 4条腿 -> (B*4, ...)
        A_flat = A_0.squeeze(1).reshape(-1, num_cons, 3) 
        b_flat = b_0.squeeze(1).reshape(-1, num_cons)    
        contact_flat = contact_0.squeeze(1).reshape(-1)  
        
        # 找出需要求解的索引 (contact > 0)
        active_indices = torch.nonzero(contact_flat > 0, as_tuple=True)[0]
        num_active = len(active_indices)

        if num_active > 0:
            # 提取有效数据
            A_act = A_flat[active_indices] # (K, M, 3)
            b_act = b_flat[active_indices] # (K, M)
            
            # ------------------------------------------------------------------
            # 3. 构建 QP 参数 (关键步骤)
            # ------------------------------------------------------------------
            # 变量 z = [cx, cy, cz, r] (dim=4)
            
            # (1) 构建 G 矩阵: [A, ||A||]
            # 计算每行的模长 ||a_i|| -> (K, M, 1)
            norm_A = torch.norm(A_act, dim=-1, keepdim=True)
            # 拼接 -> (K, M, 4)
            G = torch.cat([A_act, norm_A], dim=-1)
            
            # (2) 构建 h 向量: b
            h = b_act # (K, M)
            
            # (3) 构建 Q 矩阵: epsilon * I
            # qpth 需要 Q 是正定的，这里用 1e-6 的单位阵来近似 LP
            eps = 1e-6
            Q = torch.eye(4, device=device).unsqueeze(0).expand(num_active, 4, 4) * eps
            
            # (4) 构建 p 向量: min (-r) -> p = [0, 0, 0, -1]
            p = torch.zeros(num_active, 4, device=device)
            p[:, 3] = -1.0
            
            # (5) 等式约束 (无)
            e = torch.Tensor().to(device) # 空 tensor
            
            # ------------------------------------------------------------------
            # 4. 调用 qpth 求解
            # ------------------------------------------------------------------
            # z_star shape: (K, 4) -> [cx, cy, cz, r]
            # 注意：如果遇到 singular 错误，可以尝试稍微增大 eps
            try:
                z_star = QPFunction(verbose=False, solver=QPSolvers.PDIPM_BATCHED, eps=1e-4)(Q, p, G, h, e, e)
            except Exception as e:
                # Fallback 或者打印错误 (通常是某些多面体退化导致的)
                print(f"QP Solver failed: {e}. Falling back to random centers (or use previous gradient method).")
                # 为防止 crash，返回全 0
                z_star = torch.zeros(num_active, 4, device=device)
                # 如果多面体异常，那么取无约束输出
                contact_out[:, 0] = 0.0
                return sample_batch, A_out, b_out, contact_out

            centers = z_star[:, :3] # (K, 3)
            radii = z_star[:, 3:4]  # (K, 1)
            
            # 安全检查：防止数值误差导致 r < 0 (虽然理论上不可能，如果多面体存在)
            radii = radii.clamp(min=1e-6)

            # ------------------------------------------------------------------
            # 5. 球内均匀采样 (Vectorized)
            # ------------------------------------------------------------------
            # 随机方向
            direction = torch.randn(num_active, 3, device=device)
            direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
            
            # 随机半径比例 r * u^(1/3)
            u = torch.rand(num_active, 1, device=device)
            scale = u.pow(1.0/3.0)
            
            samples = centers + radii * scale * direction # (K, 3)
            
            # ------------------------------------------------------------------
            # 6. 填回数据
            # ------------------------------------------------------------------
            # 类似于之前的方法，使用 view 进行填值
            # sample_batch 的前12列对应 legs
            target_view = sample_batch[:, 0, :12].reshape(-1, 3) # (B*4, 3)
            
            # 必须使用 clone 或者 detach 确保梯度截断 (生成 prior data 不需要梯度回传)
            target_view[active_indices] = samples.detach()
            
            sample_batch[:, 0, :12] = target_view.reshape(batch_size, 12)

        return sample_batch, A_out, b_out, contact_out







if __name__=='__main__':

    dataset = LeggedRobotDataset(
        file_path='data/bound_go2_traj_data.npz',
        horizon=100,
        normalizer='GaussianNormalizer',
        max_path_length=1000,
        max_n_episodes=5000,
        termination_penalty=0,
        use_padding=False
    )

    constrained_contact_batch = dataset[0]
    trajectory = constrained_contact_batch.trajectories
    condition = constrained_contact_batch.conditions
    contact = constrained_contact_batch.contact
    A = constrained_contact_batch.A
    b = constrained_contact_batch.b
    print("traj:", trajectory.shape)
    print("contact:", contact.shape)
    print("A:", A.shape)
    print("b:", b.shape)

