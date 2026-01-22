from collections import namedtuple
import numpy as np
import copy
import torch
import pdb
from qpth.qp import QPFunction, QPSolvers

from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from src.utils.chebyshev_center import chebyshev_center_lp, uniform_sample_in_ball

def get_approx_center_least_squares(A, b, h, factor=0.5):
    """
    使用 [最小二乘法] 代替 [法向量求和] 来寻找射线方向。
    这能更好地处理扁平或扭曲的锥体，找到更准确的角平分线方向。
    """
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        h = torch.tensor(h, dtype=torch.float32)
    
    device = A.device
    dtype = A.dtype
    
    # --- 1. 预处理 ---
    # 归一化 A，保证数值稳定性 (这一步对最小二乘也很重要)
    row_norms = torch.norm(A, p=2, dim=2, keepdim=True).clamp(min=1e-9)
    A_norm = A / row_norms
    
    # 计算局部 b (用于识别顶点平面)
    val_at_h = torch.einsum('bni, bi -> bn', A, h)
    b_local = torch.relu(b - val_at_h) # ReLU 防止负值干扰
    
    # 识别 Tip Planes (b_local 接近 0 的平面)
    epsilon = 5e-4
    mask_tip = (b_local < epsilon).float().unsqueeze(-1) # (B, N, 1)
    
    # --- 2. 核心改进: 最小二乘方向估计 ---
    # 我们只关心 Tip Planes，其他平面的行我们要屏蔽掉 (乘0)
    A_tip = A_norm * mask_tip # (B, N, 3)
    
    # 目标: 求解 A_tip * d = -1
    # 也就是: A_tip^T * A_tip * d = - A_tip^T * 1
    
    # 构建 Hessian 矩阵 H = A^T A (3x3)
    # (B, 3, N) @ (B, N, 3) -> (B, 3, 3)
    H = torch.bmm(A_tip.transpose(1, 2), A_tip)
    
    # 添加阻尼项 (Ridge Regularization) 防止矩阵奇异
    # 特别是当平面数量 < 3 时，或者平面共线时，H 不可逆
    lambda_I = 1e-3 * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    H_reg = H + lambda_I
    
    # 构建右侧向量 g = - A^T * 1
    # sum(A_tip, dim=1) 等价于 A^T * 1
    g = -torch.sum(A_tip, dim=1).unsqueeze(-1) # (B, 3, 1)
    
    # 求解线性方程 H_reg * d = g
    # torch.linalg.solve 比手动求逆更稳
    direction = torch.linalg.solve(H_reg, g).squeeze(-1) # (B, 3)
    
    # 归一化方向
    direction = direction / torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-9)

    # --- 3. 射线投射 (Ray Casting) ---
    # (这部分逻辑保持不变，依然是撞击检测)
    
    projections = torch.einsum('bni, bi -> bn', A, direction)
    valid_proj_mask = (projections > 1e-9) # 检查所有前方平面
    
    safe_projections = torch.where(valid_proj_mask, projections, torch.tensor(1.0, device=device, dtype=dtype))
    calculated_t = b_local / safe_projections
    
    inf_tensor = torch.full_like(b, float('inf'))
    t_values = torch.where(valid_proj_mask, calculated_t, inf_tensor)
    
    t_hit = torch.min(t_values, dim=1)[0]
    t_hit = torch.where(torch.isinf(t_hit), torch.tensor(1.0, device=device, dtype=dtype), t_hit)
    
    centers = h + direction * t_hit.unsqueeze(1) * factor
    
    return centers

def get_approx_center_ray_torch(A, b, h, factor=0.4):
    """
    Batch 版本的近似切比雪夫中心求解 (PyTorch版)。
    
    参数:
        A: (batch, num_cons, 3) FloatTensor, 约束矩阵
        b: (batch, num_cons) FloatTensor, 约束向量
        h: (batch, 3) FloatTensor, 已知的多面体顶点坐标
        factor: float, 射线撞击距离的缩放因子 (0~1)
        
    返回:
        centers: (batch, 3) FloatTensor
    """
    # 确保输入是 tensor 且在同一设备
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        h = torch.tensor(h, dtype=torch.float32)
        
    device = A.device
    dtype = A.dtype
    epsilon = 1e-3
    
    # --- 步骤 1: 坐标系平移 (Shift to Local Frame) ---
    # 计算 h 在每个平面上的投影值: (batch, num_cons)
    # val_at_h = sum(A * h, axis=-1)
    val_at_h = torch.einsum('bni, bi -> bn', A, h)
    
    # 计算相对于 h 的新约束向量 b'
    b_local = b - val_at_h
    
    # --- 步骤 2: 识别构成顶点的平面 (Identify Tip Planes) ---
    # 检查 b_local 是否接近 0
    mask_tip = torch.abs(b_local) < epsilon
    
    flag = torch.sum(mask_tip, dim=-1) >= 3
    if not torch.all(flag):
        invalid_indices = torch.nonzero(~flag, as_tuple=False).squeeze(-1)
        raise ValueError(f"[get_approx_center_ray_torch] Invalid polytope at batch indices: {invalid_indices.tolist()}. Less than 3 tip planes found.")

    mask_base = ~mask_tip
    
    # --- 步骤 3: 计算射击方向 (Direction Calculation) ---
    
    # 计算每一行的 L2 范数: (batch, num_cons, 1)
    # torch.norm 在 dim=2 上计算，保持维度
    row_norms = torch.norm(A, p=2, dim=2, keepdim=True)
    row_norms = torch.clamp(row_norms, min=1e-9) # 防止除零
    
    # 归一化 A
    A_normalized = A / row_norms
    
    # 提取构成顶点的面的法向量
    # 注意：需将 mask_tip (bool) 转换为 float 才能相乘
    mask_tip_float = mask_tip.unsqueeze(-1).to(dtype) # (batch, num_cons, 1)
    relevant_normals = A_normalized * mask_tip_float
    
    # 求和并取反 (指向内部): (batch, 3)
    raw_direction = -torch.sum(relevant_normals, dim=1)
    
    # 归一化方向向量
    dir_norm = torch.norm(raw_direction, p=2, dim=1, keepdim=True)
    dir_norm = torch.clamp(dir_norm, min=1e-9)
    direction = raw_direction / dir_norm  # (batch, 3)
    
    # --- 步骤 4: 计算射线撞击距离 (Ray Casting) ---
    
    # 计算方向在各平面法向量上的投影: (batch, num_cons)
    projections = torch.einsum('bni, bi -> bn', A, direction)
    
    # 筛选有效的阻挡平面 (必须是 Base Plane 且 朝向射线)
    valid_proj_mask = mask_base & (projections > epsilon)
    
    # 避免除零处理
    # torch.where(condition, x, y)
    safe_projections = torch.where(valid_proj_mask, projections, torch.tensor(1.0, device=device, dtype=dtype))
    
    # 计算 t (使用 b_local)
    calculated_t = b_local / safe_projections
    
    # 初始化 t_values 为无穷大
    inf_tensor = torch.full_like(b, float('inf'))
    
    # 应用 Mask: 有效位置填计算值，无效位置保持 inf
    t_values = torch.where(valid_proj_mask, calculated_t, inf_tensor)
    
    # 寻找最近的撞击点
    # 注意: torch.min(dim=1) 返回 (values, indices) named tuple
    t_hit = torch.min(t_values, dim=1)[0] # 取 values
    
    # 处理无界情况 (inf -> default value 1.0)
    t_hit = torch.where(torch.isinf(t_hit), torch.tensor(1.0, device=device, dtype=dtype), t_hit)
    
    # --- 步骤 5: 还原坐标 ---
    # centers = h + direction * distance * factor
    centers = h + direction * t_hit.unsqueeze(1) * factor
    
    return centers

def compute_chebyshev_radius_batch(A, b, center):
    """
    计算给定中心点在多面体 A*x <= b 内的内切球半径 (Batch版)。
    
    参数:
        A: (batch, num_cons, 3) 约束矩阵
        b: (batch, num_cons) 约束向量
        center: (batch, 3) 待检测的中心点
        
    返回:
        radius: (batch,) 该点的内切球半径。
                如果 radius < 0，表示该 center 在多面体外部。
    """
    # 确保输入是 tensor
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        center = torch.tensor(center, dtype=torch.float32)
        
    # 1. 计算 A 中每一行的 L2 范数 (用于归一化距离)
    # shape: (batch, num_cons)
    row_norms = torch.norm(A, p=2, dim=2)
    
    # 防止除零 (虽然几何约束中 a_i 通常不为0)
    row_norms = torch.clamp(row_norms, min=1e-9)
    
    # 2. 计算投影 A * center
    # input: (batch, num_cons, 3) * (batch, 3) -> (batch, num_cons)
    projections = torch.einsum('bni, bi -> bn', A, center)
    
    # 3. 计算松弛变量 (Slack) / 原始距离
    # slacks = b - A*x
    slacks = b - projections
    
    # 4. 归一化距离 (真正的几何距离)
    # dist = (b - A*x) / ||A||
    dists = slacks / row_norms
    
    # 5. 取最小值作为半径
    # 如果所有 dists >= 0，min(dists) 就是离最近墙壁的距离
    # 如果有任意 dist < 0，min(dists) 表示违反最严重的约束的距离 (负值)
    radius, _ = torch.min(dists, dim=1)

    if not torch.all(radius >= -1e-6):
        # 可选：检查 radius 是否合理
        invalid_indices = torch.nonzero(radius < -1e-6, as_tuple=False).squeeze(-1)
        raise ValueError(f"[compute_chebyshev_radius_batch] Some centers are outside the polytope at batch indices: {invalid_indices.tolist()}.")
    
    return radius

def uniform_sample_in_ball_torch(center, radius):
    """
    在给定的球体中均匀采样点 (Batch版)。
    
    算法原理:
    1. 方向采样: 使用高斯分布采样得到均匀分布在单位球面上的方向向量 (Muller's Method)。
    2. 半径采样: 采样 u ~ Uniform(0, 1)，为了消除中心聚集效应，使得点在体积上均匀分布，
       实际缩放系数应为 u^(1/3) (三维体积与半径的三次方成正比)。
    
    :param center: (batch, 3) Tensor, 球心坐标
    :param radius: (batch, ) Tensor, 球体半径
    :return: (batch, 3) Tensor, 采样得到的点坐标
    """
    # 1. 确保数据类型和设备一致
    device = center.device
    dtype = center.dtype
    batch_size = center.shape[0]
    
    # 2. 生成随机方向 (Random Direction)
    # 使用标准正态分布生成的向量，归一化后是均匀分布在单位球面上的
    # shape: (batch, 3)
    random_dirs = torch.randn_like(center)
    
    # 归一化得到单位向量
    # dim=1 表示在空间维度 (x,y,z) 上求范数
    # keepdim=True 保持形状为 (batch, 1) 以便广播
    norms = torch.norm(random_dirs, p=2, dim=1, keepdim=True)
    
    # 加上极小值防止除以0 (虽然在高斯分布中概率极低)
    unit_dirs = random_dirs / torch.clamp(norms, min=1e-9)
    
    # 3. 生成随机半径比例 (Random Radius Scale)
    # 采样 u ~ Uniform(0, 1)
    # shape: (batch, 1)
    u = torch.rand((batch_size, 1), device=device, dtype=dtype)
    
    # 关键步骤：体积修正
    # 为了保证在球体内均匀分布，半径 r 的概率密度函数应与 r^2 成正比
    # 通过累积分布函数逆变换法 (Inverse CDF)，缩放系数应为 u^(1/3)
    scale_factors = torch.pow(u, 1.0/3.0)
    
    # 4. 组合计算最终坐标
    # result = center + radius * scale_factor * unit_direction
    # 注意维度广播: 
    # center: (B, 3)
    # radius: (B,) -> unsqueeze -> (B, 1)
    # scale_factors: (B, 1)
    # unit_dirs: (B, 3)
    
    sampled_points = center + radius.unsqueeze(1) * scale_factors * unit_dirs
    
    return sampled_points


ConstrainedContactBatch = namedtuple('ConstrainedContactBatch', 'trajectories conditions A b contact vertex')


class LeggedRobotDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, file_path, horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=5000, termination_penalty=0, use_padding=False, seed=None, initial_sample_mode='qp', action_scale=0.25):
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
        self.initial_sample_mode = initial_sample_mode
        assert initial_sample_mode in ['qp', 'lp', 'approx'], "initial_sample_mode must be one of ['qp', 'lp', 'approx']"
        self.action_scale = action_scale

        # 初始化 ReplayBuffer (只初始化一次，用于存放所有环境的数据)
        # 注意：ReplayBuffer 通常会在第一次 add_path 时确定维度，因此请确保 env_list 中的所有环境维度一致
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        try:
            npz_data = np.load(file_path, allow_pickle=True)
        except:
            print(f"[LeggedRobotDataset] Failed to load file from {file_path}")
            raise ValueError
        

        obs_traj = npz_data['obs_traj'] # (batch, horizon, obs_dim)
        act_traj = npz_data['act_traj'] * self.action_scale # (batch, horizon, act_dim)
        u_traj = npz_data['u']   # (batch, horizon,)
        A_traj = npz_data['A']   # (batch, horizon, 4, num_cons, 3)
        b_traj = npz_data['b']   # (batch, horizon, 4, num_cons,)
        contact_mask_traj = npz_data['contact_mask'] # (batch, horizon, 4)
        h_traj = npz_data['h'] # (batch, horizon, 4, 3) 非线性项
        q_all_traj = npz_data['q'] # (batch, horizon, 19) 4+3+12 世界坐标系
        v_all_traj = npz_data['v'] # (batch, horizon, 18) 3+3+12 世界坐标系
        q_traj = q_all_traj[:, :, 7:] # (batch, horizon, 12) 仅关节角度
        v_traj = v_all_traj[:, :, 6:] # (batch, horizon, 12) 仅关节角速度
        kp = npz_data['kp']      # (12, )
        kd = npz_data['kd']      # (12, )
        default_q = npz_data['defaut_q'] # (12,)

        batch_size = obs_traj.shape[0]
        horizon_length = obs_traj.shape[1]

        # 计算约束凸多面体顶点 (batch, horizon, 4, 3)
        kp_mat = kp.reshape(1, 1, 4, 3)
        kd_mat = kd.reshape(1, 1, 4, 3)
        default_q_mat = default_q.reshape(1, 1, 4, 3)
        tau_bias = kp_mat * (default_q_mat - q_traj.reshape(batch_size, horizon_length, 4, 3)) - kd_mat * v_traj.reshape(batch_size, horizon_length, 4, 3)
        vertex = (h_traj - tau_bias) / kp_mat

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
        self.raw_vertex = vertex # (batch, horizon, 4, 3)


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

        # 对 vertex 进行归一化
        self.normed_vertex = self.normalizer(self.raw_vertex.reshape(self.n_episodes*self.max_path_length, -1), 'actions')
        self.normed_vertex = self.normed_vertex.reshape(self.n_episodes, self.max_path_length, 4, 3)

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
        vertex = self.normed_vertex[path_ind, start:end]
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
            contact=contact,
            vertex=vertex,
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


    def generate_prior_data(self, batch_size, A_0, b_0, contact_0, device, h_0=None):
        """
        生成满足归一化约束的可行域初始数据分布。
         A_0: Tensor (batch, 1, 4, num_cons, 3)
         b_0: Tensor (batch, 1, 4, num_cons, )
         contact_0: (batch, 1, 4)
         h_0: (batch, 1, 4, 3)
        Returns:
            sample_batch: Tensor (B, horizon, full_dim) 
                对于约束受限部分，在对应的chebyshev球中均匀采样；对于无约束部分，在标准正态分布中采样
                contact_0 = 1 对应的腿部关节受约束，contact_0 = 0 对应的腿部关节不受约束

            A_batch: Tensor (B, horizon, 4, num_cons, 3)
            b_batch: Tensor (B, horizon, 4, num_cons)
            contact_batch: Tensor (B, horizon, 4)
        """
        if self.initial_sample_mode == 'qp':
            return self.generate_prior_data_qp(batch_size, A_0, b_0, contact_0, device)
        elif self.initial_sample_mode == 'approx':
            return self.generate_prior_data_approx(batch_size, A_0, b_0, contact_0, device, h_0)
        else:
            raise ValueError(f"Unknown initial_sample_mode: {self.initial_sample_mode}")

    def generate_prior_data_approx(self, batch_size, A_0, b_0, contact_0, device, h_0=None):
        """
        使用近似方法在 GPU 上批量求解 Chebyshev Center 并采样。
         A_0: Tensor (batch, 1, 4, num_cons, 3)
         b_0: Tensor (batch, 1, 4, num_cons, )
         contact_0: (batch, 1, 4)
         h_0: (batch, 1, 4, 3)
        """
        A_0 = A_0.to(device)
        b_0 = b_0.to(device)
        contact_0 = contact_0.to(device)
        if h_0 is not None:
            h_0 = h_0.to(device)

        # ------------------------------------------------------------------
        # 1. 初始化容器
        # ------------------------------------------------------------------
        sample_batch = torch.randn(batch_size, self.horizon, self.action_dim + self.observation_dim, device=device)
        
        # 填充 metadata
        A_out = torch.zeros(batch_size, self.horizon, 4, A_0.shape[-2], 3, device=device)
        b_out = torch.zeros(batch_size, self.horizon, 4, b_0.shape[-1], device=device)
        contact_out = torch.zeros(batch_size, self.horizon, 4, device=device)
        
        A_out[:, 0:1] = A_0
        b_out[:, 0:1] = b_0
        contact_out[:, 0:1] = contact_0

        # ------------------------------------------------------------------
        # 2. 数据准备：Flatten & Filter
        # ------------------------------------------------------------------
        num_cons = A_0.shape[-2]
        
        # 展平 Batch 和 4条腿 -> (B*4, ...)
        contain = sample_batch[:, 0, 0:12].reshape(batch_size, 4, 3).reshape(batch_size*4, 3)
        A_flat = A_0.squeeze(1).reshape(-1, num_cons, 3) 
        b_flat = b_0.squeeze(1).reshape(-1, num_cons)    
        contact_flat = contact_0.squeeze(1).reshape(-1)  
        h_flat = h_0.squeeze(1).reshape(-1, 3)
        
        # 找出需要求解的索引 (contact > 0)
        active_indices = torch.nonzero(contact_flat > 0, as_tuple=True)[0]
        num_active = len(active_indices)

        if num_active > 0:
            # 提取有效数据
            A_act = A_flat[active_indices] # (K, M, 3)
            b_act = b_flat[active_indices] # (K, M)
            h_act = h_flat[active_indices] # (K, 3)
            
            # ------------------------------------------------------------------
            # 3. 调用近似方法求解 Chebyshev Center
            # ------------------------------------------------------------------
            center_act = get_approx_center_least_squares(A_act, b_act, h_act, factor=0.5)
            radius_act = compute_chebyshev_radius_batch(A_act, b_act, center_act)

            # 从球中均匀采样
            samples_act = uniform_sample_in_ball_torch(center=center_act, radius=radius_act)

            
            contain[active_indices, 0:3] = samples_act
            sample_batch[:, 0, 0:12] = contain.reshape(batch_size, 12)

        return sample_batch, A_out, b_out, contact_out


    def generate_prior_data_qp(self, batch_size, A_0, b_0, contact_0, device="cuda:0"):
        """
        使用 qpth 在 GPU 上批量求解 LP (Chebyshev Center) 并采样。
         A_0: Tensor (batch, 1, 4, num_cons, 3)
         b_0: Tensor (batch, 1, 4, num_cons, )
         contact_0: (batch, 1, 4)
         h_0: (batch, 1, 4, 3)
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

    # dataset = LeggedRobotDataset(
    #     file_path='data/bound_go2_traj_data.npz',
    #     horizon=100,
    #     normalizer='GaussianNormalizer',
    #     max_path_length=1000,
    #     max_n_episodes=5000,
    #     termination_penalty=0,
    #     use_padding=False
    # )

    # constrained_contact_batch = dataset[0]
    # trajectory = constrained_contact_batch.trajectories
    # condition = constrained_contact_batch.conditions
    # contact = constrained_contact_batch.contact
    # A = constrained_contact_batch.A
    # b = constrained_contact_batch.b
    # print("traj:", trajectory.shape)
    # print("contact:", contact.shape)
    # print("A:", A.shape)
    # print("b:", b.shape)

    A=np.array([[-0.0666,  0.8082,  0.5851],
            [-0.0111,  0.1665, -0.9860],
            [-0.1783,  0.3013, -0.9367],
            [ 0.2009,  0.3006, -0.9323],
            [ 0.0225, -0.3064,  0.9516]])

    b=  np.array([ 2.0999, -0.5581, -0.2822, -0.2042,  9.6819])

    A=torch.from_numpy(A).unsqueeze(0).unsqueeze(0)  # (1, 1, num_cons, 3)
    b=torch.from_numpy(b).unsqueeze(0).unsqueeze(0)  # (1, 1, num_cons)

    vertex = [0.19915308, 1.96614096, 0.89579297]
    vertex = torch.from_numpy(vertex).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)

    center = get_approx_center_ray_torch(A, b, vertex, factor=0.4)

    print("center:", center)

