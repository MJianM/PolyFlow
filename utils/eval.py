import numpy as np
import torch
import ot  # POT 
from scipy.stats import gaussian_kde



# ==========================================
# 辅助函数
# ==========================================

def _to_numpy(data):
    """内部工具：将 tensor 转为 numpy，并保持 (N, T, D) 格式"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def _compute_mmd_kernel(x, y):
    """
    计算 MMD (最大均值差异) - RBF Kernel
    """
    x_kernel = x @ x.T
    y_kernel = y @ y.T
    xy_kernel = x @ y.T
    
    x_sqnorms = np.diag(x_kernel)
    y_sqnorms = np.diag(y_kernel)
    
    # 中位数启发式带宽
    combined_dist = np.concatenate([x_sqnorms, y_sqnorms])
    sigma = np.median(combined_dist) if len(combined_dist) > 0 else 1.0
    gamma = 1.0 / (2 * sigma + 1e-8)
    
    k_xx = np.exp(-gamma * (x_sqnorms[:, None] + x_sqnorms[None, :] - 2 * x_kernel))
    k_yy = np.exp(-gamma * (y_sqnorms[:, None] + y_sqnorms[None, :] - 2 * y_kernel))
    k_xy = np.exp(-gamma * (x_sqnorms[:, None] + y_sqnorms[None, :] - 2 * xy_kernel))
    
    return max(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean(), 0.0)

def _compute_wasserstein_pot(x, y):
    """
    使用 POT 库计算 Wasserstein-2 距离 (精确解)
    """
    # 构造均匀分布权重
    n, m = x.shape[0], y.shape[0]
    a, b = np.ones((n,)) / n, np.ones((m,)) / m
    
    # 计算代价矩阵 (平方欧氏距离)
    M = ot.dist(x, y, metric='sqeuclidean')
    
    # 计算 EMD (W2^2)
    w2_sq = ot.emd2(a, b, M)
    
    # 开根号返回 W2
    return np.sqrt(w2_sq)

def _compute_kl_with_kde(x_true, x_sampled, jitter=1e-5):
    """
    使用 KDE 计算 KL 散度: KL(P_true || Q_sampled)
    注意：仅适用于低维数据 (D < 10)
    """
    # 转置为 (D, N) 以适配 scipy
    data_true = x_true.T
    data_sampled = x_sampled.T
    
    # 添加 Jitter 防止奇异矩阵
    data_true += np.random.normal(0, jitter, data_true.shape)
    data_sampled += np.random.normal(0, jitter, data_sampled.shape)

    try:
        kde_p = gaussian_kde(data_true)
        kde_q = gaussian_kde(data_sampled)
        
        # 在真实数据点上评估密度
        log_p = kde_p.logpdf(data_true)
        log_q = kde_q.logpdf(data_true)
        
        # 截断防止 log(0)
        log_q = np.clip(log_q, a_min=-100.0, a_max=None)
        
        kl_div = np.mean(log_p - log_q)
        return max(kl_div, 0.0)
    except Exception as e:
        # 捕捉奇异矩阵或其他数值错误
        return float('nan')


# ==========================================
# 主函数 1: 分布差异评估
# ==========================================
def evaluate_dismatch_metrics(sampled_traj, true_traj, check_horizon_list, max_samples=1000):
    """
    在指定时间步计算三种分布差异指标：MMD, Wasserstein2, KL(KDE)。
    
    参数:
    - sampled_traj: (N, T, D) 生成数据
    - true_traj: (N, T, D) 真实数据
    - check_horizon_list: list[int], 要检查的时间步索引
    - max_samples: int, 下采样上限 (建议 <= 2000 以避免 POT 计算过慢)
    
    返回:
    - dict: {"mmd": [], "wasserstein": [], "kl": []}
    """
    s_np = _to_numpy(sampled_traj)
    t_np = _to_numpy(true_traj)
    
    num_steps = s_np.shape[1]
    
    results = {
        "mmd": [],
        "wasserstein": [],
        "kl": []
    }
    
    for t_idx in check_horizon_list:
        if t_idx < 0 or t_idx >= num_steps:
            # 越界处理：填入 NaN
            results["mmd"].append(float('nan'))
            results["wasserstein"].append(float('nan'))
            results["kl"].append(float('nan'))
            continue
            
        # 提取切片
        s_step = s_np[:, t_idx, :]
        t_step = t_np[:, t_idx, :]
        
        # 下采样 (Sampling)
        curr_N = s_step.shape[0]
        if curr_N > max_samples:
            idx_s = np.random.choice(curr_N, max_samples, replace=False)
            idx_t = np.random.choice(t_step.shape[0], max_samples, replace=False)
            s_eval = s_step[idx_s]
            t_eval = t_step[idx_t]
        else:
            s_eval = s_step
            t_eval = t_step
            
        # 1. 计算 MMD
        results["mmd"].append(_compute_mmd_kernel(s_eval, t_eval))
        
        # 2. 计算 Wasserstein (POT)
        results["wasserstein"].append(_compute_wasserstein_pot(s_eval, t_eval))
        
        # 3. 计算 KL Divergence (KDE)
        results["kl"].append(_compute_kl_with_kde(t_eval, s_eval))
        
    return results


def evaluate_distribution_gap(sampled_traj, true_traj, max_samples=2000):
    """
    比较生成轨迹和真实轨迹的数据分布差异，不考虑时序相关性
    
    参数:
    - sampled_traj: (num_traj, seq_length, x_dim) 生成数据
    - true_traj: (num_traj, seq_length, x_dim) 真实数据
    - max_samples: int, 计算 MMD 时采样的最大点数，防止 O(N^2) 计算量过大
    
    返回:
    - dict: {"distribution_mmd": float}
    """
    s_np = _to_numpy(sampled_traj)
    t_np = _to_numpy(true_traj)
    
    # 策略：将 (N, T, D) 展平为 (N*T, D) 
    # 这样比较的是状态空间(State Space)的覆盖程度，而非整条轨迹的时序相关性
    s_flat = s_np.reshape(-1, s_np.shape[-1])
    t_flat = t_np.reshape(-1, t_np.shape[-1])
    
    # 下采样以提高计算速度
    if s_flat.shape[0] > max_samples:
        idx_s = np.random.choice(s_flat.shape[0], max_samples, replace=False)
        idx_t = np.random.choice(t_flat.shape[0], max_samples, replace=False)
        s_eval = s_flat[idx_s]
        t_eval = t_flat[idx_t]
    else:
        s_eval = s_flat
        t_eval = t_flat
        
    mmd_score = _compute_mmd_kernel(s_eval, t_eval)
    
    return {
        "distribution_mmd": mmd_score
    }

# ==========================================
# 主函数 2: 单组轨迹质量评估 (安全与平滑)
# ==========================================

def evaluate_trajectory_quality(trajectories, safety_check_fn):
    """
    评估单组轨迹的安全性、物理平滑性指标。
    
    参数:
    - trajectories: (num_traj, seq_length, x_dim)
    - safety_check_fn: function, 输入轨迹，返回 bool
    
    返回:
    - dict: {
        "safety_ratio": float,
        "curvature_smoothness": float, (平均转角，越小越好)
        "acc_smoothness": float        (平均加速度模长，越小越好)
      }
    """
    traj_np = _to_numpy(trajectories)
    
    # 1. 安全比率计算
    total = len(traj_np)
    safe_flag_list = safety_check_fn(traj_np)  # 确保函数可调用
    safe_count = np.sum(safe_flag_list)
    safety_ratio = safe_count / total if total > 0 else 0.0
    
    # 2. 物理平滑性计算 (需要 numpy)
    # 如果序列长度小于 3，无法计算加速度和曲率，直接返回 NaN
    if traj_np.shape[1] < 3:
        return {
            "safety_ratio": safety_ratio,
            "curvature_smoothness": float('nan'),
            "acc_smoothness": float('nan')
        }

    # 计算速度: (N, T-1, D)
    velocity = traj_np[:, 1:, :] - traj_np[:, :-1, :]
    
    # 计算加速度: (N, T-2, D)
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
    
    # 指标 A: 加速度平滑性 (平均加速度模长)
    acc_magnitude = np.linalg.norm(acceleration, axis=-1)
    acc_smoothness = np.mean(acc_magnitude)
    
    # 指标 B: 轨迹曲率平滑性 (速度向量夹角)
    v_t = velocity[:, :-1, :]   # t 时刻
    v_t1 = velocity[:, 1:, :]   # t+1 时刻
    
    norm_v_t = np.linalg.norm(v_t, axis=-1)
    norm_v_t1 = np.linalg.norm(v_t1, axis=-1)
    
    eps = 1e-8
    dot_product = np.sum(v_t * v_t1, axis=-1)
    # 计算 Cosine 相似度
    cosine_sim = dot_product / (norm_v_t * norm_v_t1 + eps)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    # 转为弧度角
    angles = np.arccos(cosine_sim)
    curvature_smoothness = np.mean(angles)
    
    return {
        "safety_ratio": safety_ratio,
        "curvature_smoothness": curvature_smoothness,
        "acc_smoothness": acc_smoothness
    }