import numpy as np
import torch
import ot  # POT 
from scipy.stats import gaussian_kde




def _to_numpy(data):

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def _compute_mmd_kernel(x, y):
    """
    MMD 
    """
    x_kernel = x @ x.T
    y_kernel = y @ y.T
    xy_kernel = x @ y.T
    
    x_sqnorms = np.diag(x_kernel)
    y_sqnorms = np.diag(y_kernel)
    
    combined_dist = np.concatenate([x_sqnorms, y_sqnorms])
    sigma = np.median(combined_dist) if len(combined_dist) > 0 else 1.0
    gamma = 1.0 / (2 * sigma + 1e-8)
    
    k_xx = np.exp(-gamma * (x_sqnorms[:, None] + x_sqnorms[None, :] - 2 * x_kernel))
    k_yy = np.exp(-gamma * (y_sqnorms[:, None] + y_sqnorms[None, :] - 2 * y_kernel))
    k_xy = np.exp(-gamma * (x_sqnorms[:, None] + y_sqnorms[None, :] - 2 * xy_kernel))
    
    return max(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean(), 0.0)

def _compute_wasserstein_pot(x, y):
    """
    Wasserstein-2
    """

    n, m = x.shape[0], y.shape[0]
    a, b = np.ones((n,)) / n, np.ones((m,)) / m
    
    M = ot.dist(x, y, metric='sqeuclidean')
    
    w2_sq = ot.emd2(a, b, M)
    
    return np.sqrt(w2_sq)

def _compute_kl_with_kde(x_true, x_sampled, jitter=1e-5):
    """
    KL: KL(P_true || Q_sampled)
    """

    data_true = x_true.T
    data_sampled = x_sampled.T
    
    data_true += np.random.normal(0, jitter, data_true.shape)
    data_sampled += np.random.normal(0, jitter, data_sampled.shape)

    try:
        kde_p = gaussian_kde(data_true)
        kde_q = gaussian_kde(data_sampled)
        
        log_p = kde_p.logpdf(data_true)
        log_q = kde_q.logpdf(data_true)
        
        log_q = np.clip(log_q, a_min=-100.0, a_max=None)
        
        kl_div = np.mean(log_p - log_q)
        return max(kl_div, 0.0)
    except Exception as e:
        return float('nan')


def evaluate_dismatch_metrics(sampled_traj, true_traj, check_horizon_list, max_samples=1000):
    """
    MMD, Wasserstein2, KL(KDE)。
    
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
            results["mmd"].append(float('nan'))
            results["wasserstein"].append(float('nan'))
            results["kl"].append(float('nan'))
            continue
            
        s_step = s_np[:, t_idx, :]
        t_step = t_np[:, t_idx, :]
        
        curr_N = s_step.shape[0]
        if curr_N > max_samples:
            idx_s = np.random.choice(curr_N, max_samples, replace=False)
            idx_t = np.random.choice(t_step.shape[0], max_samples, replace=False)
            s_eval = s_step[idx_s]
            t_eval = t_step[idx_t]
        else:
            s_eval = s_step
            t_eval = t_step
            
        # MMD
        results["mmd"].append(_compute_mmd_kernel(s_eval, t_eval))
        
        # Wasserstein (POT)
        results["wasserstein"].append(_compute_wasserstein_pot(s_eval, t_eval))
        
        # KL Divergence (KDE)
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
    

    s_flat = s_np.reshape(-1, s_np.shape[-1])
    t_flat = t_np.reshape(-1, t_np.shape[-1])
    
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


def evaluate_trajectory_quality(trajectories, safety_check_fn, check_index_list=None):
    """
    评估单组轨迹的安全性、物理平滑性指标。
    
    参数:
    - trajectories: (num_traj, seq_length, obs_dim) 原始轨迹数据
    - safety_check_fn: function, 输入完整轨迹，返回 bool array
    - check_index_list: list[int] or None, 指定用于计算平滑性的维度索引。
                        如果为 None，则使用所有维度。
                        Hopper 取值: [0, 1, 2, 3, 4]
    
    返回:
    - dict: {
        "safety_ratio": float,
        "curvature_smoothness": float, 
        "acc_smoothness": float
      }
    """
    traj_np = _to_numpy(trajectories) 
    

    total = len(traj_np)
    if total > 0:
        safe_flag_list = safety_check_fn(traj_np)
        safe_count = np.sum(safe_flag_list)
        safety_ratio = safe_count / total
    else:
        safety_ratio = 0.0
    
    
    if check_index_list is not None:
        calc_traj = traj_np[:, :, check_index_list]
    else:
        calc_traj = traj_np

    if calc_traj.shape[1] < 3:
        return {
            "safety_ratio": safety_ratio,
            "curvature_smoothness": float('nan'),
            "acc_smoothness": float('nan')
        }

    velocity = calc_traj[:, 1:, :] - calc_traj[:, :-1, :]
    
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
    
    acc_magnitude = np.linalg.norm(acceleration, axis=-1)
    acc_smoothness = np.mean(acc_magnitude)
    
    v_t = velocity[:, :-1, :]   
    v_t1 = velocity[:, 1:, :]  
    
    norm_v_t = np.linalg.norm(v_t, axis=-1)
    norm_v_t1 = np.linalg.norm(v_t1, axis=-1)
    
    eps = 1e-8
    dot_product = np.sum(v_t * v_t1, axis=-1)
    
    cosine_sim = dot_product / (norm_v_t * norm_v_t1 + eps)
    
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    angles = np.arccos(cosine_sim)
    curvature_smoothness = np.mean(angles)
    
    return {
        "safety_ratio": safety_ratio,
        "curvature_smoothness": curvature_smoothness,
        "acc_smoothness": acc_smoothness
    }