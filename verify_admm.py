import torch
import numpy as np
from scipy.optimize import linprog
import time

# ==========================================
# 1. 修正版: PyTorch Batch ADMM Solver
# ==========================================
def solve_chebyshev_admm(A, b, rho=1.0, max_iter=200):
    """
    Args:
        A: (B, M, 3)
        b: (B, M)
        rho: ADMM 惩罚参数 (建议 1.0 - 5.0)
    """
    B, M, _ = A.shape
    device = A.device

    # 1. 构造矩阵 G = [A, ||a||]
    norm_A = torch.norm(A, dim=-1, keepdim=True) # (B, M, 1)
    G = torch.cat([A, norm_A], dim=-1)           # (B, M, 4)
    
    # 2. 预计算 (G^T G)^-1
    # ---------------------------------------------------------
    # 【核心修正】
    # 之前错误地加了 rho * I，导致结果被强制压缩向0。
    # 这里只加 1e-6 * I 防止奇异矩阵（数值稳定性），不改变物理意义。
    # ---------------------------------------------------------
    Gt = G.transpose(1, 2)
    GtG = torch.bmm(Gt, G) # (B, 4, 4)
    eps_I = torch.eye(4, device=device).unsqueeze(0) * 1e-6
    
    # K_mat = (G^T G)^-1
    K_mat = torch.linalg.inv(GtG + eps_I) 
    
    # 预计算 K = (G^T G)^-1 G^T，用于快速投影
    K = torch.bmm(K_mat, Gt) # (B, 4, M)
    
    # 3. 初始化变量
    # 问题形式: Minimize c_vec^T y  s.t.  G y + s = b,  s >= 0
    # y = [center, radius]
    y = torch.zeros(B, 4, 1, device=device) 
    s = torch.abs(torch.randn(B, M, 1, device=device)) # Slack variable (初始化为正)
    u = torch.zeros(B, M, 1, device=device) # Dual variable (Lagrange multiplier)
    
    # 目标向量: minimize -r  => c_vec = [0,0,0,-1]
    c_vec = torch.zeros(B, 4, 1, device=device)
    c_vec[:, 3, 0] = -1.0
    
    b_uns = b.unsqueeze(-1) # (B, M, 1)

    # 4. ADMM 主循环
    for k in range(max_iter):
        # --- y-update (无约束最小二乘) ---
        # 求解: G^T G y = G^T (b - s - u) - c_vec / rho
        # 解析解: y = K (b - s - u) - K_mat (c_vec / rho)
        
        target = b_uns - s - u
        
        term1 = torch.bmm(K, target)
        term2 = torch.bmm(K_mat, c_vec) * (1.0 / rho)
        
        y = term1 - term2
        
        # --- s-update (松弛变量投影) ---
        # minimize (rho/2) || G y + s - b + u ||^2  s.t. s >= 0
        # let v = b - G y - u
        # min || s - v ||^2  => s = max(0, v)
        
        Gy = torch.bmm(G, y)
        v = b_uns - Gy - u
        s = torch.relu(v) # 强制 s >= 0
        
        # --- u-update (对偶变量更新) ---
        # u = u + (G y + s - b)
        
        residual = Gy + s - b_uns
        u = u + residual
    
    # 5. 结果提取与安全修正
    center_est = y[:, :3, 0]
    
    # 使用几何公式计算严格半径，消除 ADMM 收敛残差的影响
    # r = min_i ( (b_i - a_i^T x) / ||a_i|| )
    ax = torch.bmm(A, center_est.unsqueeze(-1)).squeeze(-1)
    dists = (b - ax) / (norm_A.squeeze(-1) + 1e-8)
    
    # 取最小距离作为半径，并保证非负
    final_radius, _ = dists.min(dim=1, keepdim=True)
    final_radius = final_radius.clamp(min=1e-6)

    return center_est, final_radius

# ==========================================
# 2. 基准测试与 Scipy 对比 (保持不变)
# ==========================================
def solve_scipy_ground_truth(A_np, b_np):
    batch_size = A_np.shape[0]
    centers = []
    radii = []
    
    # print(f"Running Scipy on {batch_size} samples...")
    start_t = time.time()
    
    for i in range(batch_size):
        A_i = A_np[i]
        b_i = b_np[i]
        norm_a = np.linalg.norm(A_i, axis=1).reshape(-1, 1)
        
        c_lp = np.array([0, 0, 0, -1])
        A_lp = np.hstack((A_i, norm_a))
        
        res = linprog(c_lp, A_ub=A_lp, b_ub=b_i, bounds=[(None, None)]*3 + [(0, None)], method='highs')
        
        if res.success:
            centers.append(res.x[:3])
            radii.append(res.x[3])
        else:
            centers.append([0,0,0])
            radii.append(0.0)
            
    end_t = time.time()
    return np.array(centers), np.array(radii).reshape(-1, 1), end_t - start_t

def generate_random_polyhedra(batch_size, num_constraints, device):
    torch.manual_seed(123) # 固定种子方便复现
    A = torch.randn(batch_size, num_constraints, 3, device=device)
    A = A / torch.norm(A, dim=-1, keepdim=True)
    true_center = torch.randn(batch_size, 3, device=device)
    true_radius = torch.rand(batch_size, 1, device=device) + 0.5 # 0.5 ~ 1.5
    margin = torch.rand(batch_size, num_constraints, device=device) * 0.2
    ax = torch.bmm(A, true_center.unsqueeze(-1)).squeeze(-1)
    b = ax + true_radius + margin
    return A, b

def run_benchmark():
    BATCH_SIZE = 1000
    NUM_CONS = 10 # 稍微增加一点约束数量，测试鲁棒性
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # 1. 生成数据
    A_gpu, b_gpu = generate_random_polyhedra(BATCH_SIZE, NUM_CONS, DEVICE)
    
    # 2. Scipy Ground Truth
    A_cpu = A_gpu.cpu().numpy()
    b_cpu = b_gpu.cpu().numpy()
    gt_centers, gt_radii, t_scipy = solve_scipy_ground_truth(A_cpu, b_cpu)
    
    # 3. ADMM (Ours)
    solve_chebyshev_admm(A_gpu[:10], b_gpu[:10], max_iter=10) # Warmup
    
    print(f"Running ADMM Batch Solver...")
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    start_t = time.time()
    
    # rho=1.0 通常收敛最快，max_iter=200 保证精度
    est_centers, est_radii = solve_chebyshev_admm(A_gpu, b_gpu, rho=1.0, max_iter=200)
    
    torch.cuda.synchronize() if DEVICE == 'cuda' else None
    t_admm = time.time() - start_t
    
    # 4. 分析
    est_centers_np = est_centers.cpu().numpy()
    est_radii_np = est_radii.cpu().numpy()
    
    r_err = np.abs(gt_radii - est_radii_np)
    c_err = np.linalg.norm(gt_centers - est_centers_np, axis=1)
    
    print("\n" + "="*50)
    print(f"BENCHMARK REPORT (Batch: {BATCH_SIZE}, Cons: {NUM_CONS})")
    print("="*50)
    print(f"{'Metric':<20} | {'Scipy (Highs)':<15} | {'ADMM (Ours)':<15}")
    print("-" * 66)
    print(f"{'Time (Total)':<20} | {t_scipy:.4f} s        | {t_admm:.4f} s")
    print(f"{'Speedup':<20} | 1.0x            | {t_scipy/t_admm:.1f}x")
    print("-" * 66)
    print(f"Mean Radius Error    : {np.mean(r_err):.6f}")
    print(f"Max Radius Error     : {np.max(r_err):.6f}")
    print(f"Mean Center Distance : {np.mean(c_err):.6f}")
    print("-" * 66)
    
    # 验证 ADMM 结果的半径是否接近 Scipy 的最优解
    # 因为 ADMM 是数值迭代，且最后做了 clamp，所以半径理应略小于或等于 Scipy
    diff = gt_radii - est_radii_np
    print(f"Radius Gap (Gt - Est): Mean={np.mean(diff):.6f}, Max={np.max(diff):.6f}")
    print("(Gap > 0 means ADMM is conservative/safe. Small Gap is good.)")

if __name__ == "__main__":
    run_benchmark()