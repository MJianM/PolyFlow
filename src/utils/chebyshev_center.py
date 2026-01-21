import torch
import numpy as np


def chebyshev_center(A, b, tol=1e-6, max_iter=1000):
    """
    计算由 Ax <= b 定义的凸多面体的切比雪夫中心（最大内切球的中心）和半径
    
    参数：
        A: 形状为 (m, n) 的 torch.Tensor，约束矩阵
        b: 形状为 (m,) 的 torch.Tensor，约束向量
        tol: 收敛容忍度
        max_iter: 最大迭代次数
        
    返回：
        center: 切比雪夫中心，形状为 (n,) 的 torch.Tensor
        radius: 最大内切球半径，标量
    """
    m, n = A.shape
    
    # 计算 A 每一行的 L2 范数
    A_norms = torch.norm(A, dim=1, keepdim=True)
    
    # 构建线性规划问题的变量：x (n维) 和 r (1维)
    # 目标：最大化 r
    # 约束：A_i·x + ||A_i||_2 * r <= b_i
    
    # 使用线性规划求解：通过引入松弛变量转换为可行性问题
    # 我们可以使用二次规划或迭代方法求解
    
    # 方法：使用梯度下降法求解（适用于中等规模问题）
    # 最大化 r 等价于最小化 -r
    
    # 初始化变量
    x = torch.zeros(n, requires_grad=True)
    r = torch.tensor(0.1, requires_grad=True)  # 初始半径
    
    # 创建优化器
    optimizer = torch.optim.Adam([x, r], lr=0.1)
    
    best_solution = None
    best_radius = -float('inf')
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        # 计算约束违反程度
        constraints = A @ x + A_norms.squeeze() * r - b
        
        # 损失函数：最大化 r 同时满足约束
        # 使用拉格朗日乘子法的思想
        violation = torch.relu(constraints)  # 只考虑违反的约束
        loss = -r + 10.0 * torch.sum(violation**2)  # 惩罚违反的约束
        
        loss.backward()
        optimizer.step()
        
        # 强制 r >= 0
        with torch.no_grad():
            r.data = torch.clamp(r.data, min=0)
        
        # 检查是否满足所有约束（考虑数值容差）
        feasible = torch.all(constraints <= tol)
        
        if feasible and r.item() > best_radius:
            best_radius = r.item()
            best_solution = x.detach().clone()
        
        # 检查收敛
        if iteration > 10 and torch.abs(loss) < tol:
            break
    
    # 如果找到了可行解，使用最佳解
    if best_solution is not None:
        center = best_solution
        radius = best_radius
    else:
        # 如果没有找到可行解，使用当前解
        center = x.detach()
        radius = r.item()
    
    return center, radius


def chebyshev_center_lp(A, b, method='scipy'):
    """
    使用线性规划求解切比雪夫中心（更精确的方法）
    
    参数：
        A: 形状为 (m, n) 的 torch.Tensor 或 numpy.ndarray
        b: 形状为 (m,) 的 torch.Tensor 或 numpy.ndarray
        method: 求解方法，'scipy' 或 'cvxpy'
        
    返回：
        center: 切比雪夫中心
        radius: 最大内切球半径
    """
    # 转换为 numpy 数组（如果需要）
    if isinstance(A, torch.Tensor):
        A_np = A.cpu().numpy()
        b_np = b.cpu().numpy()
    else:
        A_np = A
        b_np = b
    
    m, n = A_np.shape
    
    try:
        if method == 'scipy':
            # 使用 scipy 的线性规划求解
            from scipy.optimize import linprog
            
            # 计算 A 每一行的 L2 范数
            A_norms = np.linalg.norm(A_np, axis=1)
            
            # 构建线性规划问题
            # 变量: [x_1, ..., x_n, r]
            # 目标: 最大化 r，等价于最小化 -r
            c = np.zeros(n + 1)
            c[-1] = -1  # 最大化 r
            
            # 约束: A_i·x + ||A_i||_2 * r <= b_i
            A_ub = np.hstack([A_np, A_norms.reshape(-1, 1)])
            b_ub = b_np
            
            # 变量边界: x 无界，r >= 0
            bounds = [(None, None) for _ in range(n)] + [(0, None)]
            
            # 求解线性规划
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                center = res.x[:-1].astype(np.float32)
                radius = res.x[-1]
                return center, radius
            else:
                raise ValueError(f"线性规划求解失败: {res.message}")
                
        elif method == 'cvxpy':
            # 使用 cvxpy 求解
            import cvxpy as cp
            
            # 定义变量
            x = cp.Variable(n)
            r = cp.Variable(nonneg=True)
            
            # 计算 A 的范数
            A_norms = np.linalg.norm(A_np, axis=1)
            
            # 约束
            constraints = []
            for i in range(m):
                constraints.append(A_np[i] @ x + A_norms[i] * r <= b_np[i])
            
            # 目标：最大化半径
            objective = cp.Maximize(r)
            
            # 求解问题
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                center = x.value.astype(np.float32)
                radius = r.value
                return center, radius
            else:
                raise ValueError(f"CVXPY 求解失败: {problem.status}")
                
    except ImportError as e:
        print(f"需要安装 {method} 包: {e}")
        print("使用梯度下降方法替代...")
        if isinstance(A, torch.Tensor):
            return chebyshev_center(A, b)
        else:
            return chebyshev_center(torch.from_numpy(A_np), torch.from_numpy(b_np))
        
def uniform_sample_in_ball(center, radius, num_samples=1, device=None, dtype=None):
    """
    在 n 维球体内均匀采样 (Optimized)
    
    参数：
        center: 球心，形状为 (n,) 的 torch.Tensor
        radius: 球半径，标量 (float 或 tensor)
        num_samples: 采样数量 (int)
        device: 指定设备 (如果 center 是 Tensor，则忽略此参数，自动对齐 center)
        dtype: 指定数据类型 (同上)
        
    返回：
        samples: 形状为 (num_samples, n) 的 torch.Tensor

    """
    # 1. 输入处理与设备/类型对齐
    # 确保 center 是 tensor，并获取其属性作为后续生成的基准
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, device=device, dtype=dtype)
    
    ctx_device = center.device
    ctx_dtype = center.dtype
    n = center.shape[-1]
    
    # 2. 生成标准正态分布样本 (Direction)
    # 使用无梯度的上下文，减少不必要的内存开销（通常采样不需要反向传播）
    with torch.no_grad():
        z = torch.randn(num_samples, n, device=ctx_device, dtype=ctx_dtype)
        
        # 3. 归一化得到均匀球面方向
        # 添加 eps 防止极小概率的除零错误 (虽在 float32/64 下几乎不可能，但在工程上是好习惯)
        z_norms = torch.norm(z, dim=1, keepdim=True)
        directions = z / (z_norms + 1e-6)
        
        # 4. 生成均匀半径 scaling
        # 半径采样分布需遵循 r ~ U[0,1]^(1/n) 以保证体积均匀
        u = torch.rand(num_samples, 1, device=ctx_device, dtype=ctx_dtype)
        radii_scale = torch.pow(u, 1.0 / n)
        
        # 5. 合成
        # 利用广播机制: (n,) + scalar * (N, 1) * (N, n)
        samples = center + radius * radii_scale * directions
    
    # 6. 返回处理
    return samples


def uniform_sample_in_polytope(A, b, num_samples=1000, max_trials=10000):
    """
    在凸多面体 Ax <= b 内均匀采样（拒绝采样法）
    
    参数：
        A: 形状为 (m, n) 的 torch.Tensor
        b: 形状为 (m,) 的 torch.Tensor
        num_samples: 需要采样的点数
        max_trials: 最大尝试次数
        
    返回：
        samples: 形状为 (num_samples, n) 的 torch.Tensor
        acceptance_rate: 接受率
    """
    m, n = A.shape
    
    # 计算多面体的边界框
    with torch.no_grad():
        # 通过求解线性规划找到每个维度的最小值和最大值
        bounds = []
        for i in range(n):
            # 最小化 x_i
            c_min = torch.zeros(n)
            c_min[i] = 1
            x_min, _ = chebyshev_center(A, b - A @ c_min)  # 简化近似
            bounds.append(x_min[i].item())
            
            # 最大化 x_i
            c_max = torch.zeros(n)
            c_max[i] = -1
            x_max, _ = chebyshev_center(A, b - A @ c_max)  # 简化近似
            bounds.append(x_max[i].item())
    
    # 使用拒绝采样
    samples = []
    trials = 0
    accepted = 0
    
    while accepted < num_samples and trials < max_trials:
        # 在边界框内均匀采样
        trial_sample = torch.rand(n)
        for i in range(n):
            low = min(bounds[2*i], bounds[2*i+1])
            high = max(bounds[2*i], bounds[2*i+1])
            trial_sample[i] = low + (high - low) * trial_sample[i]
        
        # 检查是否满足所有约束
        if torch.all(A @ trial_sample <= b):
            samples.append(trial_sample)
            accepted += 1
        
        trials += 1
    
    acceptance_rate = accepted / trials if trials > 0 else 0
    
    if accepted < num_samples:
        print(f"警告: 只找到了 {accepted} 个样本 (接受率: {acceptance_rate:.2%})")
    
    return torch.stack(samples) if samples else torch.empty(0, n), acceptance_rate