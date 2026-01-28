import torch
import numpy as np


def chebyshev_center(A, b, tol=1e-6, max_iter=1000):
    """
    
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
    
    A_norms = torch.norm(A, dim=1, keepdim=True)
    

    x = torch.zeros(n, requires_grad=True)
    r = torch.tensor(0.1, requires_grad=True)  
    
    optimizer = torch.optim.Adam([x, r], lr=0.1)
    
    best_solution = None
    best_radius = -float('inf')
    
    for iteration in range(max_iter):
        optimizer.zero_grad()
        
        constraints = A @ x + A_norms.squeeze() * r - b
        
        violation = torch.relu(constraints)  
        loss = -r + 10.0 * torch.sum(violation**2) 
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            r.data = torch.clamp(r.data, min=0)
        
        feasible = torch.all(constraints <= tol)
        
        if feasible and r.item() > best_radius:
            best_radius = r.item()
            best_solution = x.detach().clone()
        
        if iteration > 10 and torch.abs(loss) < tol:
            break
    
    if best_solution is not None:
        center = best_solution
        radius = best_radius
    else:
        center = x.detach()
        radius = r.item()
    
    return center, radius


def chebyshev_center_lp(A, b, method='scipy'):
    """
    
    参数：
        A: 形状为 (m, n) 的 torch.Tensor 或 numpy.ndarray
        b: 形状为 (m,) 的 torch.Tensor 或 numpy.ndarray
        method: 求解方法，'scipy' 或 'cvxpy'
        
    返回：
        center: 切比雪夫中心
        radius: 最大内切球半径
    """

    if isinstance(A, torch.Tensor):
        A_np = A.cpu().numpy()
        b_np = b.cpu().numpy()
    else:
        A_np = A
        b_np = b
    
    m, n = A_np.shape
    
    try:
        if method == 'scipy':

            from scipy.optimize import linprog
            
            A_norms = np.linalg.norm(A_np, axis=1)
            
            c = np.zeros(n + 1)
            c[-1] = -1  

            A_ub = np.hstack([A_np, A_norms.reshape(-1, 1)])
            b_ub = b_np
            
            bounds = [(None, None) for _ in range(n)] + [(0, None)]
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                center = res.x[:-1].astype(np.float32)
                radius = res.x[-1]
                return center, radius
            else:
                raise ValueError(f"线性规划求解失败: {res.message}")
                
        elif method == 'cvxpy':
            import cvxpy as cp
            
            x = cp.Variable(n)
            r = cp.Variable(nonneg=True)
            
            A_norms = np.linalg.norm(A_np, axis=1)
            
            constraints = []
            for i in range(m):
                constraints.append(A_np[i] @ x + A_norms[i] * r <= b_np[i])
            
            objective = cp.Maximize(r)
            
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
    
    参数：
        center: 球心，形状为 (n,) 的 torch.Tensor
        radius: 球半径，标量 (float 或 tensor)
        num_samples: 采样数量 (int)
        device: 指定设备 (如果 center 是 Tensor，则忽略此参数，自动对齐 center)
        dtype: 指定数据类型 (同上)
        
    返回：
        samples: 形状为 (num_samples, n) 的 torch.Tensor

    """

    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, device=device, dtype=dtype)
    
    ctx_device = center.device
    ctx_dtype = center.dtype
    n = center.shape[-1]
    
    with torch.no_grad():
        z = torch.randn(num_samples, n, device=ctx_device, dtype=ctx_dtype)
        
        z_norms = torch.norm(z, dim=1, keepdim=True)
        directions = z / (z_norms + 1e-6)
        
        u = torch.rand(num_samples, 1, device=ctx_device, dtype=ctx_dtype)
        radii_scale = torch.pow(u, 1.0 / n)
        
        samples = center + radius * radii_scale * directions
    
    return samples

