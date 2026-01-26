import torch
import numpy as np


def chebyshev_center(A, b, tol=1e-6, max_iter=1000):
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
                center = torch.from_numpy(res.x[:-1].astype(np.float32))
                radius = res.x[-1]
                return center, radius
            else:
                raise ValueError(f"Linear Solving Failure: {res.message}")
                
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
                center = torch.from_numpy(x.value.astype(np.float32))
                radius = r.value
                return center, radius
            else:
                raise ValueError(f"CVXPY Failure: {problem.status}")
                
    except ImportError as e:
        print(f"need to install {method} package: {e}")
        print("using gradient descent instead...")
        if isinstance(A, torch.Tensor):
            return chebyshev_center(A, b)
        else:
            return chebyshev_center(torch.from_numpy(A_np), torch.from_numpy(b_np))
        
def uniform_sample_in_ball(center, radius, num_samples=1):
    n = center.shape[0]
    
    z = torch.randn(num_samples, n)

    norms = torch.norm(z, dim=1, keepdim=True)
    directions = z / norms
    
    u = torch.rand(num_samples, 1)
    radii = torch.pow(u, 1.0 / n)

    samples = center + radius * radii * directions
    
    return samples if num_samples > 1 else samples.squeeze(0)


def uniform_sample_in_polytope(A, b, num_samples=1000, max_trials=10000):
    m, n = A.shape
    
    with torch.no_grad():
        bounds = []
        for i in range(n):
            c_min = torch.zeros(n)
            c_min[i] = 1
            x_min, _ = chebyshev_center(A, b - A @ c_min)  
            bounds.append(x_min[i].item())
            
            c_max = torch.zeros(n)
            c_max[i] = -1
            x_max, _ = chebyshev_center(A, b - A @ c_max)  
            bounds.append(x_max[i].item())
    
    samples = []
    trials = 0
    accepted = 0
    
    while accepted < num_samples and trials < max_trials:
        trial_sample = torch.rand(n)
        for i in range(n):
            low = min(bounds[2*i], bounds[2*i+1])
            high = max(bounds[2*i], bounds[2*i+1])
            trial_sample[i] = low + (high - low) * trial_sample[i]
        
        if torch.all(A @ trial_sample <= b):
            samples.append(trial_sample)
            accepted += 1
        
        trials += 1
    
    acceptance_rate = accepted / trials if trials > 0 else 0
    
    if accepted < num_samples:
        print(f"Warning: Only find {accepted} samples (acceptance rate: {acceptance_rate:.2%})")
    
    return torch.stack(samples) if samples else torch.empty(0, n), acceptance_rate