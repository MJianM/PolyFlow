from typing import List
from collections import namedtuple
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
from tqdm import tqdm

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    HAS_CVXPYLAYERS = True
except ImportError:
    HAS_CVXPYLAYERS = False
    print("Warning: cvxpylayers not found. GPU batch solving will fail.")


class SafeFlowSampler:
    def __init__(self, model, obstacles, device='cuda'):
        """
        :param model: 训练好的 TrajectoryDiT
        :param obstacles: 列表，每个元素为 [x_c, y_c, a, b, n]
        """
        self.model = model
        # 将障碍物转换为 Tensor 并移动到 GPU: (N_obs, 4)
        self.obstacles = obstacles
        self.obstacles_tensor = torch.tensor(obstacles, device=device, dtype=torch.float32)
        
        self.device = device
        self.model.eval()
        
        # 初始化 Batch QP Solver
        if HAS_CVXPYLAYERS:
            self.init_qp_solver()

    def init_qp_solver(self):
        """
        使用 cvxpy 定义单次 QP 问题结构，CvxpyLayer 会自动处理 Batch 维度
        """
        x_dim = 2
        n_obs = len(self.obstacles)
        
        # 定义优化变量 (针对单个样本)
        u = cp.Variable(x_dim)      # control correction
        delta = cp.Variable(n_obs)  # slack variables
        
        # 定义参数 (这些将在 forward 时传入)
        # 约束形式: grad^T * u + delta >= lower_bound
        # 其中 lower_bound = - (grad^T * v + phi * h)
        param_grad = cp.Parameter((n_obs, x_dim)) # 对应 grad h
        param_bound = cp.Parameter(n_obs)         # 对应 RHS 下界
        
        # 目标函数: min ||u||^2 + M * ||delta||^2
        # M 取 1e6 保证优先满足约束
        objective = cp.Minimize(cp.sum_squares(u) + 1.0 * cp.sum_squares(delta))
        
        # 约束条件
        constraints = [
            param_grad @ u + delta >= param_bound, # CBF 约束 [cite: 247]
            delta >= 0                             # 松弛变量非负
        ]
        
        problem = cp.Problem(objective, constraints)
        self.qp_layer = CvxpyLayer(problem, parameters=[param_grad, param_bound], variables=[u, delta])

    def get_h_and_grad_tensor(self, x_batch):
        """
        向量化计算 Barrier Function h(x) 及其梯度 (PyTorch 实现)
        x_batch: [N, 2]  (N = Batch * Horizon)
        Returns:
            h_vals: [N, n_obs]
            grads:  [N, n_obs, 2]
        """
        # x_batch: (N, 1, 2)
        # obstacles: (1, n_obs, 5)
        x_exp = x_batch.unsqueeze(1) 
        obs_exp = self.obstacles_tensor.unsqueeze(0)
        
        xc = obs_exp[..., 0] # (1, n_obs)
        yc = obs_exp[..., 1]
        a  = obs_exp[..., 2]
        b  = obs_exp[..., 3]
        n = obs_exp[..., 4]
        
        px = x_exp[..., 0] # (N, 1)
        py = x_exp[..., 1]
        
        # 1. 计算相对距离 (diff)
        dx = px - xc  # (N, n_obs)
        dy = py - yc
        
        # 2. 计算 h(x) 使用绝对值，防止底数为负导致的 NaN
        # h = |dx/a|^n + |dy/b|^n - 1
        abs_dx_a = torch.abs(dx / a)
        abs_dy_b = torch.abs(dy / b)
        
        term_x = torch.pow(abs_dx_a, n)
        term_y = torch.pow(abs_dy_b, n)
        
        h_vals = term_x + term_y - 1.0
        
        # 3. 计算梯度
        # d(|u|^n)/dx = n * |u|^(n-1) * sgn(u) * (1/a)
        # 这里 u = dx/a. 也就是: n/a * |dx/a|^(n-1) * sgn(dx/a)
        # 注意: sgn(dx/a) 和 sgn(dx) 是一样的（假设 a > 0）
        
        grad_x = (n / a) * torch.pow(abs_dx_a, n - 1) * torch.sign(dx)
        grad_y = (n / b) * torch.pow(abs_dy_b, n - 1) * torch.sign(dy)
        
        grads = torch.stack([grad_x, grad_y], dim=-1) # (N, n_obs, 2)

        return h_vals, grads

    def phi_function_tensor(self, t, h_vals, gamma=0.0):
        """
        向量化 Blow-up function 
        """
        T_end = 1.0
        margin = 1e-5
        # 标量 t 转换为 tensor，或者直接使用标量计算
        
        # 1. 计算 phi_1(t) (Blow-up term)
        if t < gamma:
            phi_1 = 1 + 4 * (t ** 3)
        else:
            effective_t = min(t, T_end - margin)
            phi_1 = 1.0 / (T_end - effective_t)
            # if phi_1 > 1000:
            #     phi_1 = 1000
            
        # 2. 根据 h 值选择 phi
        # 如果 h >= 0, phi = 1.0; 否则 phi = phi_1
        # phi_vals 形状与 h_vals 相同: [N, n_obs]
        phi_vals = torch.where(h_vals >= 0, torch.tensor(1.0, device=self.device), torch.tensor(phi_1, device=self.device))
        
        return phi_vals

    def solve_qp_batch(self, v_nom, h_vals, grads, phi_vals):
        """
        Batch QP 求解
        v_nom: [N, 2]
        h_vals: [N, n_obs]
        grads: [N, n_obs, 2]
        phi_vals: [N, n_obs]
        """
        if not HAS_CVXPYLAYERS:
            return torch.zeros_like(v_nom)

        # 构造 QP 参数
        # 约束: grad^T * u + delta >= - (grad^T * v + phi * h)
        # Let bound = - (grad^T * v + phi * h)
        
        # grad^T * v: (N, n_obs, 2) * (N, 1, 2) -> sum -> (N, n_obs)
        lie_deriv = torch.sum(grads * v_nom.unsqueeze(1), dim=-1)
        
        # lower_bound: [N, n_obs]
        param_bound = - (lie_deriv + phi_vals * h_vals)
        
        # param_grad: [N, n_obs, 2]
        param_grad = grads
        
        # 调用 CvxpyLayer (GPU Batch Solve)
        # 注意: cvxpylayers 可能会抛出无解异常(Infeasible)，但在我们的松弛变量设置下应该总是有解
        try:
            u_star, delta_star = self.qp_layer(param_grad, param_bound)
            return u_star # [N, 2]
        except Exception as e:
            print(f"QP Batch Solve Error: {e}")
            return torch.zeros_like(v_nom)

    @torch.no_grad()
    def sample(self, n_samples, horizon, steps=100, use_cbf=True, use_closed_form=True):
        """
        执行欧拉积分进行采样 (Fully Vectorized)
        """
        x = torch.randn(n_samples, horizon, 2).to(self.device)
        dt = 1.0 / steps
        
        for i in range(steps):
            t_curr = i * dt
            t_tensor = torch.full((n_samples,), t_curr, device=self.device)
            
            # 1. 预测名义向量场
            v_pred = self.model(x, t_tensor) # [B, H, 2]
            
            correction = torch.zeros_like(v_pred)
            
            if use_cbf:
                # 2. 准备 Batch 数据: Flatten (B, H) -> (N)
                N = n_samples * horizon
                x_flat = x.view(N, 2)
                v_flat = v_pred.view(N, 2)
                
                # 3. 计算 h, grad, phi (全 Tensor 操作)
                h_vals, grads = self.get_h_and_grad_tensor(x_flat) # [N, n_obs], [N, n_obs, 2]
                
                # 简单剪枝: 如果所有障碍物都很远(h > safe_margin)，则不需要解 QP
                # 为了保持 Batch 维度一致性，通常不剪枝，或者只对 mask 内的解 QP
                # 这里为了简单直接全解 (cvxpylayers 效率较高)
                # 实际上只有当 h < 0 或接近 0 时 QP 才会产生非零 u
                
                phi_vals = self.phi_function_tensor(t_curr, h_vals) # [N, n_obs]
                
                # 4. Batch QP 求解
                if use_closed_form:
                    u_flat = self.solve_batch_closed_form(v_flat, h_vals, grads, phi_vals)
                else:
                    u_flat = self.solve_qp_batch(v_flat, h_vals, grads, phi_vals)
                
                # Reshape back
                correction = u_flat.view(n_samples, horizon, 2)

            # 5. 更新状态 
            final_vel = v_pred + correction
            x = x + final_vel * dt
            
        return x

    def _runge_kutta_step(self, func, t, x, dt):
        """
        执行单步 Dormand-Prince (RK45) 积分
        :param func: 动力学函数 f(t, x) -> v
        :param t: 当前时间
        :param x: 当前状态
        :param dt: 步长
        :return: (x_next_5th, error_estimate)
        """
        # Dormand-Prince 参数 (Butcher Tableau)
        # c: 时间节点
        c2, c3, c4, c5 = 1/5, 3/10, 4/5, 8/9
        c6 = 1.0
        
        # a: 状态系数
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        
        # b: 5阶解权重 (b1 等同于 v5 的组合)
        b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
        # b*: 4阶解权重 (用于误差估计)
        bp1, bp3, bp4, bp5, bp6 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100
        # error coefficients E = b - b*
        e1, e3, e4, e5, e6 = b1-bp1, b3-bp3, b4-bp4, b5-bp5, b6-bp6

        # K1
        k1 = func(t, x) * dt
        
        # K2
        k2 = func(t + c2*dt, x + a21*k1) * dt
        
        # K3
        k3 = func(t + c3*dt, x + a31*k1 + a32*k2) * dt
        
        # K4
        k4 = func(t + c4*dt, x + a41*k1 + a42*k2 + a43*k3) * dt
        
        # K5
        k5 = func(t + c5*dt, x + a51*k1 + a52*k2 + a53*k3 + a54*k4) * dt
        
        # K6
        k6 = func(t + dt, x + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5) * dt

        # 5th order solution
        x_next = x + b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6
        
        # Error estimate (difference between 4th and 5th order)
        # error = sum( (b_i - bp_i) * k_i )
        # Note: k2 is not used in the final summation for 5th order, but error calculation usually involves it or simplifies
        # Standard implementation uses the difference directly:
        error = e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6
        
        # Calculate max absolute error per trajectory
        # error shape: [N, 2] -> scalar (max over dimensions and batch)
        # 这里为了更精细控制，我们返回每个样本的误差范数
        error_norm = torch.norm(error, dim=-1) # [N]
        
        return x_next, error_norm

    @torch.no_grad()
    def sample_rk45(self, n_samples, horizon, rtol=1e-5, atol=1e-5, use_cbf=True, use_closed_form=True):
        """
        实现论文 Algorithm 1: 基于自适应 RK45 的安全采样
        """
        # 1. Initialize [Algorithm 1, Line 2]
        x = torch.randn(n_samples, horizon, 2).to(self.device)
        
        t = 0.0
        dt = 0.001 # Initial integration step [Algorithm 1, Line 3]
        
        # Flatten batch for efficiency
        N = n_samples * horizon
        x_flat = x.view(N, 2)
        
        # 统计信息
        steps_count = 0
        
        pbar = tqdm(total=1000, desc="RK45 Sampling") # approximate progress
        last_t_disp = 0
        
        while t < 1.0:
            # 确保最后一步正好到达 1.0
            if t + dt > 1.0:
                dt = 1.0 - t
            
            # --- A. 计算当前时间步的安全修正量 u_t [Algorithm 1, Line 8] ---
            # 这是一个 "Zero-Order Hold" 策略：计算一次 u，在本步 RK 积分中保持不变
            t_tensor = torch.full((n_samples,), t, device=self.device)
            v_pred_curr = self.model(x_flat.view(n_samples, horizon, 2), t_tensor).view(N, 2)
            
            u_correction = torch.zeros_like(v_pred_curr)
            
            if use_cbf:
                # 1. Denormalization (Line 6-7): 
                # 假设当前 x 和 v 已经在 state space (如果需要归一化，请在此处添加 transform)
                
                # 2. CFMBF-based QP (Line 8)
                h_vals, grads = self.get_h_and_grad_tensor(x_flat)
                phi_vals = self.phi_function_tensor(t, h_vals)
                
                # Batch QP
                if use_closed_form:
                    u_correction = self.solve_batch_closed_form(v_pred_curr, h_vals, grads, phi_vals)
                else:
                    u_correction = self.solve_qp_batch(v_pred_curr, h_vals, grads, phi_vals)
                
                # 3. Normalization (Line 9): 
                # 如果有归一化，将 u_correction 转回 normalized space
            
            # --- B. 定义组合动力学函数 v_total = v_theta + u ---
            # 在 RK 的中间步骤 (t + c*dt) 中：
            # 1. 神经网络 v_theta(t', x') 需要根据新的 t' 和 x' 重新计算 (Flow Matching 本质)
            # 2. 安全修正 u_correction 保持为常数 (Algorithm 1 Line 10: "using learned v + u_hat")
            def dynamics_func(t_scalar, x_curr_flat):
                t_vec = torch.full((n_samples,), t_scalar, device=self.device)
                # Reshape for model input
                v_nn = self.model(x_curr_flat.view(n_samples, horizon, 2), t_vec).view(N, 2)
                return v_nn + u_correction # Guidance

            # --- C. 执行 RK45 积分步 [Algorithm 1, Line 10-11] ---
            x_new_flat, error_norms = self._runge_kutta_step(dynamics_func, t, x_flat, dt)
            
            # --- D. 误差检查与步长自适应 [Algorithm 1, Line 12-15] ---
            # 计算允许误差 tolerance = atol + rtol * |x|
            # 使用无穷范数或 2-范数均可
            x_norm = torch.norm(x_flat, dim=-1)
            x_new_norm = torch.norm(x_new_flat, dim=-1)
            max_norm = torch.max(x_norm, x_new_norm)
            tolerance = atol + rtol * max_norm
            
            # error_ratio = error / tolerance
            error_ratio = error_norms / tolerance
            max_error_ratio = torch.max(error_ratio).item()
            
            if max_error_ratio <= 1.0:
                # 1. Accept Step [Algorithm 1, Line 12-13]
                x_flat = x_new_flat
                t += dt
                steps_count += 1
                
                # 更新进度条
                pbar.update(int((t - last_t_disp) * 1000))
                last_t_disp = t
                
                # 2. Increase Step Size (Limit max growth to 5x to be safe)
                # Formula: dt_new = dt * safety * (1 / error_ratio)^(1/5)
                # Safety factor usually 0.9
                if max_error_ratio < 1e-4: # Avoid division by zero or huge steps
                    factor = 5.0
                else:
                    factor = 0.9 * (1.0 / max_error_ratio) ** 0.2
                    factor = min(factor, 5.0) # Cap growth
                
                dt *= factor
                
            else:
                # 3. Reject Step & Decrease Step Size
                # Formula: dt_new = dt * safety * (1 / error_ratio)^(1/5)
                factor = 0.9 * (1.0 / max_error_ratio) ** 0.2
                factor = max(factor, 0.1) # Don't shrink too fast (min 0.1x)
                dt *= factor
                
                # 防止步长过小导致死循环
                if dt < 1e-7:
                    # 如果步长太小，强制接受 (或者抛出警告)
                    print(f"Warning: Step size too small at t={t:.4f}, forcing step.")
                    x_flat = x_new_flat
                    t += 1e-7
        
        pbar.close()
        
        # Reshape back
        x_final = x_flat.view(n_samples, horizon, 2)
        
        # --- E. Terminal Safety Filter [Algorithm 1, Line 17] ---
        # 论文最后建议做一个投影，确保最终点严格满足约束
        if use_cbf:
            x_final = self.terminal_safety_filter(x_final)
            
        return x_final

    def terminal_safety_filter(self, x):
        """
        Algorithm 1, Line 17: 最终时刻的简单的投影修正
        求解: min ||x - x_end||^2 s.t. h(x) >= 0
        """
        # 简单实现：检查每个点，如果不安全，沿梯度推到边界
        # 这是一个简化的 Filter，严谨的话应该解一个 Optimization
        # 这里使用迭代投影法
        x_flat = x.view(-1, 2)
        
        # 迭代几次投影
        for _ in range(5):
            h_vals, grads = self.get_h_and_grad_tensor(x_flat)
            
            # 找出不安全的点 (h < 0)
            unsafe_mask = h_vals < 0 # [N, n_obs]
            
            if not unsafe_mask.any():
                break
                
            # 对每个不安全的约束，计算修正量 dx = - h * grad / ||grad||^2
            # 这是一个一阶近似投影
            grad_norm_sq = torch.sum(grads**2, dim=-1) + 1e-8
            
            # 修正向量 [N, n_obs, 2]
            # dx = - h * grad / norm
            correction = - (h_vals.unsqueeze(-1) * grads) / grad_norm_sq.unsqueeze(-1)
            
            # 只应用不安全的修正
            correction = correction * unsafe_mask.unsqueeze(-1).float()
            
            # 累加修正量 (简单的 sum 可能会震荡，但对于凸集通常有效)
            total_correction = torch.sum(correction, dim=1)
            
            x_flat = x_flat + total_correction
            
        return x_flat.view(x.shape)

    def solve_batch_closed_form(self, v_nom, h_vals, grads, phi_vals):
        """
        基于论文公式 (16) 实现的 Batch 闭式解。
        完全使用 PyTorch Tensor 操作，极快。
        
        v_nom: [N, 2]
        h_vals: [N, n_obs]
        grads: [N, n_obs, 2]
        phi_vals: [N, n_obs]
        """
        # 1. 计算系数 a [N, n_obs]
        # a = grad^T * v + phi * h
        # grads: [N, n_obs, 2], v_nom: [N, 1, 2]
        lie_deriv = torch.sum(grads * v_nom.unsqueeze(1), dim=-1) # [N, n_obs]
        a = lie_deriv + phi_vals * h_vals
        
        # 2. 计算分母 ||b||^2 = ||grad||^2 [N, n_obs]
        b_norm_sq = torch.sum(grads ** 2, dim=-1)
        # 防止除零
        b_norm_sq = torch.clamp(b_norm_sq, min=1e-6)
        
        # 3. 计算单一约束下的最优 u 
        # 公式 (16): u = max(0, -a/||b||^2) * b
        # 如果 a >= 0 (安全), u = 0
        # 如果 a < 0 (不安全), u = -a * b / ||b||^2
        
        lambda_val = -a / b_norm_sq     # [N, n_obs]
        mask = lambda_val > 0           # 只修正不满足约束的部分 (a < 0)
        
        # [N, n_obs] * [N, n_obs, 1] -> [N, n_obs, 2]
        u_corrections = mask.unsqueeze(-1).float() * lambda_val.unsqueeze(-1) * grads
        
        # 4. 处理复合约束 (Multiple Obstacles)
        # 策略 A: 简单累加 (Decoupled Sum) - 速度最快，论文提到了解耦的可能性
        # 虽然这不一定是联合 QP 的精确解，但在障碍物稀疏时效果很好
        # u_final = torch.sum(u_corrections, dim=1) # [N, 2]
        
        # 策略 B (可选): 只取模最大的那个修正 (应对最紧急的情况)
        norms = torch.norm(u_corrections, dim=-1)
        max_idx = torch.argmax(norms, dim=1)
        u_final = u_corrections[torch.arange(u_corrections.size(0)), max_idx]


        u_norm = torch.norm(u_final, dim=-1, keepdim=True)
        max_u = 1 # 根据迷宫尺度调整，比如设为 max_speed * 2
        clip_mask = u_norm > max_u
        u_final = torch.where(clip_mask, u_final / u_norm * max_u, u_final)
        
        return u_final
