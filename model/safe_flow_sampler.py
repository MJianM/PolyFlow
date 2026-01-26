from typing import List
from collections import namedtuple
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
from tqdm import tqdm
try:
    from qpth.qp import QPFunction
    HAS_QPTH = True
except ImportError:
    HAS_QPTH = False
    print("Warning: qpth not installed. QP solving will be skipped.")

class SafeFlowSampler:
    def __init__(self, model, obstacles, device='cuda', clip_u = 10, clip_grad = 0.5, slack_penalty = 1.0):
        """
        :param model: trained TrajectoryDiT
        :param obstacles: list, [x_c, y_c, a, b, n]
        """
        self.model = model
        self.obstacles = obstacles
        self.obstacles_tensor = torch.tensor(obstacles, device=device, dtype=torch.float32)
        
        self.device = device
        self.model.eval()
        
        self.n_obs = len(obstacles)
        self.x_dim = 2
        
        self.M_penalty = slack_penalty 
        self.qp_eps = 1e-4    

        self.clip_u = clip_u
        self.clip_grad = clip_grad

    def get_h_and_grad_tensor(self, x_batch):
        """
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
        
        dx = px - xc  # (N, n_obs)
        dy = py - yc
        
        # h = |dx/a|^n + |dy/b|^n - 1
        abs_dx_a = torch.abs(dx / a)
        abs_dy_b = torch.abs(dy / b)
        
        term_x = torch.pow(abs_dx_a, n)
        term_y = torch.pow(abs_dy_b, n)
        
        h_vals = term_x + term_y - 1.0
        
        # d(|u|^n)/dx = n * |u|^(n-1) * sgn(u) * (1/a)
        # u = dx/a: n/a * |dx/a|^(n-1) * sgn(dx/a)
        
        grad_x = (n / a) * torch.pow(abs_dx_a, n - 1) * torch.sign(dx)
        grad_y = (n / b) * torch.pow(abs_dy_b, n - 1) * torch.sign(dy)
        
        grads = torch.stack([grad_x, grad_y], dim=-1) # (N, n_obs, 2)

        return h_vals, grads

    def phi_function_tensor(self, t, h_vals, gamma=0.9):
        """
        向量化 Blow-up function 
        """
        T_end = 1.0
        margin = 1e-5
        
        # calculate phi_1(t) (Blow-up term)
        if t < gamma:
            phi_1 = 1 + 4 * (t ** 3)
        else:
            effective_t = min(t, T_end - margin)
            phi_1 = 1.0 / (T_end - effective_t)
            # if phi_1 > 1000:
            #     phi_1 = 1000

        # [N, n_obs]
        phi_vals = torch.where(h_vals >= 0, torch.tensor(1.0, device=self.device), torch.tensor(phi_1, device=self.device))
        
        return phi_vals

    def solve_qp_batch(self, v_nom, h_vals, grads, phi_vals):
        """
        v_nom: [N, 2]
        h_vals: [N, n_obs]
        grads: [N, n_obs, 2]
        phi_vals: [N, n_obs]
        """
        if not HAS_QPTH:
            return torch.zeros_like(v_nom)

        N = v_nom.shape[0]
        K = self.n_obs
        total_vars = 2 + K # [u_x, u_y, delta_1...delta_k]
        
        Q_diag = torch.cat([
            torch.tensor([2.0, 2.0], device=self.device),
            torch.full((K,), 2.0 * self.M_penalty, device=self.device)
        ])
        
        Q = torch.diag_embed(Q_diag).unsqueeze(0).expand(N, -1, -1) + \
            torch.eye(total_vars, device=self.device).unsqueeze(0) * 1e-6

        p = torch.zeros(N, total_vars, device=self.device)

        # (1) CBF: -grad^T u - delta <= -B
        # (2) Slack Positivity: -delta <= 0
        
        # B (lower_bound) = - (grad^T v + phi * h)
        lie_deriv = torch.sum(grads * v_nom.unsqueeze(1), dim=-1) # (N, K)
        lower_bound = - (lie_deriv + phi_vals * h_vals) # (N, K)

        h_part1 = -lower_bound 
        h_part2 = torch.zeros(N, K, device=self.device)
        h_ineq = torch.cat([h_part1, h_part2], dim=1) # (N, 2K)
        
        I_K = torch.eye(K, device=self.device).unsqueeze(0).expand(N, -1, -1) # (N, K, K)
        
        # Block 1 (CBF)
        G_1_u = -grads # (N, K, 2)
        G_1_delta = -I_K 
        G_1 = torch.cat([G_1_u, G_1_delta], dim=2) # (N, K, 2+K)
        
        # Block 2 (Slack)
        G_2_u = torch.zeros(N, K, 2, device=self.device)
        G_2_delta = -I_K
        G_2 = torch.cat([G_2_u, G_2_delta], dim=2) # (N, K, 2+K)
        
        G = torch.cat([G_1, G_2], dim=1) # (N, 2K, 2+K)
        e = torch.Tensor().to(self.device)

        dtype_orig = v_nom.dtype
        Q = Q.double()
        p = p.double()
        G = G.double()
        h_ineq = h_ineq.double()
        e = torch.Tensor().double().to(self.device)

        try:
            # QPFunction(verbose=False)(Q, p, G, h, A, b)
            # z shape: (N, 2+K)
            z = QPFunction(verbose=False, eps=1e-3)(Q, p, G, h_ineq, e, e)

            u_star = z[:, :2].to(dtype=dtype_orig)
            
            u_norm = torch.norm(u_star, p=2, dim=-1, keepdim=True)
            max_correction = self.clip_u  
            scale = torch.clamp(max_correction / (u_norm + 1e-6), max=1.0)
            u_star = u_star * scale
            return u_star

        except Exception as e:
            print(f"QP Error: {e}")
            return torch.zeros_like(v_nom)

    @torch.no_grad()
    def sample(self, n_samples, horizon, steps=100, use_cbf=True, use_closed_form=True):
        x = torch.randn(n_samples, horizon, 2).to(self.device)
        dt = 1.0 / steps
        
        for i in trange(steps):
            t_curr = i * dt
            t_tensor = torch.full((n_samples,), t_curr, device=self.device)
            
            v_pred = self.model(x, t_tensor) # [B, H, 2]
            
            correction = torch.zeros_like(v_pred)
            
            if use_cbf:
                # Flatten (B, H) -> (N)
                N = n_samples * horizon
                x_flat = x.view(N, 2)
                v_flat = v_pred.view(N, 2)
                
                h_vals, grads = self.get_h_and_grad_tensor(x_flat) # [N, n_obs], [N, n_obs, 2]
                
                grad_norm = torch.norm(grads, dim=-1, keepdim=True)
                grads = grads / (grad_norm + 1e-6) * torch.clamp(grad_norm, max=self.clip_grad)
                
                phi_vals = self.phi_function_tensor(t_curr, h_vals) # [N, n_obs]
                
                if use_closed_form:
                    u_flat = self.solve_batch_closed_form(v_flat, h_vals, grads, phi_vals)
                else:
                    u_flat = self.solve_qp_batch(v_flat, h_vals, grads, phi_vals)
                
                # Reshape back
                correction = u_flat.view(n_samples, horizon, 2)

            final_vel = v_pred + correction
            x = x + final_vel * dt
            
        return x

    def solve_cbf_velocity(self, x, t_val, dt, use_cbf=True, use_closed_form=False):
        t_tensor = torch.full((x.shape[0],), t_val, device=self.device)
        v_pred = self.model(x, t_tensor)
        
        correction = torch.zeros_like(v_pred)

        n_samples = x.shape[0]
        horizon = x.shape[1]
        
        if use_cbf:
            #  Flatten (B, H) -> (N)
            N = n_samples * horizon
            x_flat = x.view(N, 2)
            v_flat = v_pred.view(N, 2)
            
            h_vals, grads = self.get_h_and_grad_tensor(x_flat) # [N, n_obs], [N, n_obs, 2]
            
            grad_norm = torch.norm(grads, dim=-1, keepdim=True)
            grads = grads / (grad_norm + 1e-6) * torch.clamp(grad_norm, max=10.0)
            
            phi_vals = self.phi_function_tensor(t_val, h_vals) # [N, n_obs]
            
            if use_closed_form:
                u_flat = self.solve_batch_closed_form(v_flat, h_vals, grads, phi_vals)
            else:
                u_flat = self.solve_qp_batch(v_flat, h_vals, grads, phi_vals)
            
            # Reshape back
            correction = u_flat.view(n_samples, horizon, 2)

        final_vel = v_pred + correction
        
        return final_vel

    @torch.no_grad()
    def sample_rk4(self, n_samples, horizon, steps=100, use_cbf=True, use_closed_form=False):
        x = torch.randn(n_samples, horizon, 2).to(self.device)
        dt = 1.0 / steps
        
        for i in trange(steps):
            t = i * dt
            
            # k1
            v1 = self.solve_cbf_velocity(x, t, dt, use_cbf=use_cbf, use_closed_form=use_closed_form)
            
            # k2
            v2 = self.solve_cbf_velocity(x + 0.5 * dt * v1, t + 0.5 * dt, dt, use_cbf=use_cbf, use_closed_form=use_closed_form)
            
            # k3
            v3 = self.solve_cbf_velocity(x + 0.5 * dt * v2, t + 0.5 * dt, dt, use_cbf=use_cbf, use_closed_form=use_closed_form)
            
            # k4
            v4 = self.solve_cbf_velocity(x + dt * v3, t + dt, dt, use_cbf=use_cbf, use_closed_form=use_closed_form)
            
            x = x + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
            
        return x


    def _runge_kutta_step(self, func, t, x, dt):
        c2, c3, c4, c5 = 1/5, 3/10, 4/5, 8/9
        c6 = 1.0

        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        
        b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
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
        error_norm = torch.norm(error, dim=-1) # [N]
        
        return x_next, error_norm

    @torch.no_grad()
    def sample_rk45(self, n_samples, horizon, rtol=1e-5, atol=1e-5, use_cbf=True, use_closed_form=True):
        # 1. Initialize [Algorithm 1, Line 2]
        x = torch.randn(n_samples, horizon, 2).to(self.device)
        
        t = 0.0
        dt = 0.001 # Initial integration step [Algorithm 1, Line 3]
        
        # Flatten batch for efficiency
        N = n_samples * horizon
        x_flat = x.view(N, 2)

        steps_count = 0
        
        pbar = tqdm(total=1000, desc="RK45 Sampling") # approximate progress
        last_t_disp = 0
        
        while t < 1.0:
            if t + dt > 1.0:
                dt = 1.0 - t
            
            t_tensor = torch.full((n_samples,), t, device=self.device)
            v_pred_curr = self.model(x_flat.view(n_samples, horizon, 2), t_tensor).view(N, 2)
            
            u_correction = torch.zeros_like(v_pred_curr)
            
            if use_cbf:
                h_vals, grads = self.get_h_and_grad_tensor(x_flat)
                phi_vals = self.phi_function_tensor(t, h_vals)
                
                # Batch QP
                if use_closed_form:
                    u_correction = self.solve_batch_closed_form(v_pred_curr, h_vals, grads, phi_vals)
                else:
                    u_correction = self.solve_qp_batch(v_pred_curr, h_vals, grads, phi_vals)

            def dynamics_func(t_scalar, x_curr_flat):
                t_vec = torch.full((n_samples,), t_scalar, device=self.device)
                # Reshape for model input
                v_nn = self.model(x_curr_flat.view(n_samples, horizon, 2), t_vec).view(N, 2)
                return v_nn + u_correction # Guidance

            x_new_flat, error_norms = self._runge_kutta_step(dynamics_func, t, x_flat, dt)

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
                
                if dt < 1e-7:
                    print(f"Warning: Step size too small at t={t:.4f}, forcing step.")
                    x_flat = x_new_flat
                    t += 1e-7
        
        pbar.close()
        
        # Reshape back
        x_final = x_flat.view(n_samples, horizon, 2)

        if use_cbf:
            x_final = self.terminal_safety_filter(x_final)
            
        return x_final

    def terminal_safety_filter(self, x):
        """
        min ||x - x_end||^2 s.t. h(x) >= 0
        """
        x_flat = x.view(-1, 2)
        
        for _ in range(5):
            h_vals, grads = self.get_h_and_grad_tensor(x_flat)
            
            unsafe_mask = h_vals < 0 # [N, n_obs]
            
            if not unsafe_mask.any():
                break
                
            # dx = - h * grad / ||grad||^2
            grad_norm_sq = torch.sum(grads**2, dim=-1) + 1e-8
            
            # [N, n_obs, 2]
            # dx = - h * grad / norm
            correction = - (h_vals.unsqueeze(-1) * grads) / grad_norm_sq.unsqueeze(-1)
            
            correction = correction * unsafe_mask.unsqueeze(-1).float()
            total_correction = torch.sum(correction, dim=1)
            
            x_flat = x_flat + total_correction
            
        return x_flat.view(x.shape)

    def solve_batch_closed_form(self, v_nom, h_vals, grads, phi_vals):
        """
        v_nom: [N, 2]
        h_vals: [N, n_obs]
        grads: [N, n_obs, 2]
        phi_vals: [N, n_obs]
        """
        # a [N, n_obs]
        # a = grad^T * v + phi * h
        # grads: [N, n_obs, 2], v_nom: [N, 1, 2]
        lie_deriv = torch.sum(grads * v_nom.unsqueeze(1), dim=-1) # [N, n_obs]
        a = lie_deriv + phi_vals * h_vals
        
        # ||b||^2 = ||grad||^2 [N, n_obs]
        b_norm_sq = torch.sum(grads ** 2, dim=-1)
        b_norm_sq = torch.clamp(b_norm_sq, min=1e-6)
        
        # u = max(0, -a/||b||^2) * b
        # if a >= 0 (safe), u = 0
        # if a < 0 (unsafe), u = -a * b / ||b||^2
        
        lambda_val = -a / b_norm_sq     # [N, n_obs]
        mask = lambda_val > 0           
        
        # [N, n_obs] * [N, n_obs, 1] -> [N, n_obs, 2]
        u_corrections = mask.unsqueeze(-1).float() * lambda_val.unsqueeze(-1) * grads

        # u_final = torch.sum(u_corrections, dim=1) # [N, 2]
        
        norms = torch.norm(u_corrections, dim=-1)
        max_idx = torch.argmax(norms, dim=1)
        u_final = u_corrections[torch.arange(u_corrections.size(0)), max_idx]


        u_norm = torch.norm(u_final, dim=-1, keepdim=True)
        max_u = 2 
        clip_mask = u_norm > max_u
        u_final = torch.where(clip_mask, u_final / u_norm * max_u, u_final)
        
        return u_final
