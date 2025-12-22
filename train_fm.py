from typing import List
from collections import namedtuple
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
# import cvxopt
# from cvxopt import matrix, solvers
from tqdm import tqdm
import math
import gymnasium as gym
import gymnasium_robotics

from model.dit import TrajectoryDiT  
from dataset import BoxConsTrajDataset
from utils import set_all_seed
from visual import plot_trajectory_comparison, plot_simple_loss

# # 关闭 cvxopt 的输出
# solvers.options['show_progress'] = False

# 尝试导入 cvxpylayers，用于 GPU Batch QP
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    HAS_CVXPYLAYERS = True
except ImportError:
    HAS_CVXPYLAYERS = False
    print("Warning: cvxpylayers not found. GPU batch solving will fail.")



def train_flow_matching(
    dataset,
    save_model_path='dit_flow.pth',
    batch_size=64,
    n_epochs=100,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # 1. 加载数据
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_dim = dataset.x_dim
    seq_length = dataset.seq_length

    # 2. 初始化模型
    # 获取维度信息
    model = TrajectoryDiT(x_dim=x_dim, max_horizon=seq_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Start training on {device}...")
    
    model.train()

    tq_iter = trange(n_epochs)
    loss_history = []
    for epoch in tq_iter:
        total_loss = 0
        for batch in dataloader:

            traj_flat = batch['traj'].float().to(device)
            b_size = traj_flat.shape[0]
            traj = traj_flat.reshape(b_size, seq_length, x_dim)

            x1 = traj

            # 3. Flow Matching 核心逻辑 
            # 采样时间 t ~ Uniform[0, 1]
            t = torch.rand(b_size, device=device)
            
            # 采样噪声 x0 ~ N(0, I)
            x0 = torch.randn_like(x1)
            
            # 计算插值 xt (Optimal Transport path / Conditional Flow)
            # path: p_t(x) = (1 - t) * p_0(x) + t * p_1(x)
            # t 需要广播到 [B, 1, 1]
            t_expand = t.view(b_size, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * x1
            
            # 计算目标向量场 u_t(x|x_1) = x_1 - x_0
            # 这是 conditional vector field，使得 ODE 轨迹为直线
            target_v = x1 - x0
            
            # 4. 模型预测向量场 v_theta
            pred_v = model(xt, t)
            
            # 5. 计算 MSE Loss: || v_theta(xt, t) - (x1 - x0) ||^2
            loss = torch.mean((pred_v - target_v) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        tq_iter.set_description(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)

    # 保存模型
    torch.save(model.state_dict(), save_model_path)
    print("Training finished and model saved.")

    return model, loss_history


Rectangle = namedtuple('Rectangle', ['r_min', 'c_min', 'r_max', 'c_max'])

class MazeObs:
    def __init__(self, 
                 maze, 
                 rect_list: List[Rectangle],
                 obs_expand_dis = 0.2,
                 ellips_n = 4,
                 alpha: float = 0.5):
        """
        Docstring for __init__
        
        :param self: Description
        :param maze: 迷宫环境
        :param rect_list: 
        :param obs_expand_dis 障碍物膨胀距离
        :param ellips_n 超椭圆的阶数
        :param alpha: 缩放系数 [0, 1]
        """
        assert alpha >= 0 and alpha <= 1

        self.alpha = alpha
        self.rect_list = rect_list
        self.maze = maze
        self.obs_expand_dis = obs_expand_dis
        self.ellips_n = ellips_n

        # CBF form: (x-x_c)^2/a^2 + (y-y_c)^2/b^2 -1 >= 0
        self.ellips_list = self.create_ellips_list()

    def create_ellips_list(self):

        ellips_list = []
        for rect in self.rect_list:
            # 左上角格子中心
            p_min = self.maze.cell_rowcol_to_xy(np.array([rect.r_min, rect.c_min]))
            # 右下角格子中心
            p_max = self.maze.cell_rowcol_to_xy(np.array([rect.r_max, rect.c_max]))

            half_scale = self.maze.maze_size_scaling * 0.5
            x_min, x_max = p_min[0] - half_scale - self.obs_expand_dis, p_max[0] + half_scale + self.obs_expand_dis
            y_min, y_max = p_max[1] - half_scale - self.obs_expand_dis, p_min[1] + half_scale + self.obs_expand_dis

            x_center = (x_min + x_max) * 0.5
            y_center = (y_min + y_max) * 0.5

            x_length = x_max - x_min
            y_length = y_max - y_min

            # 最大内切椭圆
            a_in = x_length * 0.5
            b_in = y_length * 0.5
            # 最小外接椭圆
            a_out = x_length * 0.5 * math.pow(2, 1.0 / self.ellips_n)
            b_out = y_length * 0.5 * math.pow(2, 1.0 / self.ellips_n)
            # 使用参数alpha控制近似程度
            a = a_in + (a_out - a_in) * self.alpha
            b = b_in + (b_out - b_in) * self.alpha

            ellips_list.append([x_center, y_center, a, b, self.ellips_n])

        return ellips_list

    def get_ellips_list(self):

        return self.ellips_list


# class SafeFlowSampler:
#     def __init__(self, model, obstacles, device='cuda'):
#         """
#         :param model: 训练好的 TrajectoryDiT
#         :param obstacles: 列表，每个元素为 [x_c, y_c, a, b] (对应椭圆参数)
#         """
#         self.model = model
#         self.obstacles = obstacles # List of [x_c, y_c, a, b]
#         self.device = device
#         self.model.eval()

#     def get_h_and_grad(self, x_flat):
#         """
#         计算 Barrier Function h(x) 及其梯度
#         x_flat: numpy array [D] (单个时间步的单个点的状态)
        
#         对于椭圆障碍物: h(x) = (x-xc)^2/a^2 + (y-yc)^2/b^2 - 1 >= 0 (安全区域在椭圆外)
#         注意：论文中 h(s) >= 0 表示安全 [cite: 114]。
#         """
#         h_vals = []
#         grads = []
        
#         px, py = x_flat[0], x_flat[1]
        
#         for obs in self.obstacles:
#             xc, yc, a, b = obs
            
#             # 计算 h(x)
#             term_x = (px - xc)**2 / (a**2)
#             term_y = (py - yc)**2 / (b**2)
#             # 论文 maze_test.py 中定义为 (dist - 1) >= 0，即椭圆外为安全
#             h = term_x + term_y - 1.0 # 
            
#             # 计算 grad h(x)
#             grad_x = 2 * (px - xc) / (a**2)
#             grad_y = 2 * (py - yc) / (b**2)
            
#             h_vals.append(h)
#             grads.append([grad_x, grad_y])
            
#         return np.array(h_vals), np.array(grads) # [N_obs], [N_obs, 2]

#     def phi_function(self, t, h_val, gamma=0.9):
#         """
#         Blow-up function phi(t, h) [cite: 156, 162]
#         如果 h >= 0: phi = phi0
#         如果 h < 0: phi = phi1(t) (随着 t -> 1 爆炸)
#         """
#         # 这里的 T=1.0
#         T_end = 1.0
#         margin = 1e-3
#         effective_t = min(t, T_end - margin)
        
#         if h_val >= 0:
#             return 1.0
#         else:
#             # 论文中 Eq(42)
#             if t < gamma:
#                 return 1 + 4 * math.pow(t, 3)
#             else:
#                 return 1.0 / (T_end - effective_t)

#     def solve_qp_slack(self, v_nom, h_vals, grads, t):
#         """
#         求解带松弛变量的 QP 问题
#         Minimize ||u||^2 + sum(delta_j^2) * HUGE_VAL
#         s.t. 
#           1) grad_h^T * (v_nom + u) + delta >= - phi(h) * h  (CBF 约束)
#           2) delta >= 0                                      (非负约束)

#         调整为标准形式:
#         Variables z = [u_x, u_y, delta_1, ..., delta_N]
#         Minimize 1/2 z^T P z + q^T z
#         """
#         x_dim = 2
#         n_obs = len(h_vals)
#         n_vars = x_dim + n_obs
        
#         # 1. 构造 P 矩阵 (对角矩阵)
#         P_diag = [2.0] * x_dim + [1e6] * n_obs 
#         P = matrix(np.diag(P_diag))
        
#         # 2. 构造 q 向量 (全 0)
#         q = matrix(np.zeros(n_vars))
        
#         # 3. 构造约束 Gz <= h_qp
#         # 我们有两组约束:
#         # A) CBF约束: -grad^T * u - delta <= grad^T * v + phi * h
#         # B) 非负约束: -delta <= 0  (即 delta >= 0)
        
#         # 总约束数量 = n_obs (CBF) + n_obs (Non-negative)
#         n_constraints = n_obs + n_obs
        
#         G_np = np.zeros((n_constraints, n_vars))
#         h_qp_np = np.zeros(n_constraints)
        
#         for i in range(n_obs):
#             # --- Constraint A: CBF ---
#             # -grad^T * u
#             G_np[i, 0] = -grads[i, 0]
#             G_np[i, 1] = -grads[i, 1]
#             # -delta_i (coefficient is -1 because standard form is <=)
#             G_np[i, x_dim + i] = -1.0 
            
#             # RHS for CBF
#             phi = self.phi_function(t, h_vals[i])
#             Lie_f = np.dot(grads[i], v_nom)
#             h_qp_np[i] = Lie_f + phi * h_vals[i]
            
#             # --- Constraint B: Delta >= 0 ---
#             # 对应不等式: -delta_i <= 0
#             # 在 G 矩阵中，第 (n_obs + i) 行，对应 delta_i 的列设为 -1
#             row_idx = n_obs + i
#             G_np[row_idx, x_dim + i] = -1.0
#             h_qp_np[row_idx] = 0.0
            
#         G = matrix(G_np)
#         h_qp = matrix(h_qp_np)
        
#         try:
#             sol = solvers.qp(P, q, G, h_qp)
#             z = np.array(sol['x']).flatten()
#             u_safe = z[:x_dim]
#             return u_safe
#         except ValueError:
#             print("Warning: cvxqp solve failed!!")
#             return np.zeros(x_dim)

#     @torch.no_grad()
#     def sample(self, n_samples, horizon, steps=100, use_cbf=True):
#         """
#         执行欧拉积分进行采样，并在每一步应用 QP 修正
#         """
#         # 1. 初始采样 x0 ~ N(0, I)
#         x = torch.randn(n_samples, horizon, 2).to(self.device)
#         dt = 1.0 / steps
        
#         # 时间循环 [0 -> 1]
#         for i in tqdm(range(steps), desc="Sampling"):
#             t_curr = i * dt
#             t_tensor = torch.full((n_samples,), t_curr, device=self.device)
            
#             # 2. 预测名义向量场 v_theta [cite: 92]
#             v_pred = self.model(x, t_tensor) # [B, H, 2]
            
#             # 3. 安全修正 (如果启用 CBF)
#             # 这一步通常需要在 CPU 上串行处理每个样本（或者使用专门的 Batch QP 求解器）
#             # 为了演示清晰，这里使用 Python 循环 + cvxopt
#             correction = torch.zeros_like(v_pred)
            
#             if use_cbf:
#                 x_np = x.cpu().numpy()
#                 v_np = v_pred.cpu().numpy()
                
#                 # 遍历 Batch 和 Horizon (这会比较慢，实际部署需要优化或并行)
#                 # 论文提到对于 composite constraints 可以在 horizon 维度解耦 [cite: 172]
#                 for b in range(n_samples):
#                     for h_idx in range(horizon):
#                         state = x_np[b, h_idx]
#                         v_nom = v_np[b, h_idx]
                        
#                         # 计算障碍物信息
#                         h_vals, grads = self.get_h_and_grad(state)
                        
#                         # 如果所有 h >= 0 且 v 使得 h 增加，其实不需要 QP
#                         # 但为了严谨，且处理 potential unsafe flow，我们求解 QP
#                         # 只有当某个 h 接近 0 或小于 0 时，QP 才会生效产生非零 u
                        
#                         # 简单的预检查：如果离所有障碍物都很远，跳过 QP
#                         if np.all(h_vals > 0.5):
#                             continue
                            
#                         u = self.solve_qp_slack(v_nom, h_vals, grads, t_curr)
#                         correction[b, h_idx] = torch.from_numpy(u)
            
#             # 4. 欧拉积分更新: x_{t+1} = x_t + (v + u) * dt [cite: 134]
#             # 论文使用 guidance term u_t: dx/dt = v + u
#             final_vel = v_pred + correction.to(self.device)
#             x = x + final_vel * dt
            
#         return x
    



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
        
        for i in tqdm(range(steps), desc="Sampling"):
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

def train():

    epoch = 1000  # 如果batch=100,那么就相当于iteration=10000
    device = "cuda:0"
    seed = 42

    set_all_seed(seed)

    dataset = BoxConsTrajDataset(
        file_path='large_maze_traj_data.npz',
        seq_length=None
    )
    seq_length = dataset.seq_length

    # model, loss_history = train_flow_matching(
    #     dataset,
    #     save_model_path="large_maze_fm_model.pt",
    #     batch_size=100,
    #     n_epochs=epoch,
    #     lr=1e-4,
    #     device=device
    # )
    # plot_simple_loss(loss_history, f"large_maze_train_loss_fm.png")

    model = TrajectoryDiT(
        x_dim=dataset.x_dim, max_horizon=seq_length
    ).to(device)
    load_data = torch.load("large_maze_fm_model.pt", map_location=device)
    model.load_state_dict(load_data)


    LARGE_MAZE =   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 'g', 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 'r', 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    env = gym.make('PointMaze_Large-v3', maze_map=LARGE_MAZE, continuing_task=False, reset_target=False, max_episode_steps=1000,
                render_mode='rgb_array')
    maze = env.unwrapped.maze

    # (r_min, c_mn, r_max, c_max)
    rect_list = [
        Rectangle(2, 2, 2, 3),
        Rectangle(1, 5, 2, 5),
        Rectangle(4, 4, 4, 5),
        Rectangle(5, 5, 6, 5),
        Rectangle(3, 7, 4, 7),
        Rectangle(4, 8, 4, 9),
        Rectangle(6, 7, 7, 7),
        Rectangle(6, 9, 6, 10),
    ]
    obs_expand_dis = 0.2

    maze_obs = MazeObs(
        maze=maze, rect_list=rect_list, alpha=0.5, obs_expand_dis=obs_expand_dis, ellips_n=4,
    )

    safe_sampler = SafeFlowSampler(
        model=model, obstacles=maze_obs.get_ellips_list(), device=device
    )

    true_traj = dataset.sample_traj_data(n_sample=2)
    true_traj = true_traj.reshape(2, seq_length, 2)

    # 生成 5 条轨迹 (n_samples, seq_length, x_dim)
    # gene_traj = safe_sampler.sample(n_samples=5, horizon=seq_length, steps=1500, use_cbf=True, use_closed_form=True)
    gene_traj = safe_sampler.sample_rk45(n_samples=5, horizon=seq_length, atol=0.001, rtol=0.001, use_cbf=True, use_closed_form=True)
    gene_traj = gene_traj.cpu().numpy()


    plot_trajectory_comparison(
        env_maze=maze,
        true_trajs=true_traj,
        gene_trajs=gene_traj,
        obs_expand_dis=obs_expand_dis,
        ellips_list=maze_obs.get_ellips_list(),
        max_plot=100,
        save_path=f"large_maze_result_safeflow_test.png"
    )

if __name__=="__main__":
    train()