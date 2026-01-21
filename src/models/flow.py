from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
import ot

import src.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

from .diffusion import Sample


def compute_ot_permutation_indices(x0, x1):
    """
    Compute the optimal permutation indices to align x1 with x0.
    
    :param x0: Source batch (Batch, ...) e.g., Gaussian Noise
    :param x1: Target batch (Batch, ...) e.g., Data Trajectory
    :return: perm_indices (Batch,) - Indices to reorder x1
    """
    with torch.no_grad():
        B = x0.shape[0]
        x0_flat = x0.reshape(B, -1)
        x1_flat = x1.reshape(B, -1)

        # 1. Compute Cost Matrix (Squared Euclidean Distance)
        M = torch.cdist(x0_flat, x1_flat, p=2) ** 2
        M_np = M.cpu().numpy()

        # 2. Solve OT
        a = ot.unif(B)
        b = ot.unif(B)
        gamma = ot.emd(a, b, M_np) # Transport plan

        # 3. Get Permutation Indices
        # gamma[i, j] = 1/B means x0[i] matches x1[j]
        # We want to find j for each i.
        gamma_torch = torch.from_numpy(gamma).to(x0.device)
        perm_indices = torch.argmax(gamma_torch, dim=1)
        
        return perm_indices




def solve_cbf_halfcheetah(model, x, cond, t, dt,
        u1_scale, u1_limit,
        u4_scale, u4_limit,
        u3_scale, u3_limit,
        u3_scale2, u3_limit2):
    """
    使用 qpth 求解器精确计算 HalfCheetah 环境下的多约束 CBF 修正。
    采用降维 QP 策略 (只优化涉及的 4 个维度)。

    Constraints:
    1. u0 + u1_scale * u1 <= u1_limit
    2. u3 + u4_scale * u4 <= u4_limit
    3. u0 + u3_scale * u3 <= u3_limit
    4. - u0 + u3_scale2 * u3 <= u3_limit2

    u0 u1 u3 u4 对应的是 x 的 0, 1, 3, 4索引维度

    Constraints (Barrier Functions h(x) >= 0):
    1. u1_limit - (x[0] + u1_scale * x[1]) >= 0
    2. u4_limit - (x[3] + u4_scale * x[4]) >= 0
    3. u3_limit - (x[0] + u3_scale * x[3]) >= 0
    4. u3_limit2 - ( - x[0] + u3_scale2 * x[3]) >= 0

    涉及维度索引: 0, 1, 3, 4 (Reduced Dimension = 4)

    Args:
        model: Flow matching 模型 wrapper
        x: (batch, horizon, full_dim) 状态张量
        cond: 条件输入
        t: (batch,) 时间张量
        dt: 步长
        [Constraints params]: 约束参数 (scale 和 limit)

    Returns:
        final_vel: (batch, horizon, full_dim) 修正后的速度场
    """
    
    # ==========================================================
    # 1. 基础准备: 预测流场与数据扁平化
    # ==========================================================
    v_pred = model.model(x, cond, t)
    
    n_samples, horizon, full_dim = x.shape
    N = n_samples * horizon
    device = x.device
    
    # Flatten batch and horizon dimensions
    x_flat = x.view(N, -1)
    v_flat = v_pred.view(N, -1)
    
    # 处理时间 t (Broadcast to N)
    if t.dim() == 1:
        t_flat = t.unsqueeze(1).repeat(1, horizon).view(-1)
    else:
        t_flat = t.view(-1)

    # ==========================================================
    # 2. 提取子空间变量 (Subspace Extraction)
    # 我们将问题投影到 4D 空间: [idx0, idx1, idx3, idx4]
    # ==========================================================
    
    # 定义局部索引映射 (Local Mapping)
    # u_reduced vector order: [val_0, val_1, val_3, val_4]
    # index 0 -> global 0
    # index 1 -> global 1
    # index 2 -> global 3
    # index 3 -> global 4
    
    # 提取当前状态值 x
    x0 = x_flat[:, 0]
    x1 = x_flat[:, 1]
    x3 = x_flat[:, 3]
    x4 = x_flat[:, 4]
    
    # 提取预测速度值 v (Lie Derivative 部分计算需要)
    v0 = v_flat[:, 0]
    v1 = v_flat[:, 1]
    v3 = v_flat[:, 3]
    v4 = v_flat[:, 4]

    # ==========================================================
    # 3. CBF 参数设置 (Class K Function)
    # ==========================================================
    omega = 2.1 
    phi_0 = 10.0 
    t_safe = torch.clamp(t_flat, max=0.99)
    phi_1 = omega / ((1 - t_safe) ** 2)
    
    def get_gamma_h(h_val):
        """根据 Barrier 值正负选择增益系数"""
        return torch.where(h_val >= 0, phi_0 * h_val, phi_1 * h_val)

    # ==========================================================
    # 4. 构建降维 QP 矩阵 G_reduced
    # Form: G * u_corr <= h_vec
    # Shape: (N, num_constraints, reduced_dim) => (N, 4, 4)
    # G 的每一行对应约束的负梯度 (-Gradient)
    # 局部变量顺序: [u_c0, u_c1, u_c3, u_c4]
    # ==========================================================
    
    num_constraints = 4
    reduced_dim = 4
    
    G_reduced = torch.zeros(N, num_constraints, reduced_dim, device=device)
    
    # --- Constraint 1: x0 + s1*x1 <= L1 ---
    # h = L1 - x0 - s1*x1
    # Gradient w.r.t [x0, x1, x3, x4] is [-1, -s1, 0, 0]
    # G row = -Gradient = [1, s1, 0, 0]
    G_reduced[:, 0, 0] = 1.0       # coeff for u_c0
    G_reduced[:, 0, 1] = u1_scale  # coeff for u_c1
    
    # --- Constraint 2: x3 + s4*x4 <= L4 ---
    # h = L4 - x3 - s4*x4
    # Gradient w.r.t [x0, x1, x3, x4] is [0, 0, -1, -s4]
    # G row = [0, 0, 1, s4]
    G_reduced[:, 1, 2] = 1.0       # coeff for u_c3
    G_reduced[:, 1, 3] = u4_scale  # coeff for u_c4
    
    # --- Constraint 3: x0 + s3*x3 <= L3 ---
    # h = L3 - x0 - s3*x3
    # Gradient w.r.t [x0, x1, x3, x4] is [-1, 0, -s3, 0]
    # G row = [1, 0, s3, 0]
    G_reduced[:, 2, 0] = 1.0       # coeff for u_c0
    G_reduced[:, 2, 2] = u3_scale  # coeff for u_c3
    
    # --- Constraint 4: -x0 + s3_2*x3 <= L3_2 ---
    # h = L3_2 + x0 - s3_2*x3
    # Gradient is [1, 0, -s3_2, 0]
    # G row = [-1, 0, s3_2, 0]
    G_reduced[:, 3, 0] = -1.0        # coeff for u_c0
    G_reduced[:, 3, 2] = u3_scale2  # coeff for u_c3

    # ==========================================================
    # 5. 构建 QP 向量 h_vec (Inequality RHS)
    # h_vec = Gradient^T * v_pred + gamma(h_val)
    # ==========================================================
    
    h_vec = torch.zeros(N, num_constraints, device=device)
    
    # --- Constraint 1 Calculation ---
    h1_val = u1_limit - (x0 + u1_scale * x1)
    # Lie Derivative Lf_h = (-1)*v0 + (-s1)*v1
    lf_h1 = -1.0 * v0 - u1_scale * v1
    h_vec[:, 0] = lf_h1 + get_gamma_h(h1_val)
    
    # --- Constraint 2 Calculation ---
    h2_val = u4_limit - (x3 + u4_scale * x4)
    # Lie Derivative Lf_h = (-1)*v3 + (-s4)*v4
    lf_h2 = -1.0 * v3 - u4_scale * v4
    h_vec[:, 1] = lf_h2 + get_gamma_h(h2_val)
    
    # --- Constraint 3 Calculation ---
    h3_val = u3_limit - (x0 + u3_scale * x3)
    # Lie Derivative Lf_h = (-1)*v0 + (-s3)*v3
    lf_h3 = -1.0 * v0 - u3_scale * v3
    h_vec[:, 2] = lf_h3 + get_gamma_h(h3_val)
    
    # --- Constraint 4 Calculation ---
    h4_val = u3_limit2 - (-x0 + u3_scale2 * x3)
    # Lie Derivative Lf_h = 1.0*v0 + (-s3_2)*v3
    lf_h4 = 1.0 * v0 - u3_scale2 * v3
    h_vec[:, 3] = lf_h4 + get_gamma_h(h4_val)

    # ==========================================================
    # 6. 求解 Reduced QP
    # Objective: min 0.5 * ||u_reduced||^2
    # ==========================================================
    
    # Q: (4, 4) 单位矩阵
    Q = torch.eye(reduced_dim, device=device)
    # p: (N, 4) 零向量
    p = torch.zeros(N, reduced_dim, device=device)
    # No equality constraints
    e = torch.Tensor().to(device)
    
    try:
        # u_reduced shape: (N, 4)
        # 对应 [u_c0, u_c1, u_c3, u_c4]
        u_reduced = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(
            Q, p, G_reduced, h_vec, e, e
        )
    except Exception as err:
        print(f"HalfCheetah QP Solver failed: {err}")
        # Fallback to zero correction
        u_reduced = torch.zeros(N, reduced_dim, device=device)

    # ==========================================================
    # 7. 映射回全维度空间 (Scatter back)
    # ==========================================================
    
    u_full = torch.zeros_like(v_flat)
    
    # 将计算出的 4 个维度的修正量填回对应的全局索引
    u_full[:, 0] = u_reduced[:, 0] # global 0
    u_full[:, 1] = u_reduced[:, 1] # global 1
    u_full[:, 3] = u_reduced[:, 2] # global 3 (注意: u_reduced index 2)
    u_full[:, 4] = u_reduced[:, 3] # global 4 (注意: u_reduced index 3)

    # 8. 叠加修正并还原形状
    final_vel_flat = v_flat + u_full
    final_vel = final_vel_flat.view(n_samples, horizon, full_dim)
    
    return final_vel


def solve_cbf_more_complex(model, x, cond, t, dt, 
                   h_max=1.5, vel_scale=0.1, h_min=0.8, v_max=2.5, v_min=-2.5, 
                   z_idx=3, vz_idx=9):
    """
    使用 qpth 求解器精确计算多约束 CBF 修正。
    
    Constraints:
    1. Momentum Ceiling: h1(x) = h_max - z - vel_scale * vz >= 0
    2. Height Floor:     h2(x) = z - h_min >= 0
    3. Velocity Ceiling: h3(x) = v_max - vz >= 0
    4. Velocity Floor:   h4(x) = vz - v_min >= 0

    Args:
        model: Flow matching model.
        x: (B, T, D) State tensor.
        cond: Conditioning.
        t: (B,) Time tensor.
        dt: Unused here.
        [Constraints params]...

    Returns:
        final_vel: Corrected velocity field.
    """
    # 1. 获取预测流场
    v_pred = model.model(x, cond, t)
    
    # 2. 扁平化处理
    n_samples, horizon, full_dim = x.shape
    N = n_samples * horizon
    device = x.device
    
    x_flat = x.view(N, -1)
    v_flat = v_pred.view(N, -1)
    
    if t.dim() == 1:
        t_flat = t.unsqueeze(1).repeat(1, horizon).view(-1)
    else:
        t_flat = t.view(-1)

    # 3. 提取 2D 子空间的关键变量
    # 我们只关心 z 和 vz
    z_curr = x_flat[:, z_idx]
    vz_curr = x_flat[:, vz_idx]
    
    dot_z = v_flat[:, z_idx]   # dz/dt
    dot_vz = v_flat[:, vz_idx] # dvz/dt

    # 4. 准备 CBF 参数 (Gamma Function)
    omega = 2.1 
    phi_0 = 10.0 
    t_safe = torch.clamp(t_flat, max=0.99)
    phi_1 = omega / ((1 - t_safe) ** 2)
    
    def get_gamma_h(h_val):
        return torch.where(h_val >= 0, phi_0 * h_val, phi_1 * h_val)

    # ==========================================================
    # 5. 构建降维后的 QP 矩阵 G_reduced
    # 优化变量 u_reduced 是 2维向量: [u_z, u_vz]
    # G 的形状从 (N, 4, full_dim) 变为 (N, 4, 2)
    # ==========================================================
    
    num_constraints = 4
    reduced_dim = 2 # 仅优化 u_z (idx 0) 和 u_vz (idx 1)
    
    G_reduced = torch.zeros(N, num_constraints, reduced_dim, device=device)
    
    # --- Constraint 1: Momentum Ceiling (h_max - z - alpha*vz >= 0) ---
    # Gradient w.r.t [z, vz] is [-1, -alpha]
    # G row = -Gradient = [1, alpha]
    G_reduced[:, 0, 0] = 1.0        # u_z 的系数
    G_reduced[:, 0, 1] = vel_scale  # u_vz 的系数
    
    # --- Constraint 2: Height Floor (z - h_min >= 0) ---
    # Gradient w.r.t [z, vz] is [1, 0]
    # G row = [-1, 0]
    G_reduced[:, 1, 0] = -1.0
    
    # --- Constraint 3: Velocity Ceiling (v_max - vz >= 0) ---
    # Gradient w.r.t [z, vz] is [0, -1]
    # G row = [0, 1]
    G_reduced[:, 2, 1] = 1.0
    
    # --- Constraint 4: Velocity Floor (vz - v_min >= 0) ---
    # Gradient w.r.t [z, vz] is [0, 1]
    # G row = [0, -1]
    G_reduced[:, 3, 1] = -1.0

    # ==========================================================
    # 6. 构建 QP 向量 h_vec (不等式右侧)
    # h_vec 的计算与维度无关，只取决于约束值的余量
    # Shape: (N, 4)
    # ==========================================================
    
    h_vec = torch.zeros(N, num_constraints, device=device)
    
    # C1
    h1_val = h_max - z_curr - vel_scale * vz_curr
    lf_h1 = -1.0 * dot_z - vel_scale * dot_vz
    h_vec[:, 0] = lf_h1 + get_gamma_h(h1_val)
    
    # C2
    h2_val = z_curr - h_min
    lf_h2 = 1.0 * dot_z 
    h_vec[:, 1] = lf_h2 + get_gamma_h(h2_val)
    
    # C3
    h3_val = v_max - vz_curr
    lf_h3 = -1.0 * dot_vz
    h_vec[:, 2] = lf_h3 + get_gamma_h(h3_val)
    
    # C4
    h4_val = vz_curr - v_min
    lf_h4 = 1.0 * dot_vz
    h_vec[:, 3] = lf_h4 + get_gamma_h(h4_val)

    # ==========================================================
    # 7. 求解 2D QP
    # min 0.5 * u_red^T Q u_red
    # ==========================================================
    
    # Q 是 2x2 单位矩阵
    Q = torch.eye(reduced_dim, device=device)
    
    # p 是 2维零向量 (N, 2)
    p = torch.zeros(N, reduced_dim, device=device)
    
    e = torch.Tensor().to(device) # 无等式约束
    
    try:
        # u_reduced shape: (N, 2)
        # u_reduced[:, 0] 是 u_z， u_reduced[:, 1] 是 u_vz
        u_reduced = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(
            Q, p, G_reduced, h_vec, e, e
        )
    except Exception as err:
        print(f"Reduced QP Solver failed: {err}")
        u_reduced = torch.zeros(N, reduced_dim, device=device)

    # ==========================================================
    # 8. 映射回全维度空间 (Scatter/Map back)
    # ==========================================================
    
    # 创建全维度的修正向量 (初始全为0)
    u_full = torch.zeros_like(v_flat)
    
    # 将 u_reduced 的结果填入对应的 z_idx 和 vz_idx
    u_full[:, z_idx] = u_reduced[:, 0]
    u_full[:, vz_idx] = u_reduced[:, 1]

    # 9. 叠加并还原
    final_vel_flat = v_flat + u_full
    final_vel = final_vel_flat.view(n_samples, horizon, full_dim)
    
    return final_vel


def solve_cbf_complex(model, x, cond, t, dt, height_limit=1.5, vel_scale=0.1, z_idx=3, vz_idx=9):
    """
    计算基于动量约束 (Position + Velocity constraint) 的 CBF 修正流场。
    约束公式: z + vel_scale * vz <= height_limit
    即 h(x) = height_limit - z - vel_scale * vz >= 0

    Args:
        model: Flow matching 模型 wrapper。
        x: (batch, horizon, full_dim) 当前状态张量。
        cond: 条件输入。
        t: (batch,) 时间张量。
        dt: 步长 (用于 API 兼容)。
        height_limit: 安全高度限制。
        vel_scale: 速度对约束的权重系数 (alpha)。
        z_idx: 状态 x 中 z 坐标的索引 (默认为 3，与你之前的代码一致)。
        vz_idx: 状态 x 中 z 速度 (vz) 的索引。
                注意：请根据你的 Hopper 数据集格式确认此索引！
                通常如果是 [qpos, qvel]，vz 是 z_idx + num_qpos。

    Returns:
        final_vel: (batch, horizon, full_dim) 修正后的速度场。
    """
    # 1. 从 Flow Matching 模型获取预测流场 (Predicted Vector Field)
    # v_pred 代表状态 x 的时间导数 dx/dt
    v_pred = model.model(x, cond, t)
    
    # 2. 数据扁平化处理 (Batch Processing)
    n_samples, horizon, full_dim = x.shape
    N = n_samples * horizon
    
    x_flat = x.view(N, -1)
    v_flat = v_pred.view(N, -1)
    
    # 处理时间 t 的维度，使其与 flattened x 对齐
    if t.dim() == 1:
        t_flat = t.unsqueeze(1).repeat(1, horizon).view(-1)
    else:
        t_flat = t.view(-1)

    # 3. 定义 FMBF (Flow Matching Barrier Function) 参数
    # 参考 SafeFlow 论文 Eq. 8 和 Eq. 10
    omega = 2.1 
    phi_0 = 10.0 # 安全区域内的增益系数
    
    # 防止 t=1 时除以零
    t_safe = torch.clamp(t_flat, max=0.99) 
    phi_1 = omega / ((1 - t_safe) ** 2) # 不安全趋近时的激增函数 (Blow-up)

    # 4. 计算 Barrier Function h(x) 和 梯度向量 b
    # h(x) = limit - z - scale * vz
    current_z = x_flat[:, z_idx]
    current_vz = x_flat[:, vz_idx]
    
    h = height_limit - (current_z + vel_scale * current_vz)
    
    # 计算梯度 b = \nabla h(x)
    # h 对 z 的导数是 -1
    # h 对 vz 的导数是 -vel_scale
    b_vec = torch.zeros_like(x_flat)
    b_vec[:, z_idx] = -1.0
    b_vec[:, vz_idx] = -vel_scale
    
    # 计算梯度范数 ||b||^2
    # ||b||^2 = (-1)^2 + (-vel_scale)^2 = 1 + vel_scale^2
    b_norm_sq = 1.0 + (vel_scale ** 2)

    # 5. 计算增益项 Gamma(t, h) * h
    # 如果 h >= 0 (安全)，使用线性增益 phi_0
    # 如果 h < 0 (不安全/越界)，使用反比例增益 phi_1 强制拉回
    gamma_h = torch.where(
        h >= 0,
        phi_0 * h,
        phi_1 * h
    )

    # 6. 计算投影判定项 'a' (Eq. 14)
    # CBF 约束条件: b^T * v + gamma(h) >= 0
    # 我们计算 a = b^T * v + gamma(h)，如果 a < 0 则需要修正
    
    # b^T * v 点积计算 (只在 z_idx 和 vz_idx 有值)
    # v_flat[:, z_idx] 对应 z 的变化率 (即 vz)
    # v_flat[:, vz_idx] 对应 vz 的变化率 (即 az, 加速度)
    b_dot_v = (-1.0 * v_flat[:, z_idx]) + (-vel_scale * v_flat[:, vz_idx])
    
    a = b_dot_v + gamma_h

    # 7. 计算修正量 u (Eq. 16: QP Closed-form solution)
    # Optimization target: min ||u||^2 s.t. a + b^T u >= 0
    # Solution: u = max(0, -a / ||b||^2) * b (注意这里 a<0 时才修正，所以符号取反)
    
    unsafe_mask = a < 0  # 找出违反 CBF 约束的样本
    
    u = torch.zeros_like(v_flat)
    
    # 计算修正系数 lambda = -a / ||b||^2
    correction_scalar = -a[unsafe_mask] / b_norm_sq
    
    # 应用修正: u = scalar * b
    # unsqueeze(1) 用于广播 scalar 到向量维度
    u[unsafe_mask] = correction_scalar.unsqueeze(1) * b_vec[unsafe_mask]

    # 8. 叠加修正并还原形状
    final_vel_flat = v_flat + u
    final_vel = final_vel_flat.view(n_samples, horizon, full_dim)
    
    return final_vel

def rk4_step(model, x, cond, t, dt, **kwargs):
    """
    Runge-Kutta 4 (RK4) integration step.
    Error: O(dt^5), Total Error: O(dt^4)
    """
    # k1: 初始点的斜率
    k1 = model.model(x, cond, t)
    
    # k2: 中点的斜率 (使用 k1 预测位置)
    x_2 = x + 0.5 * k1 * dt
    t_mid = t + 0.5 * dt
    k2 = model.model(x_2, cond, t_mid)
    
    # k3: 中点的斜率 (使用 k2 预测位置)
    x_3 = x + 0.5 * k2 * dt
    # t_mid is same as above
    k3 = model.model(x_3, cond, t_mid)
    
    # k4: 终点的斜率 (使用 k3 预测位置)
    x_4 = x + k3 * dt
    t_end = t + dt
    k4 = model.model(x_4, cond, t_end)
    
    # 组合加权平均斜率
    v_final = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    
    # Integration
    x_next = x + v_final * dt
    
    return x_next, 0.0

def euler_step(model, x, cond, t, dt, **kwargs):
    """
    Standard Euler integration step: x_{new} = x + v * dt
    """
    # Predict velocity field v_t
    v_pred = model.model(x, cond, t)
    
    # Integration
    x_next = x + v_pred * dt
    
    return x_next, 0.0


def guided_euler_step(model, x, cond, t, dt, guide=None, scale=1.0, n_guide_steps=1, **kwargs):
    """
    Euler step with explicit guidance optimization.
    Matches the logic of your 'n_step_guided_p_sample'.
    """
    grad_value = 0.0
    
    # 1. Guidance Optimization (Optional Inner Loop)
    # 在计算这一步的流动方向之前，先根据 guide 修正当前位置 x
    if guide is not None and n_guide_steps > 0:
        # Clone current x to avoid modifying the integration path variable directly in case of errors
        x_guided = x.clone()

        for _ in range(n_guide_steps):
            # Critical Fix: 
            # ValueGuide.gradients calls x.requires_grad_().
            # This requires x to be a leaf variable (detached from graph).
            x_guided = x_guided.detach()

            # Enable grad specifically for the guide calculation
            with torch.enable_grad():
                # returns value (y) and gradient (grad)
                y, grad = guide.gradients(x_guided, cond, t)

            # Gradient Ascent (assuming we want to maximize the value y)
            # If y is a cost to minimize, change '+' to '-'
            x_guided = x_guided + scale * grad
            
            # Re-apply conditioning (Ensure constraints are met during optimization)
            x_guided = apply_conditioning(x_guided, cond, model.action_dim)
        
        # Update x to the optimized position
        x = x_guided
        grad_value = y.mean().item()

    # 2. ODE Flow Step
    v_pred = model.model(x, cond, t)
    x_next = x + v_pred * dt
    
    return x_next, grad_value

class FlowMatching(nn.Module):

    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps,
            loss_type='l2', action_weight=1.0, loss_discount=1.0, loss_weights=None, 
            use_ot_batch=False):
        super().__init__()

        # for normalization obs
        self.means = 0.0
        self.stds = 0.0
        # for normalization act
        self.act_means = 0.0
        self.act_stds = 0.0

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.n_timesteps = n_timesteps

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        self.use_ot_batch = use_ot_batch
        print(f"[FlowMatching]: use ot batch: {self.use_ot_batch}")

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
            return:
                (horizon, transition_dim)
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights
    
    def loss(self, x, cond, *args):
        """
        compute batch flow matching loss
        
        :param self: Description
        :param x: (batch, horizon, act_dim+obs_dim)
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        return: 
            loss: 
            info: dict
        
          model API: self.model(x, cond, t)
        """
        batch_size = len(x)
        device = x.device

        # 1. Sample continuous time t uniform in [0, 1]
        # shape: (batch,)
        t = torch.rand((batch_size,), device=device)

        # 2. Sample noise x_0 from standard normal
        x_0 = torch.randn_like(x)
        x_1 = x # The target data

        # 3. Compute Interpolation (Optimal Transport Path)
        # Formula: x_t = (1 - t) * x_0 + t * x_1
        # Reshape t for broadcasting: (batch,) -> (batch, 1, 1)
        t_b = t.view(batch_size, 1, 1)
        x_t = (1 - t_b) * x_0 + t_b * x_1

        # 4. Apply conditioning to model input
        # 这一点借鉴了你的 Diffusion 实现：强制模型输入的特定维度（如起点）符合条件
        # 这有助于模型在推理时感知到正确的上下文
        x_t = apply_conditioning(x_t, cond, self.action_dim)

        # 5. Model Forward
        # Predict the vector field v_t
        # model signature: (x, cond, t)
        v_pred = self.model(x_t, cond, t)

        # 6. Compute Target Velocity
        # For OT path x_t = (1-t)x_0 + t*x_1, the derivative dx/dt is (x_1 - x_0)
        v_target = x_1 - x_0

        # Note: 
        # 在 Diffusion 实现中，你对 output (x_recon) 再次使用了 apply_conditioning。
        # 在 Flow Matching 中，模型预测的是速度 v 而不是状态 x。
        # 对速度应用状态的 conditioning 数值上通常是不对的，且 v_target 已经包含了由 x_1 确定的方向信息，
        # 所以这里通常不需要对 v_pred 进行 apply_conditioning。

        # 7. Compute Loss
        loss, info = self.loss_fn(v_pred, v_target)

        return loss, info

    def forward(self, cond, verbose, **kwargs):
        return self.conditional_sample(cond, verbose, **kwargs)

    @torch.no_grad()
    def conditional_sample(self, cond, verbose=True, return_chain=True, **sample_kwargs): 
        """
        conditional_sample
        
        :param self: Description
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        :param verbose: Description
        :param return_chain: Description
        :param sample_kwargs: Description
        """
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, verbose=verbose, return_chain=return_chain, **sample_kwargs)    # debug

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=None, guide=None, **sample_kwargs):
        """
        Docstring for p_sample_loop
        
        :param self: Description
        :param shape: Description
        :param cond: Description
        :param verbose: Description
        :param return_chain: Description
        :param sample_fn: Description
        :param guide: Description
        :param sample_kwargs: Description
        """
        device = next(iter(self.model.parameters())).device # assuming model is on correct device
        batch_size = shape[0]

        # 1. Start from Noise (t=0)
        x = torch.randn(shape, device=device)
        
        # 2. Force start condition
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        
        # Setup progress bar
        iterator = range(self.n_timesteps)
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        
        # Define time step size dt = 1 / N
        dt = 1.0 / self.n_timesteps

        # Select sampling function
        if sample_fn is None:
            sample_fn = guided_euler_step if guide is not None else euler_step

        # 3. Integration Loop: t goes from 0 to 1
        for i in iterator:
            # Current time t (scalar)
            t_value = i / self.n_timesteps
            
            # Create batch time tensor
            t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

            # 4. Step: x_{t+1} <- x_t + v(x_t)*dt
            # sample_fn returns next_x and any info (like guidance value)
            x, values = sample_fn(self, x, cond, t, dt, guide=guide, **sample_kwargs)

            # 5. Re-apply conditioning
            # 这一步非常重要，确保轨迹始终“锚定”在起始点/终点约束上
            x = apply_conditioning(x, cond, self.action_dim)

            if return_chain: chain.append(x)

        progress.stamp()

        if return_chain: chain = torch.stack(chain, dim=1)
        
        # Return Sample namedtuple to match your interface
        return Sample(x, values, chain), 0
    

class SafeFlowMathcing(FlowMatching):

    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps, 
                 loss_type='l2', action_weight=1, loss_discount=1, loss_weights=None,
                 env_name="hopper", safe_method='safeflow', 
                 height_limit=1.6, vel_scale=0.01, obs_vel_idx=6, height_min=0.8, v_max=2.5, v_min=-2.5,
                 leg_limit=1.2, torsion_limit=0.8):
        super().__init__(
            model, horizon, observation_dim, action_dim, n_timesteps, 
            loss_type, action_weight, loss_discount, loss_weights)
        
        self.env_name = env_name
        self.safe_method = safe_method
        assert self.env_name in ['hopper', 'hopper_cpx', 'hopper_cpx2', 'walker2d', 'walker2d_cpx', 'walker2d_cpx2', 'halfcheetah']
        assert self.safe_method in ['safeflow']

        self.height_limit = height_limit
        self.vel_scale = vel_scale
        self.obs_vel_idx = obs_vel_idx
        self.height_min = height_min
        self.v_max = v_max
        self.v_min = v_min
        # for halfcheetah
        self.leg_limit = leg_limit
        self.torsion_limit = torsion_limit

        print(f"Using Env: {self.env_name}  Safe Method: {self.safe_method}")

    def forward(self, cond, verbose, **kwargs):
        return self.conditional_sample(cond, verbose, **kwargs)

    @torch.no_grad()
    def conditional_sample(self, cond, verbose=True, return_chain=True, **sample_kwargs): 
        """
        conditional_sample
        
        :param self: Description
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        :param verbose: Description
        :param return_chain: Description
        :param sample_kwargs: Description
        """
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, verbose=verbose, return_chain=return_chain, **sample_kwargs)    # debug

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=None, guide=None, **sample_kwargs):
        """
        Docstring for p_sample_loop
        
        :param self: Description
        :param shape: Description
        :param cond: Description
        :param verbose: Description
        :param return_chain: Description
        :param sample_fn: Description
        :param guide: Description
        :param sample_kwargs: Description
        """
        device = next(iter(self.model.parameters())).device # assuming model is on correct device
        batch_size = shape[0]

        # 1. Start from Noise (t=0)
        x = torch.randn(shape, device=device)
        
        # 2. Force start condition
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        
        # Setup progress bar
        iterator = range(self.n_timesteps)
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        
        # Define time step size dt = 1 / N
        dt = 1.0 / self.n_timesteps

        # 3. Integration Loop: t goes from 0 to 1
        for i in iterator:
            # Current time t (scalar)
            t_value = i / self.n_timesteps
            
            # Create batch time tensor
            t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

            # 4. Step: x_{t+1} <- x_t + v(x_t)*dt
            # sample_fn returns next_x and any info (like guidance value)
            if self.env_name == 'hopper' or self.env_name == 'walker2d':
                v_final = self.model(x, cond, t)
            elif self.env_name == 'hopper_cpx2' or self.env_name == 'walker2d_cpx2':
                if self.safe_method == 'safeflow':
                    vel_scale = self.vel_scale * self.stds[self.obs_vel_idx] / self.stds[0]
                    height_max = (self.height_limit - self.means[0]) / self.stds[0] \
                        - self.vel_scale * self.means[self.obs_vel_idx] / self.stds[0]
                    height_min = (self.height_min - self.means[0]) / self.stds[0]
                    v_max = (self.v_max - self.means[self.obs_vel_idx]) / self.stds[self.obs_vel_idx]
                    v_min = (self.v_min - self.means[self.obs_vel_idx]) / self.stds[self.obs_vel_idx]
                    v_final = solve_cbf_more_complex(self, x, cond, t, dt,
                            h_max=height_max, vel_scale=vel_scale, h_min=height_min,
                            v_max=v_max, v_min=v_min, z_idx=self.action_dim+0,
                            vz_idx=self.action_dim+self.obs_vel_idx)
                else:
                    v_final = self.model(x, cond, t)
            elif self.env_name == 'hopper_cpx' or self.env_name == 'walker2d_cpx':
                if self.safe_method == 'safeflow':
                    vel_scale = self.vel_scale * self.stds[self.obs_vel_idx] / self.stds[0]
                    height_max = (self.height_limit - self.means[0]) / self.stds[0] \
                        - self.vel_scale * self.means[self.obs_vel_idx] / self.stds[0]

                    v_final = solve_cbf_complex(self, x, cond, t, dt, height_max, vel_scale, z_idx=self.action_dim+0, vz_idx=self.action_dim+self.obs_vel_idx)
                else:
                    v_final = self.model(x, cond, t)
            elif self.env_name == 'halfcheetah':
                if self.safe_method == 'safeflow':
                    back_knee_mean, back_knee_std = self.act_means[1], self.act_stds[1]
                    back_thigh_mean, back_thigh_std = self.act_means[0], self.act_stds[0]
                    front_knee_mean, front_knee_std = self.act_means[4], self.act_stds[4]
                    front_thigh_mean, front_thigh_std = self.act_means[3], self.act_stds[3]

                    # u0 + back_knee_scale * u1 <= back_leg_limit
                    back_knee_scale = back_knee_std / back_thigh_std
                    back_leg_limit = (self.leg_limit - back_thigh_mean - back_knee_mean) / back_thigh_std

                    # u3 + front_knee_scale * u4 <= front_leg_limit
                    front_knee_scale = front_knee_std / front_thigh_std
                    front_leg_limit = (self.leg_limit - front_thigh_mean - front_knee_mean) / front_thigh_std

                    # u0 + front_thigh_scale * u3 <= front_torsion_limit 
                    front_thigh_scale = - front_thigh_std / back_thigh_std
                    front_torsion_limit = (self.torsion_limit - (back_thigh_mean - front_thigh_mean)) / back_thigh_std

                    # -u0 - front_thigh_scale * u3 <= front_torsion_limit2
                    front_torsion_limit2 = (self.torsion_limit - (front_thigh_mean - back_thigh_mean)) / back_thigh_std

                    v_final = solve_cbf_halfcheetah(self, x, cond, t, dt, 
                            u1_scale=back_knee_scale, u1_limit=back_leg_limit,
                            u4_scale=front_knee_scale, u4_limit=front_leg_limit,
                            u3_scale=front_thigh_scale, u3_limit=front_torsion_limit,
                            u3_scale2=-front_thigh_scale, u3_limit2=front_torsion_limit2)
                else:
                    v_final = self.model(x, cond, t)

            # euler step
            x = x + v_final * dt

            # 5. Re-apply conditioning
            x = apply_conditioning(x, cond, self.action_dim)

            if return_chain: chain.append(x)

        progress.stamp()

        if return_chain: chain = torch.stack(chain, dim=1)
        
        # Return Sample namedtuple to match your interface
        values = torch.zeros(batch_size, device=x.device)
        return Sample(x, values, chain), 0
    

class DiscreteFlowMatching(FlowMatching):


    def loss(self, x, cond, A, b, x0):
        """
        compute batch discrete flow matching loss
        
        :param self: Description
        :param x: (batch, horizon, act_dim+obs_dim)
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        :param A: (batch, horizon, num_cons, sub_dim)
        :param b: (batch, horizon, num_cons)
        :param x0: (batch, horizon, act+obs) 初始分布
        return: 
            loss: 
            info: dict
        
        model API: delta, _, _ = self.model(x, cond, t, A, b)
        """
        device = x.device


        # --- Optimal Transport Pairing ---
        # 注意：这里的重排只对固定约束的任务有效！
        if getattr(self, 'use_ot_batch', False):
            # 1. 计算将 x 对齐到 x0 所需的索引
            # 注意：这里假设 x0 是参考基准（通常是噪声），我们重排数据 x
            idx = compute_ot_permutation_indices(x0, x)
            
            # 2. 重排 x (Target Data)
            x = x[idx]
            
            # 3. 重排 cond (Conditioning)
            # cond 是字典，需要遍历处理每一个 horizon_idx 对应的数据
            new_cond = {}
            for k, v in cond.items():
                new_cond[k] = v[idx]
            cond = new_cond
            
            # 4. 重排 A 和 b (Constraints)
            # 约束通常也是绑定在具体的那条数据轨迹上的，必须同步重排
            if A is not None:
                A = A[idx]
            if b is not None:
                b = b[idx]
        # ------

        # Flow Matching Loss Calculation
        steps = self.n_timesteps
        k = torch.randint(0, steps, (x.size(0),), device=device)
        t_curr = (k.float() / steps).unsqueeze(1).unsqueeze(1)       # [B, 1, 1]
        t_next = ((k.float() + 1) / steps).unsqueeze(1).unsqueeze(1) # [B, 1, 1]
        
        x_curr = (1 - t_curr) * x0 + t_curr * x
        x_next = (1 - t_next) * x0 + t_next * x
        target_delta = x_next - x_curr
        
        # 4. Apply conditioning to model input
        # 这有助于模型在推理时感知到正确的上下文
        x_curr = apply_conditioning(x_curr, cond, self.action_dim)

        pred_delta, _, _ = self.model(x_curr, cond, t_curr.flatten(), A, b)

        # 对于obs condition部分，不计算loss
        target_delta[:, 0, self.action_dim:] = 0.0
        pred_delta[:, 0, self.action_dim:] = 0.0
        loss, info = self.loss_fn(pred_delta, target_delta)

        # debug
        if torch.isnan(loss).any():
            print("nan!")
            print("somehting wrong!")

        return loss, info

    def forward(self, cond, verbose, **kwargs):
        """
        Docstring for forward
        
        :param self: Description
        :param cond: Description
        :param verbose: Description
        还必须输入以下参数：
        :param A (batch, horizon, num_cons, dim_c)
        :param b (batch, horizon, num_cons,)
        :param x0 (batch, horizon, act+obs) 初始分布采样
        """
        return self.conditional_sample(cond, verbose, **kwargs)
    
    @torch.no_grad()
    def conditional_sample(self, cond, verbose=True, return_chain=True, A=None, b=None, x0=None, **sample_kwargs): 
        """
        conditional_sample
        
        :param self: Description
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        :param A [batch, horizon, num_cons, dim_c]
        :param b [batch, horizon, num_cons,]
        :param x0 (batch, horizon, act+obs) 初始分布采样
        :param verbose: Description
        :param return_chain: Description
        :param sample_kwargs: Description
        """
        batch_size = len(cond[0])
        horizon = self.horizon
        assert horizon == A.shape[1]
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, A=A, b=b, x0=x0, verbose=verbose, return_chain=return_chain, **sample_kwargs)    # debug

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, A=None, b=None, x0=None, verbose=True, return_chain=False, sample_fn=None, guide=None, **sample_kwargs):
        """
        Docstring for p_sample_loop
        
        :param self: Description
        :param shape: Description
        :param cond: Description
        :param verbose: Description
        :param return_chain: Description
        :param sample_fn: Description
        :param guide: Description
        :param sample_kwargs: Description
        """
        device = next(iter(self.model.parameters())).device # assuming model is on correct device
        batch_size = shape[0]

        x, A, b = x0.float().to(device), A.float().to(device), b.float().to(device)
        
        # 2. Force start condition
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        
        # Setup progress bar
        iterator = range(self.n_timesteps)
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        

        # 3. Integration Loop: t goes from 0 to 1
        for i in iterator:
            # Current time t (scalar)
            t_value = i / self.n_timesteps
            
            # Create batch time tensor
            t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

            pred_delta, _, _ = self.model(x, cond, t, A, b)
            x = x + pred_delta

            # 5. Re-apply conditioning
            x = apply_conditioning(x, cond, self.action_dim)

            if return_chain: chain.append(x)

        progress.stamp()

        if return_chain: chain = torch.stack(chain, dim=1)
        
        # Return Sample namedtuple to match your interface
        values = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        return Sample(x, values, chain), 0
    


class DiscreteFlowMatchingGo2(FlowMatching):
    
    def loss(self, x, cond, A, b, contact, x0):
        """
        compute batch discrete flow matching loss
        
        :param self: Description
        :param x: (batch, horizon, act_dim+obs_dim)
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        :param A: (batch, horizon, 4, num_cons, 3)
        :param b: (batch, horizon, 4, num_cons)
        :param contact: (batch, horizon, 4)
        :param x0: (batch, horizon, act+obs) 初始分布
        return: 
            loss: 
            info: dict
        
        model API: delta, _, _ = self.model(x, cond, t, A, b, contact)
        """
        device = x.device


        # Flow Matching Loss Calculation
        steps = self.n_timesteps
        k = torch.randint(0, steps, (x.size(0),), device=device)
        t_curr = (k.float() / steps).unsqueeze(1).unsqueeze(1)       # [B, 1, 1]
        t_next = ((k.float() + 1) / steps).unsqueeze(1).unsqueeze(1) # [B, 1, 1]
        
        x_curr = (1 - t_curr) * x0 + t_curr * x
        x_next = (1 - t_next) * x0 + t_next * x
        target_delta = x_next - x_curr
        
        # 4. Apply conditioning to model input
        # 这有助于模型在推理时感知到正确的上下文
        x_curr = apply_conditioning(x_curr, cond, self.action_dim)

        pred_delta, _, _ = self.model(x_curr, cond, t_curr.flatten(), A, b, contact)

        # 对于obs condition部分，不计算loss
        target_delta[:, 0, self.action_dim:] = 0.0
        pred_delta[:, 0, self.action_dim:] = 0.0
        loss, info = self.loss_fn(pred_delta, target_delta)

        # debug
        if torch.isnan(loss).any():
            print("nan!")
            print("somehting wrong!")

        return loss, info

    def forward(self, cond, verbose, **kwargs):
        """
        Docstring for forward
        
        :param self: Description
        :param cond: Description
        :param verbose: Description
        还必须输入以下参数：
        :param A (batch, horizon, 4, num_cons, 3)
        :param b (batch, horizon, 4, num_cons,)
        :param contact (batch, horizon, 4)
        :param x0 (batch, horizon, act+obs) 初始分布采样
        """
        return self.conditional_sample(cond, verbose, **kwargs)
    
    @torch.no_grad()
    def conditional_sample(self, cond, verbose=True, return_chain=True, A=None, b=None, contact=None, x0=None, **sample_kwargs): 
        """
        conditional_sample
        
        :param self: Description
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        :param A [batch, horizon, 4, num_cons, 3]
        :param b [batch, horizon, 4, num_cons,]
        :param contact [batch, horizon, 4]
        :param x0 (batch, horizon, act+obs) 初始分布采样
        :param verbose: Description
        :param return_chain: Description
        :param sample_kwargs: Description
        """
        batch_size = len(cond[0])
        horizon = self.horizon
        assert horizon == A.shape[1]
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, A=A, b=b, contact=contact, x0=x0, verbose=verbose, return_chain=return_chain, **sample_kwargs)    # debug

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, A=None, b=None, contact=None, x0=None, verbose=True, return_chain=False, sample_fn=None, guide=None, **sample_kwargs):
        """
        Docstring for p_sample_loop
        
        :param self: Description
        :param shape: Description
        :param cond: Description
        :param verbose: Description
        :param return_chain: Description
        :param sample_fn: Description
        :param guide: Description
        :param sample_kwargs: Description
        """
        device = next(iter(self.model.parameters())).device # assuming model is on correct device
        batch_size = shape[0]

        x, A, b, contact = x0.float().to(device), A.float().to(device), b.float().to(device), contact.float().to(device)
        
        # 2. Force start condition
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        
        # Setup progress bar
        iterator = range(self.n_timesteps)
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        

        # 3. Integration Loop: t goes from 0 to 1
        for i in iterator:
            # Current time t (scalar)
            t_value = i / self.n_timesteps
            
            # Create batch time tensor
            t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

            pred_delta, _, _ = self.model(x, cond, t, A, b, contact)
            x = x + pred_delta

            # 5. Re-apply conditioning
            x = apply_conditioning(x, cond, self.action_dim)

            if return_chain: chain.append(x)

        progress.stamp()

        if return_chain: chain = torch.stack(chain, dim=1)
        
        # Return Sample namedtuple to match your interface
        values = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        return Sample(x, values, chain), 0
    


class GaugeFlowMatching(nn.Module):

    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps,
            loss_type='l2', action_weight=1.0, loss_discount=1.0, loss_weights=None,
            full_constrained_idxs = [], normed_single_A = None, normed_single_b = None, normed_center_point = None,
            ):
        """
        Gauge Flow Matching Implemention based on Li et al. (2025).
        
        full_constrained_idxs: [idx1, idx2, ...] indices of dimensions constrained by Ax <= b
        normed_single_A: (num_cons, len(full_constrained_idxs))
        normed_single_b: (num_cons,)
        normed_center_point: (len(full_constrained_idxs),) Interior point x_circle
        """

        super().__init__()

        # for normalization obs/act placeholders
        self.means = 0.0
        self.stds = 0.0
        self.act_means = 0.0
        self.act_stds = 0.0

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.n_timesteps = n_timesteps

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        self.loss_weights = loss_weights # Store weights to apply later

        self.full_constrained_idxs = full_constrained_idxs
        
        # Convert constraints to tensors
        self.normed_single_A = None
        self.normed_single_b = None
        self.normed_center_point = None

        if len(self.full_constrained_idxs) > 0:
            # Ensure float32
            self.normed_single_A = torch.from_numpy(normed_single_A).float()
            self.normed_single_b = torch.from_numpy(normed_single_b).float()
            self.normed_center_point = torch.from_numpy(normed_center_point).float()

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
            return:
                (horizon, transition_dim)
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights
    
    def _get_boundary_distance(self, v):
        """
        计算从内部点 x_circle 沿着方向 v 到达多面体边界的距离 d_C(x_circle, v)。
        基于论文 Table 3 的 Linear Constraints 公式。
        
        kappa(v) = max_i { (a_i^T v) / (b_i - a_i^T x_circle) }^+
        d_C = 1 / kappa
        
        v: (batch, ..., constr_dim)
        return: dist (batch, ..., 1)
        """
        # Ensure devices match
        A = self.normed_single_A.to(v.device) # (num_cons, dim)
        b = self.normed_single_b.to(v.device) # (num_cons)
        x_c = self.normed_center_point.to(v.device) # (dim)

        # 1. Numerator: A @ v
        # v shape: (B, H, D), A shape: (K, D) -> (B, H, K)
        # We assume v is NOT normalized in this calculation if we follow Eq 13 strictly,
        # but usually ray casting uses direction. The paper implies v is a direction unit vector for d_C definition.
        # However, for efficiency, we calculate with v and normalize implicitly or explicitly.
        # Let's use normalized direction u = v / ||v|| for the definition.
        
        norm_v = torch.norm(v, dim=-1, keepdim=True) + 1e-6
        u = v / norm_v # Unit vector direction

        numerator = torch.matmul(u, A.T) # (..., num_cons)

        # 2. Denominator: b - A @ x_c
        # This is constant, can be precomputed
        denom = b - torch.matmul(A, x_c) # (num_cons,)
        denom = denom.view(1, 1, -1) # Broadcastable

        # 3. Ratio
        # We only care about constraints we are moving TOWARDS (numerator > 0)
        # If numerator <= 0, the ray moves away from that constraint boundary (infinite distance)
        # kappa is the inverse distance.
        ratio = numerator / (denom + 1e-8)
        
        # 4. Max over constraints (Intersection)
        # kappa = max(0, max_i(ratio_i))
        kappa = torch.max(ratio, dim=-1, keepdim=True)[0]
        kappa = torch.relu(kappa) # Ensure non-negative

        # 5. Distance d_C = 1 / kappa
        # If kappa is 0 (unbounded in that direction), dist is inf. 
        # But for Compact sets, kappa > 0 usually.
        dist = 1.0 / (kappa + 1e-6)
        
        return dist

    def project_to_ball(self, x):
        """
        实现论文中的 Gauge Mapping Inverse: Phi^{-1}(x) : C -> B
        将受约束空间中的点 x 映射到单位球 B 内的点 z。
        
        Equation (14): Phi^{-1}(x) = (x - x_c) / d_C(x_c, direction)
        
        Input: x (batch, horizon, dim) - includes unconstrained dims
        Return: z (batch, horizon, dim)
        """
        if len(self.full_constrained_idxs) == 0:
            return x

        z = x.clone()
        device = x.device
        
        # 提取受约束的部分
        x_cons = x[..., self.full_constrained_idxs] # (B, H, D_cons)
        x_c = self.normed_center_point.to(device)

        # Vector from center
        diff = x_cons - x_c # x - x_circle
        
        # Calculate distance to boundary along the direction of diff
        # d_boundary is the distance from center to boundary
        d_boundary = self._get_boundary_distance(diff) 
        
        # Mapping to unit ball
        # The magnitude in Ball space is proportional to magnitude in C space relative to boundary distance
        # z = diff / d_boundary
        z_cons = diff / (d_boundary + 1e-6)
        
        # Check: If x is on boundary, ||diff|| = d_boundary -> ||z|| = 1. Correct.
        
        z[..., self.full_constrained_idxs] = z_cons
        return z

    def transfer_to_target(self, z):
        """
        实现论文中的 Gauge Mapping Forward: Phi(z) : B -> C
        将单位球 B 内的点 z 映射回受约束空间 C。
        
        Equation (13): Phi(z) = d_C(x_c, z/||z||) * z + x_c
        Wait, Eq 13 says: Phi(z) = d_C * z + x_c. 
        If z is in unit ball, this scales z by the boundary distance.
        Note: The paper formula in Eq 13 seems to have a small notation overlap.
        More precisely: x = x_c + (z / ||z||) * ||z|| * d_C(direction)
                          = x_c + z * d_C(direction)
        This assumes we map percentage of radius.
        
        Input: z (batch, horizon, dim) - latent space
        Return: x (batch, horizon, dim) - target space
        """
        if len(self.full_constrained_idxs) == 0:
            return z

        x = z.clone()
        device = z.device
        
        z_cons = z[..., self.full_constrained_idxs]
        x_c = self.normed_center_point.to(device)
        
        # Distance to boundary along direction z
        d_boundary = self._get_boundary_distance(z_cons)
        
        # Map back
        x_cons_mapped = z_cons * d_boundary + x_c
        
        x[..., self.full_constrained_idxs] = x_cons_mapped
        return x

    def random_sample_source(self, x_ref):
        """
        Sample z_0.
        Constrained dims: Uniformly from Unit Ball (Paper Eq 3 text).
        Unconstrained dims: Standard Normal (Gaussian).
        """
        batch_size, horizon, dim = x_ref.shape
        z0 = torch.randn_like(x_ref)
        
        if len(self.full_constrained_idxs) > 0:
            idx = self.full_constrained_idxs
            dim_cons = len(idx)
            
            # Sample uniform in L2 ball
            # Method: Sample Gaussian, normalize, scale by r^(1/d)
            raw = torch.randn(batch_size, horizon, dim_cons, device=x_ref.device)
            direction = raw / (torch.norm(raw, dim=-1, keepdim=True) + 1e-6)
            
            # r ~ U[0, 1]^(1/d)
            u = torch.rand(batch_size, horizon, 1, device=x_ref.device)
            r = torch.pow(u, 1.0 / dim_cons)
            
            z0_cons = r * direction
            z0[..., idx] = z0_cons
            
        return z0

    def _reflect_in_ball(self, z):
        """
        Inference helper: Ensure z stays within unit ball via reflection/projection.
        Used during Euler integration (Reflected FM variant).
        """
        if len(self.full_constrained_idxs) == 0:
            return z
            
        idx = self.full_constrained_idxs
        z_cons = z[..., idx]
        
        # Check norm
        norm = torch.norm(z_cons, dim=-1, keepdim=True)
        mask_out = (norm > 1.0).float()
        
        if mask_out.sum() == 0:
            return z
            
        # Projection/Reflection Strategy
        # Simple projection: z = z / norm
        # Reflection (as per paper Reflected FM reference): 
        # If outside, reflect velocity? Since we are doing discrete steps x_{t+1},
        # we map the point back.
        # A simple robust way is projection to boundary for feasibility:
        z_cons_proj = z_cons / (norm + 1e-6)
        
        # Or simple "bounce" back: z_new = z_boundary - (z_out - z_boundary)
        # z_new = 2*z_proj - z_out
        # This keeps the "energy" but changes direction.
        # Let's use projection to ensure strict safety as shown in "Projected-GFM"
        # or strict reflection. Let's do Projection for stability in simple implementation.
        # Paper Table 2 shows Projected-GFM is also very good.
        z_cons_safe = z_cons * (1.0 - mask_out) + z_cons_proj * mask_out
        
        z[..., idx] = z_cons_safe
        return z

    def loss(self, x, cond, *args):
        """
        Compute Gauge Flow Matching Loss.
        Training is performed in the latent space (Unit Ball for constrained dims).
        
        x: (batch, horizon, dim) -- Target Data (in Constrained Space C)
        """
        batch_size = len(x)
        device = x.device

        # 1. Transform Data x -> Latent z
        # Eq (2): z_1 = Phi^{-1}(x_1)
        z_1 = self.project_to_ball(x)
        
        # 2. Sample continuous time t uniform in [0, 1]
        t = torch.rand((batch_size,), device=device)

        # 3. Sample source noise z_0
        # Constrained dims: Uniform(Ball), Unconstrained: Normal(0,I)
        z_0 = self.random_sample_source(x)

        # 4. Compute Interpolation (Optimal Transport Path in Latent Space)
        # z_t = (1 - t) * z_0 + t * z_1
        t_b = t.view(batch_size, 1, 1)
        z_t = (1 - t_b) * z_0 + t_b * z_1

        # 5. apply condition
        z_t = apply_conditioning(z_t, cond, self.action_dim)
            
        # 6. Model Forward
        # Predict vector field v_theta(z_t, t)
        v_pred = self.model(z_t, cond, t)

        # 7. Compute Target Velocity
        # v_target = z_1 - z_0
        v_target = z_1 - z_0

        # 不计算condition部分的loss
        v_pred[:, 0, self.action_dim:] = 0.0
        v_target[:, 0, self.action_dim:] = 0.0

        # 8. Compute Loss
        # We assume loss_fn is MSE. Weighted by dimensions.
        loss, info = self.loss_fn(v_pred, v_target) # (B, H, D)

        return loss, info

    def forward(self, cond, verbose, **kwargs):
        return self.conditional_sample(cond, verbose, **kwargs)

    @torch.no_grad()
    def conditional_sample(self, cond, verbose=True, return_chain=True, **sample_kwargs): 
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, verbose=verbose, return_chain=return_chain, **sample_kwargs)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=None, guide=None, **sample_kwargs):
        """
        Inference Loop for Gauge Flow Matching.
        Process:
        1. Sample z_0 from Uniform(Ball).
        2. Integrate ODE in Latent Space (Ball).
        3. Apply Reflection/Projection to keep z_t in Ball.
        4. Map final z_1 -> x_1 using Gauge Map.
        """
        device = next(iter(self.model.parameters())).device 
        batch_size = shape[0]

        # 1. Start from Latent Source (z=0)
        # Construct a dummy tensor to use random_sample_source
        dummy_x = torch.zeros(shape, device=device)
        z = self.random_sample_source(dummy_x)
        
        # Apply conditioning (Project condition to ball first!)
        # Assuming apply_conditioning helper exists in utils or outer scope
        # Here we just conceptualize: cond_z = Phi^{-1}(cond_x)
        z = apply_conditioning(z, cond, self.action_dim)
        
        chain = [z] if return_chain else None
        
        iterator = range(self.n_timesteps)
        
        # Define time step size dt
        dt = 1.0 / self.n_timesteps

        # Default Euler Step
        # Need to ensure sample_fn signature matches
        # def euler_step(model, x, cond, t, dt, ...): return x + v*dt
        
        for i in iterator:
            t_value = i / self.n_timesteps
            t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

            # 2. Predict Velocity & Step
            v_pred = self.model(z, cond, t)
            z = z + v_pred * dt

            # 3. Reflection / Projection (Eq 4 in paper)
            # 必须保证生成过程在 Unit Ball 内部，否则 Gauge Map 会失效或产生巨大的 Lipschitz 误差
            z = self._reflect_in_ball(z)

            # Re-apply conditioning
            z = apply_conditioning(z, cond, self.action_dim)

            if return_chain: chain.append(z)

        # 4. Final Transform: Latent (z) -> Data (x)
        # Gauge Mapping Forward
        x_final = self.transfer_to_target(z)
        x_final = apply_conditioning(x_final, cond, self.action_dim)
        
        # If returning chain, we might want to map the whole chain or just the end
        # Usually visualization wants the x-space chain
        if return_chain:
            # Map all steps to x-space
            chain_z = torch.stack(chain, dim=1) # (B, Steps, H, D)
            chain_x = self.transfer_to_target(chain_z)
        else:
            chain_x = None
        
        # Return Sample
        values = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        return Sample(x_final, values, chain_x), 0
