import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import mujoco
import gymnasium as gym # 仅用于确认环境依赖，实际计算使用 mujoco 原生绑定
from src.utils.eval import evaluate_dismatch_metrics

# --- 1. 用户提供的 Hopper-v5 XML 定义 ---
# 我们将其直接嵌入代码，确保计算模型与你的环境完全一致
HOPPER_V5_XML = """
<mujoco model="hopper">
  <compiler angle="degree" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1" solimp=".8 .8 .01" solref=".02 1"/>
    <motor ctrllimited="true" ctrlrange="-.4 .4"/>
  </default>
  <option integrator="RK4" timestep="0.002"/>
  <visual>
    <map znear="0.02"/>
  </visual>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
    <body name="torso" pos="0 0 1.25">
      <camera name="track" mode="trackcom" pos="0 -3 -0.25" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 -1.25" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 -1.25" ref="1.25" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom friction="0.9" name="torso_geom" size="0.05 0.19999999999999996" type="capsule"/>
      <body name="thigh" pos="0 0 -0.19999999999999996">
        <joint axis="0 -1 0" name="thigh_joint" pos="0 0 0" range="-150 0" type="hinge"/>
        <geom friction="0.9" pos="0 0 -0.22500000000000009" name="thigh_geom" size="0.05 0.22500000000000003" type="capsule"/>
        <body name="leg" pos="0 0 -0.70000000000000007">
          <joint axis="0 -1 0" name="leg_joint" pos="0 0 0.25" range="-150 0" type="hinge"/>
          <geom friction="0.9" name="leg_geom" size="0.04 0.25" type="capsule"/>
          <body name="foot" pos="0.13 0 -0.35">
            <joint axis="0 -1 0" name="foot_joint" pos="-0.13 0 0.1" range="-45 45" type="hinge"/>
            <geom friction="2.0" pos="-0.065 0 0.1" quat="0.70710678118654757 0 -0.70710678118654746 0" name="foot_geom" size="0.06 0.195" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="thigh_joint"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="leg_joint"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="foot_joint"/>
  </actuator>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
</mujoco>
"""


def plot_hopper_unified(traj_path_list, label_list, height_limit, plot_horizon_length, save_path, mode='rollout', select='length'):
    """
    Unified Hopper Trajectory Visualization (ICML Style).
    
    Args:
        traj_path_list (list): List of .npz file paths.
        label_list (list): List of algorithm names.
        height_limit (float): Safety threshold for root height.
        plot_horizon_length (int): Max steps to visualize.
        save_path (str): Output filename (.pdf).
        mode (str): 'rollout' (Global X-axis) or 'generated' (Time-step X-axis).
    """
    
    if mode not in ['rollout', 'generated']:
        raise ValueError("Mode must be either 'rollout' or 'generated'")

    # --- 1. Init MuJoCo ---
    try:
        model = mujoco.MjModel.from_xml_string(HOPPER_V5_XML)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"MuJoCo Init Error: {e}")
        return

    # --- 2. Style Setup ---
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 14, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "lines.linewidth": 1.5, "figure.dpi": 300,
        "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5
    })

    num_algs = len(traj_path_list)
    fig, axes = plt.subplots(num_algs, 1, figsize=(8, 2.8 * num_algs), sharex=True, sharey=True)
    if num_algs == 1: axes = [axes]

    colors = {
        'traj_line': '#1f77b4', 'violation_shade': '#d62728', 
        'limit_line': '#333333', 'skeleton': '#525252',
        'root_safe': '#1f77b4', 'root_fail': '#d62728'
    }

    # --- 3. Main Loop ---
    for idx, (filepath, label) in enumerate(zip(traj_path_list, label_list)):
        ax = axes[idx]
        
        try:
            # A. Load Data
            npz_data = np.load(filepath, allow_pickle=True)
            
            trajectories = [] # list of (T, obs_dim)
            x_axes = []       # list of (T,) corresponding X values
            ret_list = []
            
            if mode == 'rollout':
                # Load from obs_traj_list / obs_expand_traj_list
                # Raw data might be object arrays
                raw_obs = npz_data['obs_traj_list']
                raw_exp = npz_data['obs_expand_traj_list']
                raw_ret = npz_data['ret_list']
                
                # Convert all valid trajectories to a standard list format
                for i in range(len(raw_obs)):
                    try:
                        t_obs = np.array(raw_obs[i]).astype(float)
                        t_exp = np.array(raw_exp[i]).astype(float)
                        ret_list.append(raw_ret[i])
                        if len(t_obs) > 1:
                            trajectories.append(t_obs)
                            x_axes.append(t_exp[:, -1]) # Global X
                    except: continue
                    
            elif mode == 'generated':
                # Load from gene_traj (Batch, T, Dim)
                gene_batch = npz_data['gene_traj']
                for i in range(len(gene_batch)):
                    t_obs = gene_batch[i].astype(float)
                    trajectories.append(t_obs)
                    x_axes.append(np.arange(len(t_obs)).astype(float)) # Time step X
            
            if not trajectories:
                print(f"No valid trajectories found in {filepath}")
                continue

            # B. Select "Most Unsafe" Trajectory
            # Strategy: Pick the one with highest sum of height violations
            best_idx = 0
            
            max_score = -float('inf')
            
            for i, traj in enumerate(trajectories):
                # z is index 0
                z_vals = traj[:, 0]
                diff = z_vals - height_limit

                if select == 'unsafe' or "RoSD" in label:
                    # Score = Sum of positive violations
                    # score = np.sum(diff[diff > 0])
                    score = np.max(diff)
                    
                    # # If tied or safe, prefer longer ones
                    # if score == 0: 
                    #     score = -1.0 / (len(z_vals) + 1) # Small negative, longer is "less negative"
                    
                    if score > max_score:
                        max_score = score
                        best_idx = i
                else:
                    # Score = length
                    score = traj.shape[0]
                    if score > max_score:
                        max_score = score
                        best_idx = i

            # Extract final data
            final_traj = trajectories[best_idx]     # (T, Obs)
            final_x = x_axes[best_idx]              # (T,)
            
            # Clip to horizon length
            T = min(len(final_traj), plot_horizon_length)
            final_traj = final_traj[:T]
            final_x = final_x[:T]
            z_arr = final_traj[:, 0]
            
            # C. Plot Background Curves
            ax.plot(final_x, z_arr, color=colors['traj_line'], alpha=0.4, lw=1.5, label="Torso Trajectory")
            ax.axhline(y=height_limit, color=colors['limit_line'], ls='--', lw=1.5, label="Height Limit")
            eps = 1e-3
            ax.fill_between(final_x, z_arr, height_limit, where=(z_arr > height_limit + eps), 
                            color=colors['violation_shade'], alpha=0.2, interpolate=True)

            # D. Draw Skeleton
            stride = max(1, 8)
            
            for t in range(0, T, stride):
                # Construct qpos
                qpos = np.zeros(6)
                
                if mode == 'rollout':
                    # Use actual physical X
                    qpos[0] = final_x[t]
                else:
                    # Use local 0, mapping happens later
                    qpos[0] = 0.0
                
                # Common joints
                qpos[1] = final_traj[t, 0] # Z
                qpos[2] = final_traj[t, 1] # Angle
                qpos[3:] = final_traj[t, 2:5] # Thigh, Leg, Foot
                
                # Update MuJoCo
                data.qpos[:] = qpos
                mujoco.mj_kinematics(model, data)
                
                # Get coords
                def get_xz_global(body_name):
                    pos = data.body(body_name).xpos.copy()
                    return pos[0], pos[2] # x, z
                
                x_root, z_root = get_xz_global('torso')
                x_hip, z_hip = get_xz_global('thigh')
                x_knee, z_knee = get_xz_global('leg')
                x_ankle, z_ankle = get_xz_global('foot')
                
                # Foot rotation logic
                mat_foot = data.body('foot').xmat.reshape(3, 3)
                pos_foot_center = data.body('foot').xpos.copy()
                vec_toe = mat_foot @ np.array([0.2, 0, 0])
                vec_heel = mat_foot @ np.array([-0.1, 0, 0])
                
                x_toe, z_toe = pos_foot_center[0] + vec_toe[0], pos_foot_center[2] + vec_toe[2]
                x_heel, z_heel = pos_foot_center[0] + vec_heel[0], pos_foot_center[2] + vec_heel[2]
                
                # Coordinate Mapping Function
                def map_coord(x_phys, z_phys):
                    if mode == 'rollout':
                        return x_phys, z_phys # Identity
                    else:
                        # Generated mode: Plot X = Local X + TimeStep
                        return x_phys + final_x[t], z_phys

                # Apply mapping
                mx_root, mz_root = map_coord(x_root, z_root)
                mx_hip, mz_hip = map_coord(x_hip, z_hip)
                mx_knee, mz_knee = map_coord(x_knee, z_knee)
                mx_ankle, mz_ankle = map_coord(x_ankle, z_ankle)
                mx_toe, mz_toe = map_coord(x_toe, z_toe)
                mx_heel, mz_heel = map_coord(x_heel, z_heel)
                
                # Draw Lines
                # Torso visual
                theta = qpos[2]
                # Note: If mode=generated, we need local X for torso top calculation first
                phys_x_top = x_root + np.sin(theta) * 0.3 # Use physics coords first
                phys_z_top = z_root + np.cos(theta) * 0.3
                mx_top, mz_top = map_coord(phys_x_top, phys_z_top) # Then map
                
                kw = dict(color=colors['skeleton'], lw=1.2, alpha=0.3, solid_capstyle='round')
                ax.plot([mx_top, mx_root], [mz_top, mz_root], **kw) # Torso
                ax.plot([mx_root, mx_hip], [mz_root, mz_hip], **kw) # Hip conn
                ax.plot([mx_hip, mx_knee], [mz_hip, mz_knee], **kw) # Thigh
                ax.plot([mx_knee, mx_ankle], [mz_knee, mz_ankle], **kw) # Leg
                # ax.plot([mx_heel, mx_toe], [mz_heel, mz_toe], **kw) # Foot

                # Root Marker
                c = colors['root_fail'] if mz_root > height_limit else colors['root_safe']
                ax.scatter(mx_root, mz_root, color=c, s=30, zorder=10, edgecolors='white', lw=0.8)

            # E. Labeling
            ax.text(0.02, 0.9, f"{label}", transform=ax.transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9))
            ax.set_ylabel("Height ($z$)")
            ax.set_ylim(0, max(height_limit * 1.5, 2.0))
            sns.despine(ax=ax)

        except Exception as e:
            print(f"Error processing {label}: {e}")
            import traceback
            traceback.print_exc()

    # F. Final Layout
    xlabel = "Global Position ($x$) [m]" if mode == 'rollout' else "Prediction Horizon ($t$)"
    axes[-1].set_xlabel(xlabel)
    
    handles, labels_leg = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))

    # --- 修改点 1: 调整图例位置 ---
    # bbox_to_anchor=(0.5, 0.98) 将图例稍微拉下来一点 (原为 1.02)
    # 建议把 loc 改为 'lower center' 并把 y 设为 1.0 左右，或者保持 upper center 并微调
    fig.legend(by_label.values(), by_label.keys(), 
               loc='lower center',       # 改用 lower center，让图例的底部对齐锚点
               bbox_to_anchor=(0.5, 0.9), # 锚点设在 0.95 (即子图的顶部)
               ncol=3, frameon=False)
    
    # --- 修改点 2: 调整 tight_layout 和 subplots_adjust ---
    # rect 参数可以直接为 tight_layout 指定内容区域 [left, bottom, right, top]
    # top=0.95 意味着给图例留出 5% 的空间，其余给子图，这样就紧凑了
    plt.tight_layout(rect=[0, 0, 1, 0.95])


    # plt.subplots_adjust(top=0.9)
    print(f"Saving ({mode}) figure to {save_path}")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)

    # calculate_traj_metric(traj_path_list, label_list, height_limit, vel_scale=0.01, 
    #                       obs_v_idx=6, v_limit=2.5, height_min=0.8)


def calculate_traj_metric(traj_path_list, label_list, height_limit, vel_scale, obs_v_idx, v_limit, height_min):
    
    # --- 3. Main Loop ---
    for idx, (filepath, label) in enumerate(zip(traj_path_list, label_list)):
    
        # A. Load Data
        npz_data = np.load(filepath, allow_pickle=True)
        
        trajectories = [] # list of (T, obs_dim)
        x_axes = []       # list of (T,) corresponding X values
        ret_list = []
        
        # Load from obs_traj_list / obs_expand_traj_list
        # Raw data might be object arrays
        raw_obs = npz_data['obs_traj_list']
        raw_exp = npz_data['obs_expand_traj_list']
        raw_ret = npz_data['ret_list']
        
        # Convert all valid trajectories to a standard list format
        for i in range(len(raw_obs)):
            try:
                t_obs = np.array(raw_obs[i]).astype(float)
                t_exp = np.array(raw_exp[i]).astype(float)
                ret_list.append(raw_ret[i])
                if len(t_obs) > 1:
                    trajectories.append(t_obs)
                    x_axes.append(t_exp[:, -1]) # Global X
            except: continue

        # 统计每个rollout中约束最大违背量,和约束违背比例
        max_violation_ratio = []
        violation_cnt_ratio = []
        for traj in trajectories:
            cur_max_height = np.max(traj[:, 0])
            cur_max_ratio = (cur_max_height - height_limit) / height_limit
            if v_limit is not None:
                # print("debug")
                cur_max_v = np.max(np.abs(traj[:, obs_v_idx]))
                cur_max_v_ratio = (cur_max_v - v_limit) / v_limit
                cur_max_ratio = max(cur_max_ratio, cur_max_v_ratio)
                flag = (traj[:, 0] > height_limit) | (np.abs(traj[:, obs_v_idx]) > v_limit)
            else:
                flag = (traj[:, 0] > height_limit)

            cur_unsafe_cnt = np.sum(flag)

            cur_unsafe_ratio = cur_unsafe_cnt / traj.shape[0]
            max_violation_ratio.append(cur_max_ratio)
            violation_cnt_ratio.append(cur_unsafe_ratio)
        max_violation_ratio = np.array(max_violation_ratio)
        violation_cnt_ratio = np.array(violation_cnt_ratio)

        print(f"Label {label}")
        print(f"max_violation_ratio: {np.max(max_violation_ratio)} \n {max_violation_ratio}")
        print(f"violation cnt ratio: {np.max(violation_cnt_ratio)} \n {violation_cnt_ratio}")
        
        ret_arr = np.array(ret_list)
        ret_mean = np.mean(ret_arr)
        ret_std = np.std(ret_arr)
        print(f"Ret mean: {ret_mean}.  Ret std: {ret_std}")


def calculate_generate_metric(traj_path_list, label_list):

    # --- 3. Main Loop ---
    for idx, (filepath, label) in enumerate(zip(traj_path_list, label_list)):
    
        # A. Load Data
        npz_data = np.load(filepath, allow_pickle=True)

        true_traj = npz_data['true_traj'] # (B, H, O)
        gene_traj = npz_data['gene_traj']
        horizon = true_traj.shape[1]
        check_horizon = [i for i in range(1, horizon)]
        eval_metrics = evaluate_dismatch_metrics(
            sampled_traj=gene_traj,
            true_traj=true_traj,
            check_horizon_list=check_horizon
        )
        print(f"Label {label}")
        print(f"----------------------")
        for key, val in eval_metrics.items():
            print(f"{key}:")
            print(f"mean {np.mean(val)}")
            print(f"std {np.std(val)}")
            print(f"min {np.min(val)}")
            print(f"max {np.max(val)}")
            print(f"---------------------")




def plot_hopper_height_vel(traj_path_list, label_list, height_limit, vel_scale, height_min, v_max, v_min, plot_horizon_length, save_path, select='length', env='hopper'):
    """
    修改版：在ICML单栏宽度内绘制 2列 x N行 的子图网格，对比不同方法的生成轨迹约束满足情况。
    
    :param traj_path_list: [final_traj.npz, ...]
    :param label_list: [method_name, ...]
    :param height_limit: z 方向高度限制
    :param vel_scale: 速度系数
    :param plot_horizon_length: 绘制的轨迹horizon长度
    :param save_path: 保存路径
    :param select: 'length' (batch中取一条) 或 'all' (batch中所有点)
    """
    
    # === 1. 样式配置 (ICML / 论文 风格) ===
    # 字体和线宽针对小图进行了微调，保证缩小后文字依然清晰
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.7,
        "figure.dpi": 300,
        "mathtext.fontset": "cm" # 使用 LaTeX 风格数学字体
    })

    # === 2. 布局计算 ===
    num_methods = len(traj_path_list)
    ncols = 2
    nrows = math.ceil(num_methods / ncols)
    
    # ICML 单栏宽度约为 3.25 英寸
    # 高度根据行数动态计算，每行大约给 1.3 英寸
    fig_width = 3.25
    fig_height = 1.3 * nrows
    
    # sharex=False, sharey=True 可以让Y轴统一，方便对比高度违规情况
    # 但如果不同方法的 velocity 差异巨大，sharex=True 可能会压缩某些图
    # 这里建议 sharey=True (高度约束是一样的)，sharex=False (根据数据自适应)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True)
    axes = axes.flatten() # 展平以便遍历

    # === 3. 维度定义 ===
    if env == 'hopper':
        DIM_Z = 0
        DIM_VZ = 6 
        y_aixs_max = 1.8
    else:
        DIM_Z = 0
        DIM_VZ = 9
        y_axis_max = 1.5
    
    # 颜色库
    colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']
    
    # === 4. 循环绘制 ===
    for i in range(nrows * ncols):
        ax = axes[i]
        
        # 如果子图数量多于方法数量，隐藏多余的子图
        if i >= num_methods:
            ax.axis('off')
            continue
            
        path = traj_path_list[i]
        label = label_list[i]
        
        # --- 数据加载 ---
        points_z, points_vz = [], []
        has_data = False
        
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                # 仅处理 Generated 模式
                if 'gene_traj' in data:
                    gene_traj = data['gene_traj'] # (batch, horizon, dim)
                    
                    if select == 'length':
                        # 取 batch 中第一个轨迹
                        traj = gene_traj[0]
                        traj = traj[1:min(traj.shape[0], plot_horizon_length)]
                        points_z, points_vz = traj[:, DIM_Z], traj[:, DIM_VZ]
                        
                    elif select == 'all':
                        # 展平 batch 和 horizon
                        # 截断 horizon
                        traj_cut = gene_traj[:, 1:plot_horizon_length, :]
                        traj_flat = traj_cut.reshape(-1, traj_cut.shape[-1])
                        points_z, points_vz = traj_flat[:, DIM_Z], traj_flat[:, DIM_VZ]
                    
                        # # safety check
                        # flag = (points_z + vel_scale * points_vz < height_limit + 1e-3) & \
                        #     (points_z > height_min - 1e-3) & (points_vz < v_max + 1e-3) & \
                        #     (points_vz > v_min - 1e-3)
                        # safe_rate = np.sum(flag) / points_z.shape[0]
                        # print(f"Label: {label} SafeRate: {safe_rate}")

                    has_data = True
            except Exception as e:
                print(f"Error loading {path}: {e}")

        # --- 散点绘制 ---
        if has_data and len(points_z) > 0:
            # 这里的颜色根据 i 变化，也可以固定一种颜色
            color = colors[i % len(colors)]
            # 散点设置：透明度低一点以便看清密度
            alpha_val = 0.6 if select == 'length' else 0.15
            size_val = 6 if select == 'length' else 1.5
            
            # if 'PolyFlow' in label:
            #     outlier_mask = (points_vz < -1) & (points_z < 0.95)
            #     points_vz = points_vz[~outlier_mask]
            #     points_z = points_z[~outlier_mask]

            ax.scatter(points_vz, points_z, c=color, s=size_val, alpha=alpha_val, 
                       edgecolors='none', rasterized=True)
            
            # --- 动态计算边界绘制范围 ---
            # 获取当前数据的 x 范围，并稍微外扩一点
            x_min_data, x_max_data = np.min(points_vz), np.max(points_vz)
            margin = (x_max_data - x_min_data) * 0.1 if (x_max_data != x_min_data) else 0.5
            # x_range = np.linspace(x_min_data - margin, x_max_data + margin, 200)
        else:
            # 默认范围
            # x_range = np.linspace(-2, 2, 200)
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)

        x_range = np.linspace(-3, 3, 200)
        # --- 绘制安全边界 (Constraint Boundary) ---
        # 逻辑: z <= H  且  z <= H - k * vz
        limit_static = np.full_like(x_range, height_limit)
        limit_cbf = height_limit - vel_scale * x_range
        boundary_z = np.minimum(limit_static, limit_cbf)
        
        ax.plot(x_range, boundary_z, color='black', linestyle='--', linewidth=1.2, label='Boundary')

        if height_min is not None and v_max is not None and v_min is not None:
            # z >= 0.8
            boundary_y = np.full_like(x_range, height_min)
            ax.plot(x_range, boundary_y, color='black', linestyle='--', linewidth=1.2, label='Boundary')
            # v <= 2.5
            boundary_y = np.linspace(0.7, 1.8, 150)
            x_range = np.full_like(boundary_y, v_max)
            ax.plot(x_range, boundary_y, color='black', linestyle='--', linewidth=1.2, label='Boundary')
            # v >= -2.5
            x_range = np.full_like(boundary_y, v_min)
            ax.plot(x_range, boundary_y, color='black', linestyle='--', linewidth=1.2, label='Boundary')

        # --- 子图装饰 ---
        ax.set_title(label, fontsize=9, pad=3)
        
        # 坐标轴范围限制 (Y轴)
        # 确保能看到 safe boundary (height_limit) 和数据最高点
        # y_max_data = np.max(points_z) if (has_data and len(points_z)>0) else height_limit
        # ax.set_ylim(0, max(height_limit * 1.3, y_max_data * 1.1))


        ax.set_ylim((0.7, y_axis_max))
        ax.set_xlim((-3, 3))
        
        # 去除上方和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # # 智能标签：只在左侧第一列显示 Y Label，只在最后一行显示 X Label
        # # 判断是否为左侧列
        # if i % ncols == 0:
        #     ax.set_ylabel(r'$z$(m)', labelpad=1)
        
        # # 判断是否为最后一行 (注意：如果最后一行有的格子是空的，倒数第二行对应的位置也需要标签)
        # # 简单处理：给所有图都加 x 标签，但为了紧凑布局，可以用 layout='compressed'
        # # 或者：仅当行号是最后一行时。
        # row_idx = i // ncols
        # if row_idx == nrows - 1:
        #     ax.set_xlabel(r'$v_z$(m/s)', labelpad=1)

        # 判断是否为左侧第一列 -> 放置 Y 轴标题 (z)
        if i % ncols == 0:
            # transform=ax.transAxes 意味着 (0,0) 是左下, (1,1) 是右上
            # x=-0.05: 稍微向左一点
            # y=1.02:  放在顶部上方一点点
            # ha='right': 右对齐，防止遮挡图内内容
            # rotation=0: 水平显示，不竖着排
            ax.text(-0.05, 1.03, r'$z$', transform=ax.transAxes, 
                    ha='left', va='bottom', fontsize=9, rotation=0)

        # 判断是否为最后一行 -> 放置 X 轴标题 (vz)
        row_idx = i // ncols
        if row_idx == nrows - 1:
            # x=1.02: 放在右侧外面一点点
            # y=0.02: 对齐 X 轴线
            # ha='left': 左对齐
            ax.text(1.0, -0.1, r'$v_z$', transform=ax.transAxes, 
                    ha='left', va='bottom', fontsize=9)

    # === 5. 保存与展示 ===
    # tight_layout 会自动调整子图间距，h_pad 控制行间距，w_pad 控制列间距
    plt.tight_layout(pad=0.15, h_pad=0.5, w_pad=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        if save_path.endswith('.pdf'):
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


if __name__=='__main__':

    #%%% 
    # plot_hopper_unified(
    #     traj_path_list=[
    #         "outputs/hoppercpx/flow_sample/42_2026-01-02_13-42-02/final_traj.npz",
    #         "outputs/hoppercpx/polyflow_sample/0_2026-01-04_13-39-35/final_traj.npz",
    #         "outputs/hoppercpx/safeflow_sample/42_2026-01-04_15-36-05/final_traj.npz",
    #         "outputs/hoppercpx/RoS_sample/42_2026-01-04_13-59-20/final_traj.npz",
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", "RoSD"], 
    #     height_limit=1.6, 
    #     plot_horizon_length=800, 
    #     save_path="hopper_rollout.pdf",
    #     mode="rollout",
    #     select='length'
    # )

    # plot_hopper_height_vel(
    #     traj_path_list=[
    #         "outputs/hoppercpx/flow_sample/42_2026-01-02_13-42-02/final_traj.npz",
    #         "outputs/hoppercpx/polyflow_sample/0_2026-01-04_13-39-35/final_traj.npz",
    #         "outputs/hoppercpx/safeflow_sample/42_2026-01-04_15-36-05/final_traj.npz",
    #         "outputs/hoppercpx/RoS_sample/42_2026-01-04_13-59-20/final_traj.npz",
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", "RoSD"], 
    #     height_limit=1.6, 
    #     vel_scale=0.01,
    #     plot_horizon_length=100, 
    #     save_path="hopper_height_vel.pdf",
    #     select='all'
    # )

    calculate_generate_metric(
        traj_path_list=[
            "outputs/hoppercpx/flow_sample/42_2026-01-02_13-42-02/final_traj.npz",
            "outputs/hoppercpx/polyflow_sample/0_2026-01-04_13-39-35/final_traj.npz",
            "outputs/hoppercpx/safeflow_sample/42_2026-01-04_15-36-05/final_traj.npz",
            "outputs/hoppercpx/RoS_sample/42_2026-01-04_13-59-20/final_traj.npz",
        ], 
        label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", "RoSD"], 
    )

    #%%% height 1.5  vel_scale 0.01 height_min=0.8 v_max=2.5 v_min=-2.5
    # plot_hopper_unified(
    #     traj_path_list=[
    #         "outputs/hoppercpx/flow_sample/42_2026-01-02_13-42-02/final_traj.npz",
    #         "outputs/hoppercpx2/polyflow_train/42_2026-01-05_14-55-21/final_traj.npz",
    #         "outputs/hoppercpx2/safeflow_sample_step100/42_2026-01-05_15-43-22/final_traj.npz",
    #         "outputs/hoppercpx2/RoS_sample/42_2026-01-05_16-16-56/final_traj.npz",
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", "RoSD"], 
    #     height_limit=1.5, 
    #     plot_horizon_length=800, 
    #     save_path="hopper_rollout.png",
    #     mode="rollout",
    #     select='length'
    # )

    # plot_hopper_height_vel(
    #     traj_path_list=[
    #         "outputs/hoppercpx/flow_sample/42_2026-01-02_13-42-02/final_traj.npz",
    #         "outputs/hoppercpx2/polyflow_train/42_2026-01-05_14-55-21/final_traj.npz",
    #         "outputs/hoppercpx2/safeflow_sample_step100/42_2026-01-05_15-43-22/final_traj.npz",
    #         "outputs/hoppercpx2/RoS_sample/42_2026-01-05_16-16-56/final_traj.npz",
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", "RoSD"], 
    #     height_limit=1.5, 
    #     vel_scale=0.01,
    #     height_min=0.8,
    #     v_max=2.5,
    #     v_min=-2.5,
    #     plot_horizon_length=100, 
    #     save_path="hopper_height_vel.pdf",
    #     select='all'
    # )

    # calculate_generate_metric(
    #     traj_path_list=[
    #         "outputs/hoppercpx/flow_sample/42_2026-01-02_13-42-02/final_traj.npz",
    #         "outputs/hoppercpx2/polyflow_train/42_2026-01-05_14-55-21/final_traj.npz",
    #         "outputs/hoppercpx2/safeflow_sample_step100/42_2026-01-05_15-43-22/final_traj.npz",
    #         "outputs/hoppercpx2/RoS_sample/42_2026-01-05_16-16-56/final_traj.npz",
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", "RoSD"], 
    # )
