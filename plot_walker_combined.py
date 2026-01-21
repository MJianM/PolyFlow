import matplotlib.pyplot as plt
import numpy as np
import os

def plot_2x5_compact(
    simple_config,
    hard_config,
    save_path,
    plot_horizon_length=100,
    select='all'
):
    """
    绘制 2行 x 5列 的紧凑对比图
    Row 1: Simple Task
    Row 2: Hard Task
    Cols: Flow, PolyFlow, SafeFlow, RoSD, GaugeFlow
    """
    
    # === 1. 样式配置 ===
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "figure.dpi": 300,
        "mathtext.fontset": "cm"
    })

    PALETTE = {
        'Flow': '#7f7f7f',       # 灰色
        'PolyFlow': '#d62728',   # 红色
        'SafeFlow': '#1f77b4',   # 蓝色
        'RoSD': '#2ca02c',       # 绿色
        'GaugeFlow': "#8b2ca0"   # 紫色
    }

    # === 2. 准备画布 ===
    # ICML 双栏宽度 ~6.75英寸。高度设为 3.0 左右以保持比例
    # sharex=True, sharey=True 自动隐藏内部刻度
    fig, axes = plt.subplots(2, 5, figsize=(6.75, 2.7), sharex=True, sharey=True)
    
    # 调整间距：hspace, wspace 设得很小以实现“紧凑”
    plt.subplots_adjust(wspace=0.1, hspace=0.15, left=0.06, right=0.96, bottom=0.12, top=0.90)

    # === 3. 数据加载辅助函数 ===
    def load_data(path, horizon, sel_mode):
        if not os.path.exists(path):
            return np.array([]), np.array([])
        try:
            data = np.load(path, allow_pickle=True)
            if 'gene_traj' not in data:
                return np.array([]), np.array([])
            
            gene_traj = data['gene_traj']
            DIM_Z, DIM_VZ = 0, 9
            
            if sel_mode == 'length':
                traj = gene_traj[0]
                traj = traj[1:min(traj.shape[0], horizon)]
                return traj[:, DIM_Z], traj[:, DIM_VZ]
            elif sel_mode == 'all':
                traj_cut = gene_traj[:, 1:horizon, :]
                traj_flat = traj_cut.reshape(-1, traj_cut.shape[-1])
                return traj_flat[:, DIM_Z], traj_flat[:, DIM_VZ]
        except:
            return np.array([]), np.array([])

    def draw_boundary(ax, config):
        h_lim = config['height_limit']
        v_scale = config['vel_scale']
        x_range_vals = np.linspace(-3, 3, 200)
        
        # CBF 斜线边界
        limit_static = np.full_like(x_range_vals, h_lim)
        limit_cbf = h_lim - v_scale * x_range_vals
        boundary_z = np.minimum(limit_static, limit_cbf)
        
        ax.plot(x_range_vals, boundary_z, color='black', linestyle='--', linewidth=1.0)
        
        # Hard Task 额外边界 (Box constraints)
        if config['height_min'] is not None:
            ax.axhline(y=config['height_min'], color='black', linestyle='--', linewidth=1.0)
        if config['v_max'] is not None:
            # 用 vlines 绘制垂直线，限制 y 范围看起来更整洁
            ax.vlines(config['v_max'], 0.7, 1.8, colors='black', linestyles='--', linewidth=1.0)
        if config['v_min'] is not None:
            ax.vlines(config['v_min'], 0.7, 1.8, colors='black', linestyles='--', linewidth=1.0)

    # === 4. 循环绘制 ===
    # 定义行内容：(行索引, 配置对象, 任务名称)
    rows_info = [
        (0, simple_config, "Simple Task"),
        (1, hard_config, "Hard Task")
    ]

    for row_idx, config, task_name in rows_info:
        traj_list = config['traj_list']
        label_list = config['label_list']
        
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            
            # 获取当前列对应的方法信息
            path = traj_list[col_idx]
            label_raw = label_list[col_idx]
            
            # 确定颜色
            color_key = label_raw.split('(')[0]
            if color_key not in PALETTE: color_key = 'Flow'
            
            # 1. 绘制散点
            pz, pv = load_data(path, plot_horizon_length, select)
            if len(pz) > 0:
                ax.scatter(pv, pz, c=PALETTE.get(color_key, 'black'), s=1.0, alpha=0.15, 
                           edgecolors='none', rasterized=True)
            
            # 2. 绘制边界 (根据当前行的 config)
            draw_boundary(ax, config)
            
            # 3. 设置范围
            ax.set_ylim(0.7, 1.6)
            ax.set_xlim(-2.8, 2.8)
            
            # 4. 仅在第一行显示标题 (Method Name)
            if row_idx == 0:
                # 粗体高亮 PolyFlow
                # fw = 'bold' if 'Poly' in label_raw else 'normal'
                ax.set_title(label_raw, fontsize=9, pad=4, fontweight='normal')
            
            # 5. 坐标轴标签管理
            # 只在最后一行显示 X Label
            if row_idx == 1:
                # 稍微调整 labelpad 让它离轴近一点
                ax.set_xlabel(r'$v_z$', labelpad=1)
            
            # 只在第一列显示 Y Label
            if col_idx == 0:
                ax.set_ylabel(r'$z$', labelpad=1)
            
            # 去除上右边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # === 5. 添加右侧任务名称 (Simple / Hard) ===
    # 使用 fig.text 或在最右侧子图添加 text
    # 这里我们在每一行的最右侧子图外侧添加
    
    # Simple Task Label (Row 0)
    ax_row0_right = axes[0, -1]
    ax_row0_right.text(1.15, 0.5, "Simple\nTask", transform=ax_row0_right.transAxes,
                       ha='center', va='center', rotation=-90, fontsize=9, fontweight='bold', color='#444')
    
    # Hard Task Label (Row 1)
    ax_row1_right = axes[1, -1]
    ax_row1_right.text(1.15, 0.5, "Hard\nTask", transform=ax_row1_right.transAxes,
                       ha='center', va='center', rotation=-90, fontsize=9, fontweight='bold', color='#444')

    # === 6. 保存 ===
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # bbox_inches='tight' 会裁掉多余白边
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()

# === 配置数据 (保持你的路径不变) ===
hard_config = {
    'traj_list': [
        "outputs/walker2dcpx2/flow_sample_step100/42_2026-01-08_14-32-32/final_traj.npz",
        "outputs/walker2dcpx2/polyflow_sample/42_2026-01-08_18-27-38/final_traj.npz",
        "outputs/walker2dcpx2/safeflow_sample_step100/42_2026-01-08_18-30-27/final_traj.npz",
        "outputs/walker2dcpx2/RoS_sample_horizon160/42_2026-01-08_14-52-18/final_traj.npz",
        "outputs/walker2dcpx2/gaugeflow_train_step100/42_2026-01-09_15-58-19/final_traj.npz"
    ],
    'label_list': ["Flow", "PolyFlow", "SafeFlow", 'RoSD', 'GaugeFlow'],
    'height_limit': 1.35,
    'vel_scale': 0.01,
    'height_min': 0.9,
    'v_max': 1.4,
    'v_min': -1.4
}

simple_config = {
    'traj_list': [
        "outputs/walker2dcpx/flow_sample_step100/42_2026-01-08_23-53-54/final_traj.npz",
        "outputs/walker2dcpx/polyflow_sample/42_2026-01-09_00-35-45/final_traj.npz",
        "outputs/walker2dcpx/safeflow_sample_step100/42_2026-01-09_13-20-46/final_traj.npz",
        "outputs/walker2dcpx/RoS_sample_horizon160/42_2026-01-09_13-19-30/final_traj.npz",
        "outputs/walker2dcpx/gaugeflow_train_step100/42_2026-01-09_15-57-56/final_traj.npz"
    ],
    'label_list': ["Flow", "PolyFlow", "SafeFlow", 'RoSD', 'GaugeFlow'],
    'height_limit': 1.35,
    'vel_scale': 0.01,
    'height_min': None,
    'v_max': None,
    'v_min': None
}

# === 执行 ===
plot_2x5_compact(
    simple_config=simple_config,
    hard_config=hard_config,
    save_path="walker_dist_combined.pdf",
    select='all'
)