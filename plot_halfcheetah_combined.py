import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_halfcheetah(
    traj_path_list = [
        "outputs/halfcheetah/flow_sample_step100/42_2026-01-08_18-03-03/final_traj.npz",
        "outputs/halfcheetah/polyflow_sample/42_2026-01-08_18-05-28/final_traj.npz",
        "outputs/halfcheetah/safeflow_sample_step100/42_2026-01-10_15-38-47/final_traj.npz",
        "outputs/halfcheetah/RoS_sample_horizon160/42_2026-01-09_13-14-40/final_traj.npz",
        "outputs/halfcheetah/gaugeflow_train_step100/42_2026-01-09_15-56-16/final_traj.npz"
    ],
    label_list = ["Flow", "PolyFlow", "SafeFlow", 'RoSD', 'GaugeFlow'],
    leg_limit=1.2,
    torsion_limit=0.8,
    plot_horizon_length=100,
    save_path='halfcheetah_dist_combined.png'
):
    """
    绘制 HalfCheetah 动作空间分布及约束边界。
    """

    # --- 1. 配置绘图风格 (ICML 风格) ---
    plt.rcParams.update({
        "text.usetex": False,       # 如果系统没有安装 LaTeX，设为 False
        "font.family": "serif",     # ICML 常用衬线字体
        "font.serif": ["Times New Roman"], 
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
    })

    PALETTE = {
        'Flow': '#7f7f7f',       # 灰色
        'PolyFlow': '#d62728',   # 红色
        'SafeFlow': '#1f77b4',   # 蓝色
        'RoSD': '#2ca02c',       # 绿色
        'GaugeFlow': "#8b2ca0"   # 紫色
    }

    # --- 2. 加载数据 ---
    def get_u_traj_list(traj_path_list, label_list, plot_horizon_length):
        u_traj_list = []
        for idx, (filepath, label) in enumerate(zip(traj_path_list, label_list)):
            try:
                # 尝试加载数据，为了鲁棒性添加了 try-except
                npz_data = np.load(filepath, allow_pickle=True)
                if 'gene_act_traj' in npz_data:
                    act_traj = npz_data['gene_act_traj']
                else:
                    # 如果找不到键，生成随机数据用于测试/占位 (实际使用请确保路径正确)
                    print(f"Warning: Could not find action key in {filepath}, using random data.")
                    act_traj = np.random.uniform(-1, 1, (64, 1000, 6))

                horizon_length = min(act_traj.shape[1], plot_horizon_length)
                act_traj = act_traj[:, :horizon_length, :] # (Batch, Horizon, Dim)
                
                # 展平数据以便绘制散点 (N_points, Dim)
                flat_traj = act_traj.reshape(-1, act_traj.shape[-1])
                u_traj_list.append(flat_traj)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                # 发生错误时塞入空数组避免后续崩溃
                u_traj_list.append(np.zeros((0, 6)))

        return u_traj_list
    
    u_traj_list = get_u_traj_list(traj_path_list, label_list, plot_horizon_length)

    # --- 3. 绘图设置 ---
    num_methods = len(label_list)
    rows = 3
    cols = num_methods
    
    # 设置图像大小：宽度约为14-16英寸以适应双栏并在缩小后保持清晰，高度适中
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 7.5), constrained_layout=True)
    
    # 确保 axes 是二维数组
    if num_methods == 1:
        axes = axes[:, np.newaxis]

    # 定义每一行的绘制配置
    # (x_idx, y_idx, x_label, y_label, constraint_type, constraint_val)
    plot_configs = [
        (0, 1, r"$u_0$", r"$u_1$", "sum_lt", leg_limit),
        (3, 4, r"$u_3$", r"$u_4$", "sum_lt", leg_limit),
        (0, 3, r"$u_0$", r"$u_3$", "diff_abs_lt", torsion_limit)
    ]

    # 坐标轴范围 (略大于 [-1, 1])
    axis_limit = 1.15
    ticks = [-1.0, 0.0, 1.0]

    # --- 4. 辅助函数：绘制约束 ---
    def draw_constraint_region(ax, c_type, limit, x_range=np.linspace(-1.2, 1.2, 100)):
        """在背景中绘制不可行区域"""
        # 1. 绘制 [-1, 1] 的 Box 边界
        rect = patches.Rectangle((-1, -1), 2, 2, linewidth=1, edgecolor='black', facecolor='none', linestyle='-', zorder=10, alpha=0.3)
        ax.add_patch(rect)

        # 2. 绘制具体约束
        if c_type == "sum_lt": 
            # u_x + u_y < limit  =>  y < -x + limit
            y_line = -x_range + limit
            
            # 绘制边界线
            ax.plot(x_range, y_line, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # 填充不满足约束的区域 (右上方)
            # 实际上我们关心的是 Box 内的部分，填充 Box 之外或者 Box 内不可行的部分
            # 为了清晰，我们在 Box 内部填充淡红色表示不可行
            ax.fill_between(x_range, y_line, 2, where=(y_line < 2), color='red', alpha=0.1, zorder=0)

        elif c_type == "diff_abs_lt":
            # |x - y| < limit => -limit < x - y < limit
            # => y > x - limit  AND  y < x + limit
            
            y_upper = x_range + limit
            y_lower = x_range - limit
            
            ax.plot(x_range, y_upper, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.plot(x_range, y_lower, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # 填充不可行区域 (两条线之外)
            ax.fill_between(x_range, y_upper, 2, color='red', alpha=0.1, zorder=0)
            ax.fill_between(x_range, -2, y_lower, color='red', alpha=0.1, zorder=0)

    # --- 5. 主循环绘图 ---
    for row_idx, config in enumerate(plot_configs):
        u_x_idx, u_y_idx, xlabel, ylabel, c_type, c_val = config
        
        for col_idx, (method_name, data) in enumerate(zip(label_list, u_traj_list)):
            ax = axes[row_idx, col_idx]
            color = PALETTE.get(method_name, 'gray')
            
            # 5.1 绘制散点数据
            if len(data) > 0:
                # 使用较小的点和透明度来处理大量数据
                # 随机采样一部分点以防渲染过慢 (如果数据量>10k)
                plot_data = data
                if len(data) > 5000:
                    idx = np.random.choice(len(data), 5000, replace=False)
                    plot_data = data[idx]
                
                ax.scatter(plot_data[:, u_x_idx], plot_data[:, u_y_idx], 
                           s=2, c=color, alpha=0.7, edgecolors='none', rasterized=True) 
                           # rasterized=True 对矢量图导出很重要，减小文件体积

            # 5.2 绘制约束背景
            draw_constraint_region(ax, c_type, c_val)

            # 5.3 设置坐标轴属性
            ax.set_xlim(-axis_limit, axis_limit)
            ax.set_ylim(-axis_limit, axis_limit)
            ax.set_aspect('equal')
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            
            # 仅在最左侧显示 Y 轴标签
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=14, labelpad=0)
            else:
                ax.set_yticklabels([])
            
            # 仅在最底部显示 X 轴标签
            if row_idx == rows - 1:
                ax.set_xlabel(xlabel, fontsize=14)
            else:
                ax.set_xticklabels([])

            # 仅在第一行显示方法标题
            if row_idx == 0:
                ax.set_title(method_name, fontsize=14, fontweight='bold', pad=10)
            
            # 网格线 (可选，设为非常淡)
            ax.grid(True, linestyle=':', alpha=0.3)

    # --- 6. 保存与调整 ---
    # 调整子图间距 (constrained_layout 已经处理了大部分，这里微调)
    # plt.subplots_adjust(wspace=0.1, hspace=0.1) 
    
    print(f"Saving figure to {save_path}...")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done.")

# 如果是作为脚本直接运行
if __name__ == "__main__":

    plot_halfcheetah(
        save_path='halfcheetah_dist_combined.pdf'
    )