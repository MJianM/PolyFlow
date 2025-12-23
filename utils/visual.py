import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_simple_loss(loss_history, save_path=None):
    """最简单的loss曲线绘制函数"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Simple loss curve saved to {save_path}")
    

def plot_results(true_traj, gene_traj, fig_name='test.png'):
    """
    绘制真实轨迹与生成轨迹对比图
    
    :param true_traj: [n_samples, 2*T] 真实轨迹，排列为[x1,y1,x2,y2,...]
    :param gene_traj: [n_samples, 2*T] 生成轨迹，排列为[x1,y1,x2,y2,...]
    :param fig_name: str 保存的图片文件名
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # === 1. 绘制L型走廊边界 ===
    corridor_1 = [0.0, 3.2, 0.8, 1.2]  # [x_min, x_max, y_min, y_max]
    corridor_2 = [2.8, 3.2, 0.8, 4.5]  # [x_min, x_max, y_min, y_max]
    corridors = [corridor_1, corridor_2]
    
    # 绘制走廊边界
    for i, corridor in enumerate(corridors):
        x_min, x_max, y_min, y_max = corridor
        width = x_max - x_min
        height = y_max - y_min
        
        # 添加矩形边框
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='black', facecolor='none',
            linestyle='--', alpha=0.7, label=f'Corridor {i+1}' if i == 0 else ""
        )
        ax.add_patch(rect)
    
    # === 2. 绘制轨迹 ===
    # 确保输入是numpy数组
    true_traj = np.array(true_traj)
    gene_traj = np.array(gene_traj)
    
    # 检查维度
    if len(true_traj.shape) == 1:
        true_traj = true_traj.reshape(1, -1)
    if len(gene_traj.shape) == 1:
        gene_traj = gene_traj.reshape(1, -1)
    
    n_samples_true = true_traj.shape[0]
    n_samples_gene = gene_traj.shape[0]
    
    # 绘制真实轨迹
    for i in range(n_samples_true):
        traj = true_traj[i]
        # 重塑为(T, 2)格式
        T = len(traj) // 2
        points = traj.reshape(T, 2)
        
        # 提取x,y坐标
        x = points[:, 0]
        y = points[:, 1]
        
        # 绘制轨迹线和散点
        ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7, 
                label='True Trajectory' if i == 0 else "")
        ax.scatter(x, y, c='blue', s=20, alpha=0.6)
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], c='green', s=100, marker='o', 
                  edgecolors='black', linewidth=2, label='Start' if i == 0 else "")
        ax.scatter(x[-1], y[-1], c='red', s=100, marker='s', 
                  edgecolors='black', linewidth=2, label='End' if i == 0 else "")
    
    # 绘制生成轨迹
    for i in range(n_samples_gene):
        traj = gene_traj[i]
        # 重塑为(T, 2)格式
        T = len(traj) // 2
        points = traj.reshape(T, 2)
        
        # 提取x,y坐标
        x = points[:, 0]
        y = points[:, 1]
        
        # 绘制轨迹线和散点
        ax.plot(x, y, 'r-', linewidth=1.5, alpha=0.7,
                label='Generated Trajectory' if i == 0 else "")
        ax.scatter(x, y, c='red', s=20, alpha=0.6)
        
        # 标记起点和终点
        ax.scatter(x[0], y[0], c='green', s=100, marker='o', 
                  edgecolors='black', linewidth=2)
        ax.scatter(x[-1], y[-1], c='red', s=100, marker='s', 
                  edgecolors='black', linewidth=2)
    
    # === 3. 图形美化 ===
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Trajectory Comparison in L-shaped Corridor', fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(-0.5, 5.0)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    # 去重
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
    
    # 设置坐标轴比例相等
    ax.set_aspect('equal', adjustable='box')
    
    # 添加文本说明
    plt.figtext(0.02, 0.98, f'True samples: {n_samples_true}', 
                fontsize=10, verticalalignment='top')
    plt.figtext(0.02, 0.95, f'Generated samples: {n_samples_gene}', 
                fontsize=10, verticalalignment='top')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved as {fig_name}")


# def plot_trajectory_comparison(env_maze, true_trajs, gene_trajs, ellips_list=None, max_plot=100, save_path=None):
#     """
#     pointmaze环境，可视化对比真实轨迹和生成轨迹的分布，并绘制椭圆障碍物。
    
#     :param env_maze: gymnasium_robotics 的 maze 对象 (env.unwrapped.maze)
#     :param true_trajs: (n_samples, seq_length, 2) 真实专家/数据集轨迹
#     :param gene_trajs: (n_samples, seq_length, 2) 模型生成的轨迹
#     :param ellips_list: List[tuple], 每个元素为 (x_c, y_c, a, b)，表示椭圆中心和半轴长
#     :param max_plot: 为了绘图速度和清晰度，最大绘制的轨迹条数。默认为 100。
#     :param save_path: 如果提供路径，则保存图片；否则 plt.show()
#     """
    
#     # 创建画布
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     # 获取迷宫尺寸信息
#     try:
#         rows, cols = env_maze.map_length, env_maze.map_width
#     except AttributeError:
#         # 兼容旧版本或不同实现的属性名
#         rows, cols = env_maze.maze_map.shape
        
#     scale = env_maze.maze_size_scaling

#     # ==========================
#     # 1. 绘制迷宫墙壁 (Background)
#     # ==========================
#     for r in range(rows):
#         for c in range(cols):
#             if env_maze.maze_map[r][c] == 1: # 1 is wall
#                 center_xy = env_maze.cell_rowcol_to_xy((r, c))
#                 patch = patches.Rectangle(
#                     (center_xy[0] - scale/2, center_xy[1] - scale/2), 
#                     scale, scale, 
#                     linewidth=0, edgecolor=None, facecolor='#333333', # 深灰色墙壁
#                     zorder=1
#                 )
#                 ax.add_patch(patch)

#     # ==========================
#     # 2. 绘制椭圆障碍物 (Obstacles) - 新增部分
#     # ==========================
#     if ellips_list is not None:
#         # 标记 flag，防止图例中出现重复的 label
#         label_added = False
        
#         for obs in ellips_list:
#             xc, yc, a, b = obs
            
#             # 只有第一个椭圆才添加 label，避免 legend 里全是重复项
#             lbl = 'CBF Obstacle' if not label_added else None
            
#             # 注意：patches.Ellipse 接受的是 width 和 height (全长)
#             # 而输入的 a, b 通常是半轴长，所以需要 * 2
#             ellipse = patches.Ellipse(
#                 xy=(xc, yc), 
#                 width=a * 2, 
#                 height=b * 2, 
#                 angle=0, 
#                 facecolor='magenta', 
#                 edgecolor='purple', 
#                 alpha=0.3,       # 半透明，以便看到被遮挡的轨迹（如果有）
#                 linewidth=2,
#                 linestyle='-',
#                 zorder=2,        # 图层顺序：在墙壁之上，轨迹之下(或之上，视需求而定)
#                 label=lbl
#             )
#             ax.add_patch(ellipse)
#             label_added = True

#     # ==========================
#     # 3. 绘制轨迹 (Trajectories)
#     # ==========================
    
#     n_true = min(len(true_trajs), max_plot)
#     n_gene = min(len(gene_trajs), max_plot)
    
#     plot_true = true_trajs[:n_true]
#     plot_gene = gene_trajs[:n_gene]
    
#     # --- A. 绘制真实轨迹 (Ground Truth) ---
#     ax.plot(plot_true[0, :, 0], plot_true[0, :, 1], 
#             color='royalblue', linewidth=2, alpha=0.4, label='Ground Truth', zorder=3)
    
#     for i in range(1, n_true):
#         ax.plot(plot_true[i, :, 0], plot_true[i, :, 1], 
#                 color='royalblue', linewidth=2, alpha=0.4, zorder=3)

#     # --- B. 绘制生成轨迹 (Generated) ---
#     ax.plot(plot_gene[0, :, 0], plot_gene[0, :, 1], 
#             color='darkorange', linewidth=1.5, alpha=0.6, linestyle='--', label='Generated', zorder=4)
            
#     for i in range(1, n_gene):
#         ax.plot(plot_gene[i, :, 0], plot_gene[i, :, 1], 
#                 color='darkorange', linewidth=1.5, alpha=0.6, linestyle='--', zorder=4)

#     # ==========================
#     # 4. 绘制端点分布
#     # ==========================
#     start_points = plot_gene[:, 0, :]
#     end_points = plot_gene[:, -1, :]
    
#     ax.scatter(start_points[:, 0], start_points[:, 1], c='lime', s=15, 
#                zorder=10, alpha=0.8, edgecolors='black', linewidth=0.5, label='Gen Start')
#     ax.scatter(end_points[:, 0], end_points[:, 1], c='red', s=20, marker='x', 
#                zorder=10, alpha=0.8, linewidth=1, label='Gen End')

#     # ==========================
#     # 5. 样式调整
#     # ==========================
#     ax.set_aspect('equal')
#     ax.set_xlabel("X Position")
#     ax.set_ylabel("Y Position")
    
#     title_str = f"True (N={n_true}) vs Generated (N={n_gene})"
#     if ellips_list:
#         title_str += " with CBF Obstacles"
#     ax.set_title(title_str)
    
#     # 自动调整图例位置
#     ax.legend(loc='upper right', framealpha=0.9, fontsize='small')
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150)
#         print(f"Comparison plot saved to {save_path}")
#     else:
#         plt.show()

def get_superellipse_points(xc, yc, a, b, n, num_points=200):
    """
    生成超椭圆的 (x, y) 坐标点
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # 超椭圆参数方程:
    # x = a * sgn(cos(t)) * |cos(t)|^(2/n)
    # y = b * sgn(sin(t)) * |sin(t)|^(2/n)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    x = xc + a * np.sign(cos_t) * (np.abs(cos_t)) ** (2 / n)
    y = yc + b * np.sign(sin_t) * (np.abs(sin_t)) ** (2 / n)
    
    return np.column_stack([x, y])



def plot_trajectory_comparison(env_maze, true_trajs, gene_trajs, obs_expand_dis=0.2, ellips_list=None, max_plot=100, save_path=None):
    """
    可视化对比真实轨迹和生成轨迹，包含：
    1. 迷宫墙壁背景
    2. 椭圆障碍物 (CBF)
    3. 轨迹线条 (Line)
    4. 轨迹上的离散点 (Scatter/Waypoints) - [新增]
    5. 起点和终点高亮
    
    :param env_maze: gymnasium_robotics 的 maze 对象
    :param true_trajs: (n_samples, seq_length, 2) 真实专家轨迹
    :param gene_trajs: (n_samples, seq_length, 2) 生成轨迹
    :param ellips_list: List[tuple], 元素为 (x_c, y_c, a, b)
    :param max_plot: 最大绘制条数
    :param save_path: 保存路径
    """
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 获取迷宫尺寸
    try:
        rows, cols = env_maze.map_length, env_maze.map_width
    except AttributeError:
        rows, cols = env_maze.maze_map.shape
    scale = env_maze.maze_size_scaling

    # ==========================
    # 1. 绘制迷宫墙壁 (Background)
    # ==========================
    for r in range(rows):
        for c in range(cols):
            if env_maze.maze_map[r][c] == 1:
                center_xy = env_maze.cell_rowcol_to_xy((r, c))
                patch = patches.Rectangle(
                    (center_xy[0] - scale/2 - obs_expand_dis, center_xy[1] - scale/2 - obs_expand_dis), 
                    scale + 2*obs_expand_dis, scale + 2*obs_expand_dis, 
                    linewidth=0, facecolor='#333333', zorder=1
                )
                ax.add_patch(patch)

    # ==========================
    # 2. 绘制椭圆障碍物 (Obstacles)
    # ==========================
    if ellips_list is not None:
        label_added = False
        for obs in ellips_list:
            xc, yc, a, b, n = obs
            # 仅给第一个障碍物加标签，避免图例重复
            lbl = 'CBF Obstacle' if not label_added else None
            
            # ellipse = patches.Ellipse(
            #     xy=(xc, yc), 
            #     width=a * 2, height=b * 2, # Matplotlib 使用直径
            #     angle=0, 
            #     facecolor='magenta', edgecolor='purple', 
            #     alpha=0.5, linewidth=2, linestyle='-', zorder=2,
            #     label=lbl
            # )

            # 1. 生成点
            points = get_superellipse_points(xc, yc, a, b, n)
            # 2. 创建 Polygon (替代 Ellipse)
            super_ellipse = patches.Polygon(
                points,
                closed=True,
                facecolor='magenta', 
                edgecolor='purple',
                alpha=0.5, 
                linewidth=2, 
                linestyle='-', 
                zorder=2,
                label=lbl
            )
            ax.add_patch(super_ellipse)
            label_added = True

    # ==========================
    # 3. 准备数据
    # ==========================
    n_true = min(len(true_trajs), max_plot)
    n_gene = min(len(gene_trajs), max_plot)
    
    plot_true = true_trajs[:n_true]
    plot_gene = gene_trajs[:n_gene]

    # ==========================
    # 4. 绘制真实轨迹 (Ground Truth)
    # ==========================
    # 4.1 绘制线条 (Loop for lines)
    ax.plot(plot_true[0, :, 0], plot_true[0, :, 1], 
            color='royalblue', linewidth=2, alpha=0.3, zorder=3, label='Ground Truth (Line)')
    for i in range(1, n_true):
        ax.plot(plot_true[i, :, 0], plot_true[i, :, 1], 
                color='royalblue', linewidth=2, alpha=0.3, zorder=3)
    
    # 4.2 绘制轨迹点 (Batch scatter for points) - [新增部分]
    # 将所有真实轨迹的点展平，一次性 scatter，提高效率
    flat_true = plot_true.reshape(-1, 2)
    ax.scatter(flat_true[:, 0], flat_true[:, 1], 
               c='royalblue', s=10, alpha=0.3, zorder=3, marker='.', label='Ground Truth (Points)')

    # ==========================
    # 5. 绘制生成轨迹 (Generated)
    # ==========================
    # 5.1 绘制线条
    ax.plot(plot_gene[0, :, 0], plot_gene[0, :, 1], 
            color='darkorange', linewidth=1.5, alpha=0.6, linestyle='--', zorder=4, label='Generated (Line)')
    for i in range(1, n_gene):
        ax.plot(plot_gene[i, :, 0], plot_gene[i, :, 1], 
                color='darkorange', linewidth=1.5, alpha=0.6, linestyle='--', zorder=4)

    # 5.2 绘制轨迹点 - [新增部分]
    flat_gene = plot_gene.reshape(-1, 2)
    ax.scatter(flat_gene[:, 0], flat_gene[:, 1], 
               c='darkorange', s=15, alpha=0.6, zorder=4, marker='.', label='Generated (Points)')

    # ==========================
    # 6. 绘制端点 (Start/End) - 高亮
    # ==========================
    start_points = plot_gene[:, 0, :]
    end_points = plot_gene[:, -1, :]
    
    # 起点：绿色圆点
    ax.scatter(start_points[:, 0], start_points[:, 1], 
               c='lime', s=30, zorder=10, edgecolors='black', linewidth=0.5, label='Gen Start')
    # 终点：红色X
    ax.scatter(end_points[:, 0], end_points[:, 1], 
               c='red', s=40, marker='x', zorder=10, linewidth=1.5, label='Gen End')

    # ==========================
    # 7. 样式与图例
    # ==========================
    ax.set_aspect('equal')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    title_str = f"True (N={n_true}) vs Generated (N={n_gene})"
    if ellips_list:
        title_str += " with CBF Obstacles"
    ax.set_title(title_str)
    
    # 图例设置：去重，防止 scatter 和 plot 重复占位
    # 这里通过 dict 技巧自动去重 label
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9, fontsize='small')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()