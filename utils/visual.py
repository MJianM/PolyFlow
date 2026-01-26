import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_simple_loss(loss_history, save_path=None):
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
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corridor_1 = [0.0, 3.2, 0.8, 1.2]  # [x_min, x_max, y_min, y_max]
    corridor_2 = [2.8, 3.2, 0.8, 4.5]  # [x_min, x_max, y_min, y_max]
    corridors = [corridor_1, corridor_2]
    
    for i, corridor in enumerate(corridors):
        x_min, x_max, y_min, y_max = corridor
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor='black', facecolor='none',
            linestyle='--', alpha=0.7, label=f'Corridor {i+1}' if i == 0 else ""
        )
        ax.add_patch(rect)
    
    true_traj = np.array(true_traj)
    gene_traj = np.array(gene_traj)
    
    if len(true_traj.shape) == 1:
        true_traj = true_traj.reshape(1, -1)
    if len(gene_traj.shape) == 1:
        gene_traj = gene_traj.reshape(1, -1)
    
    n_samples_true = true_traj.shape[0]
    n_samples_gene = gene_traj.shape[0]
    
    for i in range(n_samples_true):
        traj = true_traj[i]
        T = len(traj) // 2
        points = traj.reshape(T, 2)
        
        x = points[:, 0]
        y = points[:, 1]
        
        ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7, 
                label='True Trajectory' if i == 0 else "")
        ax.scatter(x, y, c='blue', s=20, alpha=0.6)
        
        ax.scatter(x[0], y[0], c='green', s=100, marker='o', 
                  edgecolors='black', linewidth=2, label='Start' if i == 0 else "")
        ax.scatter(x[-1], y[-1], c='red', s=100, marker='s', 
                  edgecolors='black', linewidth=2, label='End' if i == 0 else "")
    
    for i in range(n_samples_gene):
        traj = gene_traj[i]
        T = len(traj) // 2
        points = traj.reshape(T, 2)
        
        x = points[:, 0]
        y = points[:, 1]
        
        ax.plot(x, y, 'r-', linewidth=1.5, alpha=0.7,
                label='Generated Trajectory' if i == 0 else "")
        ax.scatter(x, y, c='red', s=20, alpha=0.6)
        
        ax.scatter(x[0], y[0], c='green', s=100, marker='o', 
                  edgecolors='black', linewidth=2)
        ax.scatter(x[-1], y[-1], c='red', s=100, marker='s', 
                  edgecolors='black', linewidth=2)
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Trajectory Comparison in L-shaped Corridor', fontsize=14, fontweight='bold')
    
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(-0.5, 5.0)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
    
    ax.set_aspect('equal', adjustable='box')
    
    plt.figtext(0.02, 0.98, f'True samples: {n_samples_true}', 
                fontsize=10, verticalalignment='top')
    plt.figtext(0.02, 0.95, f'Generated samples: {n_samples_gene}', 
                fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved as {fig_name}")


# def plot_trajectory_comparison(env_maze, true_trajs, gene_trajs, ellips_list=None, max_plot=100, save_path=None):
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     try:
#         rows, cols = env_maze.map_length, env_maze.map_width
#     except AttributeError:
#         rows, cols = env_maze.maze_map.shape
        
#     scale = env_maze.maze_size_scaling

#     for r in range(rows):
#         for c in range(cols):
#             if env_maze.maze_map[r][c] == 1: # 1 is wall
#                 center_xy = env_maze.cell_rowcol_to_xy((r, c))
#                 patch = patches.Rectangle(
#                     (center_xy[0] - scale/2, center_xy[1] - scale/2), 
#                     scale, scale, 
#                     linewidth=0, edgecolor=None, facecolor='#333333',
#                     zorder=1
#                 )
#                 ax.add_patch(patch)

#     if ellips_list is not None:
#         label_added = False
        
#         for obs in ellips_list:
#             xc, yc, a, b = obs
            
#             lbl = 'CBF Obstacle' if not label_added else None
            
#             ellipse = patches.Ellipse(
#                 xy=(xc, yc), 
#                 width=a * 2, 
#                 height=b * 2, 
#                 angle=0, 
#                 facecolor='magenta', 
#                 edgecolor='purple', 
#                 alpha=0.3,       
#                 linewidth=2,
#                 linestyle='-',
#                 zorder=2,       
#                 label=lbl
#             )
#             ax.add_patch(ellipse)
#             label_added = True

#     n_true = min(len(true_trajs), max_plot)
#     n_gene = min(len(gene_trajs), max_plot)
    
#     plot_true = true_trajs[:n_true]
#     plot_gene = gene_trajs[:n_gene]
    
#     ax.plot(plot_true[0, :, 0], plot_true[0, :, 1], 
#             color='royalblue', linewidth=2, alpha=0.4, label='Ground Truth', zorder=3)
    
#     for i in range(1, n_true):
#         ax.plot(plot_true[i, :, 0], plot_true[i, :, 1], 
#                 color='royalblue', linewidth=2, alpha=0.4, zorder=3)

#     ax.plot(plot_gene[0, :, 0], plot_gene[0, :, 1], 
#             color='darkorange', linewidth=1.5, alpha=0.6, linestyle='--', label='Generated', zorder=4)
            
#     for i in range(1, n_gene):
#         ax.plot(plot_gene[i, :, 0], plot_gene[i, :, 1], 
#                 color='darkorange', linewidth=1.5, alpha=0.6, linestyle='--', zorder=4)


#     start_points = plot_gene[:, 0, :]
#     end_points = plot_gene[:, -1, :]
    
#     ax.scatter(start_points[:, 0], start_points[:, 1], c='lime', s=15, 
#                zorder=10, alpha=0.8, edgecolors='black', linewidth=0.5, label='Gen Start')
#     ax.scatter(end_points[:, 0], end_points[:, 1], c='red', s=20, marker='x', 
#                zorder=10, alpha=0.8, linewidth=1, label='Gen End')

#     ax.set_aspect('equal')
#     ax.set_xlabel("X Position")
#     ax.set_ylabel("Y Position")
    
#     title_str = f"True (N={n_true}) vs Generated (N={n_gene})"
#     if ellips_list:
#         title_str += " with CBF Obstacles"
#     ax.set_title(title_str)
    
#     ax.legend(loc='upper right', framealpha=0.9, fontsize='small')
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150)
#         print(f"Comparison plot saved to {save_path}")
#     else:
#         plt.show()

def get_superellipse_points(xc, yc, a, b, n, num_points=200):
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # x = a * sgn(cos(t)) * |cos(t)|^(2/n)
    # y = b * sgn(sin(t)) * |sin(t)|^(2/n)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    x = xc + a * np.sign(cos_t) * (np.abs(cos_t)) ** (2 / n)
    y = yc + b * np.sign(sin_t) * (np.abs(sin_t)) ** (2 / n)
    
    return np.column_stack([x, y])



def plot_trajectory_comparison(env_maze, true_trajs, gene_trajs, obs_expand_dis=0.2, ellips_list=None, max_plot=100, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        rows, cols = env_maze.map_length, env_maze.map_width
    except AttributeError:
        rows, cols = env_maze.maze_map.shape
    scale = env_maze.maze_size_scaling

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


    if ellips_list is not None:
        label_added = False
        for obs in ellips_list:
            xc, yc, a, b, n = obs

            lbl = 'CBF Obstacle' if not label_added else None
            
            # ellipse = patches.Ellipse(
            #     xy=(xc, yc), 
            #     width=a * 2, height=b * 2, # Matplotlib 
            #     angle=0, 
            #     facecolor='magenta', edgecolor='purple', 
            #     alpha=0.5, linewidth=2, linestyle='-', zorder=2,
            #     label=lbl
            # )

            points = get_superellipse_points(xc, yc, a, b, n)
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

    n_true = min(len(true_trajs), max_plot)
    n_gene = min(len(gene_trajs), max_plot)
    
    plot_true = true_trajs[:n_true]
    plot_gene = gene_trajs[:n_gene]


    ax.plot(plot_true[0, :, 0], plot_true[0, :, 1], 
            color='royalblue', linewidth=2, alpha=0.3, zorder=3, label='Ground Truth (Line)')
    for i in range(1, n_true):
        ax.plot(plot_true[i, :, 0], plot_true[i, :, 1], 
                color='royalblue', linewidth=2, alpha=0.3, zorder=3)

    flat_true = plot_true.reshape(-1, 2)
    ax.scatter(flat_true[:, 0], flat_true[:, 1], 
               c='royalblue', s=10, alpha=0.3, zorder=3, marker='.', label='Ground Truth (Points)')

    ax.plot(plot_gene[0, :, 0], plot_gene[0, :, 1], 
            color='darkorange', linewidth=1.5, alpha=0.6, linestyle='--', zorder=4, label='Generated (Line)')
    for i in range(1, n_gene):
        ax.plot(plot_gene[i, :, 0], plot_gene[i, :, 1], 
                color='darkorange', linewidth=1.5, alpha=0.6, linestyle='--', zorder=4)

    flat_gene = plot_gene.reshape(-1, 2)
    ax.scatter(flat_gene[:, 0], flat_gene[:, 1], 
               c='darkorange', s=15, alpha=0.6, zorder=4, marker='.', label='Generated (Points)')

    start_points = plot_gene[:, 0, :]
    end_points = plot_gene[:, -1, :]
    
    ax.scatter(start_points[:, 0], start_points[:, 1], 
               c='lime', s=30, zorder=10, edgecolors='black', linewidth=0.5, label='Gen Start')
    ax.scatter(end_points[:, 0], end_points[:, 1], 
               c='red', s=40, marker='x', zorder=10, linewidth=1.5, label='Gen End')

    ax.set_aspect('equal')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    title_str = f"True (N={n_true}) vs Generated (N={n_gene})"
    if ellips_list:
        title_str += " with CBF Obstacles"
    ax.set_title(title_str)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9, fontsize='small')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()