import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from env.maze2d_env import Maze2DEnv
from utils.visual import get_superellipse_points



def plot_maze2d_environment(
        env: Maze2DEnv,
        file_path_list: list = None,
        label_list: list = None,
        max_plot: int = 5,
        save_path: str = "method_compare.pdf"
    ):

    if file_path_list is None or label_list is None:
        raise ValueError("file_path_list and label_list must not be None")
    

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "pdf.fonttype": 42, 
        "ps.fonttype": 42
    })

    num_methods = len(file_path_list)
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.0))
    axes_flat = axes.flatten() 

    rows, cols = env.env_maze.map_length, env.env_maze.map_width
    scale = env.env_maze.maze_size_scaling
    ellips_list_to_plot = env.maze_obs.get_ellips_list()

    def get_unsafe_scores(trajs):
        scores = []
        for traj in trajs:
            unsafe_steps = 0
            for t in range(traj.shape[0]):
                x, y = traj[t]

                in_any_rect = False
                for (x_min, x_max, y_min, y_max) in env.valid_rect_bounds:
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        in_any_rect = True
                        break
                if not in_any_rect:
                    unsafe_steps += 1
            scores.append(unsafe_steps)
        return np.array(scores)

    for i in range(4): 
        ax = axes_flat[i]
        
        if i >= num_methods:
            ax.axis('off')
            continue
            
        path = file_path_list[i]
        label = label_list[i]

        try:
            data = np.load(path)
            if label_list[i].lower() == 'dataset':
                gen_trajs_all = data['true_traj'] 
            else:
                gen_trajs_all = data['generated_traj'] 
        except Exception as e:
            print(f"Error loading {path}: {e}")
            ax.set_title(f"{label} (Load Error)")
            continue

        for r in range(rows):
            for c in range(cols):
                if env.env_maze.maze_map[r][c] == 1:
                    center_xy = env.env_maze.cell_rowcol_to_xy((r, c))
                    patch = patches.Rectangle(
                        (center_xy[0] - scale/2 - env.obs_expand_dis, 
                         center_xy[1] - scale/2 - env.obs_expand_dis), 
                        scale + 2*env.obs_expand_dis, 
                        scale + 2*env.obs_expand_dis, 
                        linewidth=0, facecolor='#404040', zorder=1
                    )
                    ax.add_patch(patch)

        if 'safe' in label.lower() or 'RoS' in label or 'ReS' in label or 'TVS' in label:
            for obs in ellips_list_to_plot:
                xc, yc, a, b, n = obs
                points = get_superellipse_points(xc, yc, a, b, n)
                super_ellipse = patches.Polygon(
                    points, closed=True, facecolor='magenta', edgecolor='purple',
                    alpha=0.3, linewidth=1, zorder=2, label='Barrier' if obs is ellips_list_to_plot[0] else None
                )
                ax.add_patch(super_ellipse)

        trajs_to_plot = []
        if len(gen_trajs_all) > 0:
            if 'poly-flow' in label.lower() or 'ours' in label.lower():
                n_gen = min(len(gen_trajs_all), max_plot)
                trajs_to_plot = gen_trajs_all[:n_gen]
            else:
                scores = get_unsafe_scores(gen_trajs_all)
                sorted_indices = np.argsort(scores)[::-1]
                top_indices = sorted_indices[:max_plot]
                trajs_to_plot = gen_trajs_all[top_indices]

        if ('poly-flow' in label.lower() or 'ours' in label.lower()) and len(trajs_to_plot) > 0:
            active_rect_indices = set()
            
            for traj in trajs_to_plot:
                for point in traj:
                    x, y = point
                    for idx, (x_min, x_max, y_min, y_max) in enumerate(env.valid_rect_bounds):
                        if x_min <= x <= x_max and y_min <= y <= y_max:
                            active_rect_indices.add(idx)

            
            rect_drawn_flag = False
            for idx in active_rect_indices:
                x_min, x_max, y_min, y_max = env.valid_rect_bounds[idx]
                width = x_max - x_min
                height = y_max - y_min
                
                rect_patch = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=1.2, edgecolor='red', facecolor='none', 
                    linestyle='--', zorder=3,
                    label='Valid Polytope' if not rect_drawn_flag else None
                )
                ax.add_patch(rect_patch)
                rect_drawn_flag = True

        if len(trajs_to_plot) > 0:
            for traj in trajs_to_plot:
                ax.plot(traj[:, 0], traj[:, 1], 
                        color='darkorange', linewidth=0.8, alpha=0.6, linestyle='-', zorder=4)
            
            flat_gen = trajs_to_plot.reshape(-1, 2)
            ax.scatter(flat_gen[:, 0], flat_gen[:, 1],
                       c='darkorange', s=2, alpha=0.9, zorder=4, marker='o') 

            start_pts = trajs_to_plot[:, 0, :]
            end_pts = trajs_to_plot[:, -1, :]
            ax.scatter(start_pts[:, 0], start_pts[:, 1], c='lime', s=25, zorder=5, edgecolors='black', linewidth=0.5, label='Start')
            ax.scatter(end_pts[:, 0], end_pts[:, 1], c='red', s=30, marker='X', zorder=5, linewidth=1, edgecolors='black', label='Goal')

        ax.set_aspect('equal')
        ax.set_title(label, fontweight='bold', pad=2)
        

        if i >= 2: 
            ax.set_xlabel("X Position")
        if i % 2 == 0: 
            ax.set_ylabel("Y Position")

        handles, lbls = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(lbls, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.95, borderpad=0.3)

    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.2)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()




if __name__ == "__main__":

    maze2d_env = Maze2DEnv()
    # plot_maze2d_environment(
    #     env=maze2d_env,
    #     file_path_list=[
    #         "outputs/maze2d/flow_sample/42_2025-12-24_19-54-43/sampled_traj.npz",
    #         "outputs/maze2d/oneray_sample/42_2025-12-24_19-55-47/sampled_traj.npz",
    #         "outputs/maze2d/flowtrunc_sample/42_2025-12-24_19-55-17/sampled_traj.npz",
    #         "outputs/maze2d/safeflow_sample/42_2025-12-24_19-41-10/sampled_traj.npz"
    #     ],
    #     label_list=["flow", "poly-flow(ours)", "flow-trunc", "safeflow"],
    #     max_plot=3,
    #     save_path="maze2d_method_comparison.pdf"
    # )

    plot_maze2d_environment(
        env=maze2d_env,
        file_path_list=[
            "outputs/maze2d/oneray_sample/42_2025-12-24_19-55-47/sampled_traj.npz",
            "outputs/maze2d/flow_sample/42_2025-12-24_19-54-43/sampled_traj.npz",
            "outputs/maze2d/oneray_sample/42_2025-12-24_19-55-47/sampled_traj.npz",
            "outputs/maze2d/flowtrunc_sample/42_2025-12-24_19-55-17/sampled_traj.npz"
        ],
        label_list=["Dataset", "Flow", "PolyFlow(Ours)", "FlowTrunc"],
        max_plot=3,
        save_path="maze2d_method_comparison_group1.pdf"
    )

    plot_maze2d_environment(
        env=maze2d_env,
        file_path_list=[
            "outputs/maze2d/safeflow_sample/42_2025-12-24_19-41-10/sampled_traj.npz",
            "outputs/maze2d/RoS_sample/42_2025-12-25_20-32-45/sampled_traj.npz",
            "outputs/maze2d/ReS_sample/42_2025-12-25_22-37-42/sampled_traj.npz",
            "outputs/maze2d/TVS_sample/42_2025-12-25_22-52-15/sampled_traj.npz"
        ],
        label_list=["SafeFlow", "RoSD", "ReSD", "TVSD"],
        max_plot=3,
        save_path="maze2d_method_comparison_group2.pdf"
    )