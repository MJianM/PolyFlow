from typing import List
from collections import namedtuple
import numpy as np
import torch
import math
import gymnasium as gym
import gymnasium_robotics

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from utils.visual import get_superellipse_points

LARGE_MAZE_EMPTY = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

LARGE_MAZE =   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 'g', 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 'r', 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

#  (r_min, c_min, r_max, c_max) 
Rectangle = namedtuple('Rectangle', ['r_min', 'c_min', 'r_max', 'c_max'])

class MazeObs:
    def __init__(self, 
                 maze, 
                 rect_list: List[Rectangle],
                 obs_expand_dis = 0.2,
                 ellips_n = 4,
                 alpha: float = 0.5):
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
            p_min = self.maze.cell_rowcol_to_xy(np.array([rect.r_min, rect.c_min]))
            p_max = self.maze.cell_rowcol_to_xy(np.array([rect.r_max, rect.c_max]))

            half_scale = self.maze.maze_size_scaling * 0.5
            x_min, x_max = p_min[0] - half_scale - self.obs_expand_dis, p_max[0] + half_scale + self.obs_expand_dis
            y_min, y_max = p_max[1] - half_scale - self.obs_expand_dis, p_min[1] + half_scale + self.obs_expand_dis

            x_center = (x_min + x_max) * 0.5
            y_center = (y_min + y_max) * 0.5

            x_length = x_max - x_min
            y_length = y_max - y_min

            a_in = x_length * 0.5
            b_in = y_length * 0.5
            a_out = x_length * 0.5 * math.pow(2, 1.0 / self.ellips_n)
            b_out = y_length * 0.5 * math.pow(2, 1.0 / self.ellips_n)
            a = a_in + (a_out - a_in) * self.alpha
            b = b_in + (b_out - b_in) * self.alpha

            ellips_list.append([x_center, y_center, a, b, self.ellips_n])

        return ellips_list

    def get_ellips_list(self):

        return self.ellips_list


class Maze2DEnv:

    def __init__(self, maze_map=LARGE_MAZE, obs_expand_dis=0.2, ellips_n=4, alpha=0.5):
        self.maze_map = maze_map

        self.env = gym.make('PointMaze_Large-v3', maze_map=maze_map, continuing_task=False, reset_target=False, max_episode_steps=1000,
                            render_mode='rgb_array')
        self.env_maze = self.env.unwrapped.maze
        self.rows, self.cols = self.env_maze.map_length, self.env_maze.map_width
        
        self.obs_expand_dis = obs_expand_dis

        self.binary_map = self._create_binary_map()

        self.maximal_rects = self._find_maximal_rectangles()
        self.valid_rect_bounds = self._find_valid_rect_bounds()

        obs_rect_list = [
            Rectangle(2, 2, 2, 3),
            Rectangle(1, 5, 2, 5),
            Rectangle(4, 4, 4, 5),
            Rectangle(5, 5, 6, 5),
            Rectangle(3, 7, 4, 7),
            Rectangle(4, 8, 4, 9),
            Rectangle(6, 7, 7, 7),
            Rectangle(6, 9, 6, 10),
        ]
        self.maze_obs = MazeObs(self.env_maze, obs_rect_list, obs_expand_dis=self.obs_expand_dis, ellips_n=ellips_n, alpha=alpha)

    def Shield(self, x, x_new, t):
        """
        :param x: (batch_size, seq_length, x_dim) 
        :param x_new: (batch_size, seq_length, x_dim)
        :param t: (batch_size,)
        
        return:
        - x_proj: (batch_size, seq_length, x_dim) 
        """

        if not hasattr(self, '_rect_bounds_tensor'):
            # valid_rect_bounds (x_min, x_max, y_min, y_max)
            self._rect_bounds_tensor = torch.tensor(
                self.valid_rect_bounds, 
                dtype=x_new.dtype, 
                device=x_new.device
            )
        
        if self._rect_bounds_tensor.device != x_new.device:
            self._rect_bounds_tensor = self._rect_bounds_tensor.to(x_new.device)

        rects = self._rect_bounds_tensor  # shape: (K, 4)

        # x_new shape: (B, S, 2) -> (B, S, 1, 2)
        pts_expanded = x_new.unsqueeze(-2) 

        # rects shape: (K, 4) ->  (1, 1, K)
        r_x_min = rects[:, 0].view(1, 1, -1)
        r_x_max = rects[:, 1].view(1, 1, -1)
        r_y_min = rects[:, 2].view(1, 1, -1)
        r_y_max = rects[:, 3].view(1, 1, -1)

        # pts_x/y shape: (B, S, 1)
        pts_x = pts_expanded[..., 0]
        pts_y = pts_expanded[..., 1]

        # clamped_x/y shape: (B, S, K)
        clamped_x = torch.clamp(pts_x, min=r_x_min, max=r_x_max)
        clamped_y = torch.clamp(pts_y, min=r_y_min, max=r_y_max)

        # shape: (B, S, K)
        diff_x = pts_x - clamped_x
        diff_y = pts_y - clamped_y
        dist_sq = diff_x**2 + diff_y**2

        # min_indices shape: (B, S)
        min_indices = torch.argmin(dist_sq, dim=-1)

        gather_indices = min_indices.unsqueeze(-1) # shape: (B, S, 1)

        best_x = torch.gather(clamped_x, 2, gather_indices).squeeze(-1)
        best_y = torch.gather(clamped_y, 2, gather_indices).squeeze(-1)

        # x_proj shape: (B, S, 2)
        x_proj = torch.stack([best_x, best_y], dim=-1)

        return x_proj


    def _create_binary_map(self):
        binary_map = np.zeros((self.rows, self.cols), dtype=np.int8)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze_map[r][c] == 1:
                    binary_map[r, c] = 1
                else:
                    binary_map[r, c] = 0
        return binary_map

    def _find_maximal_rectangles(self):
        rects = []
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                if self.binary_map[r1, c1] == 1: continue
                
                for r2 in range(r1, self.rows):
                    for c2 in range(c1, self.cols):
                        sub_map = self.binary_map[r1:r2+1, c1:c2+1]
                        if np.any(sub_map == 1):
                            break 
                        
                        is_maximal = True
                        
                        # Check Up
                        if r1 > 0 and np.all(self.binary_map[r1-1:r2+1, c1:c2+1] == 0): is_maximal = False
                        # Check Down
                        if r2 < self.rows - 1 and np.all(self.binary_map[r1:r2+2, c1:c2+1] == 0): is_maximal = False
                        # Check Left
                        if c1 > 0 and np.all(self.binary_map[r1:r2+1, c1-1:c2+1] == 0): is_maximal = False
                        # Check Right
                        if c2 < self.cols - 1 and np.all(self.binary_map[r1:r2+1, c1:c2+2] == 0): is_maximal = False
                        
                        if is_maximal:
                            rects.append(Rectangle(r1, c1, r2, c2))
        return rects

    def _find_valid_rect_bounds(self):
        rect_bounds = []
        for rect in self.maximal_rects:
            p_min = self.env_maze.cell_rowcol_to_xy(np.array([rect.r_min, rect.c_min]))
            p_max = self.env_maze.cell_rowcol_to_xy(np.array([rect.r_max, rect.c_max]))

            half_scale = self.env_maze.maze_size_scaling * 0.5
            x_min, x_max = p_min[0] - half_scale + self.obs_expand_dis, p_max[0] + half_scale - self.obs_expand_dis
            y_min, y_max = p_max[1] - half_scale + self.obs_expand_dis, p_min[1] + half_scale - self.obs_expand_dis
            
            rect_bounds.append((x_min, x_max, y_min, y_max))

        return rect_bounds


    def safety_check(self, trajectories):
        seq_length = trajectories.shape[1]
        safe_flags = []
        for traj in trajectories:
            is_safe = True
            for t in range(seq_length):
                x, y = traj[t]
                point_safe = False
                for (x_min, x_max, y_min, y_max) in self.valid_rect_bounds:
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        point_safe = True
                        break
                if not point_safe:
                    is_safe = False
                    break
            safe_flags.append(is_safe)

        return safe_flags
    

    def plot_trajectory_comparison(self, true_trajs, gene_trajs, plot_ellips=False, max_plot=100, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        rows, cols = self.env_maze.map_length, self.env_maze.map_width

        scale = self.env_maze.maze_size_scaling
        for r in range(rows):
            for c in range(cols):
                if self.env_maze.maze_map[r][c] == 1:
                    center_xy = self.env_maze.cell_rowcol_to_xy((r, c))
                    patch = patches.Rectangle(
                        (center_xy[0] - scale/2 - self.obs_expand_dis, center_xy[1] - scale/2 - self.obs_expand_dis), 
                        scale + 2*self.obs_expand_dis, scale + 2*self.obs_expand_dis, 
                        linewidth=0, facecolor='#333333', zorder=1
                    )
                    ax.add_patch(patch)

        if plot_ellips:
            label_added = False
            ellips_list = self.maze_obs.get_ellips_list()
            for obs in ellips_list:
                xc, yc, a, b, n = obs
                lbl = 'CBF Obstacle' if not label_added else None

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
        if plot_ellips:
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

if __name__ == "__main__":
    env = Maze2DEnv()
    rects = env._find_maximal_rectangles()
    print("Found Maximal Rectangles:")
    for rect in rects:
        print(rect)