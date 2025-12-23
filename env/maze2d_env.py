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

# 定义矩形结构方便处理: (r_min, c_min, r_max, c_max) 包含边界
Rectangle = namedtuple('Rectangle', ['r_min', 'c_min', 'r_max', 'c_max'])

class MazeObs:
    """
        CBF 方法使用的椭圆障碍物
    """
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


class Maze2DEnv:

    def __init__(self, maze_map=LARGE_MAZE, obs_expand_dis=0.2, ellips_n=4, alpha=0.5):
        """
        初始化 Maze2D 环境
        
        :param maze_map: 二维列表，表示迷宫结构，1 表示墙壁，0 表示通道
        :param obs_expand_dis: float，障碍物膨胀距离
        """
        self.maze_map = maze_map

        self.env = gym.make('PointMaze_Large-v3', maze_map=maze_map, continuing_task=False, reset_target=False, max_episode_steps=1000,
                            render_mode='rgb_array')
        self.env_maze = self.env.unwrapped.maze
        self.rows, self.cols = self.env_maze.map_length, self.env_maze.map_width
        
        self.obs_expand_dis = obs_expand_dis

        self.binary_map = self._create_binary_map()

        self.maximal_rects = self._find_maximal_rectangles()
        self.valid_rect_bounds = self._find_valid_rect_bounds()

        # CBF 障碍物对象
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
        投影函数，将越界点拉回安全区域
        策略：顺序遍历每个障碍物，如果点在障碍物内，则沿径向（相对于障碍物中心）投影到表面。
        
        :param x: (batch_size, seq_length, x_dim) 原点 tensor
        :param x_new: (batch_size, seq_length, x_dim) 新点 tensor
        :param t: 当前时间步 (batch_size,)（未使用）tensor
        
        返回:
        - x_proj: (batch_size, seq_length, x_dim) 投影后的点
        """
        device = x_new.device
        # 克隆数据，避免原地修改影响后续梯度计算（虽然Shield本身不可导，但为了安全）
        x_proj = x_new.clone()
        
        # 获取障碍物列表: list of [xc, yc, a, b, n]
        obstacles = self.maze_obs.get_ellips_list()
        
        # 转换为 Tensor 方便计算，或者直接循环处理。
        # 由于障碍物数量较少（~8个），循环处理每个障碍物可以自然解决重叠问题（虽不是全局最优，但很稳健）
        # 如果追求极致速度，可以将 obstacles 转为 tensor 进行广播，但在 Shield 中处理重叠比较麻烦。
        
        for obs in obstacles:
            xc, yc, a, b, n = obs
            
            # 1. 计算当前点相对于该障碍物中心的偏移
            dx = x_proj[..., 0] - xc
            dy = x_proj[..., 1] - yc
            
            # 2. 计算超椭圆方程的值 V
            # V = (|dx|/a)^n + (|dy|/b)^n
            # 添加 epsilon 避免除零
            term_x = torch.abs(dx / (a + 1e-6))
            term_y = torch.abs(dy / (b + 1e-6))
            
            # value shape: (batch_size, seq_length)
            value = torch.pow(term_x, n) + torch.pow(term_y, n)
            
            # 3. 找出在障碍物内部的点 (value < 1)
            # 为了数值稳定性，判定阈值可以设为 0.999 或者 1.0
            unsafe_mask = value < 1.0
            
            if not unsafe_mask.any():
                continue
                
            # 4. 计算缩放因子 scale
            # 我们希望找到 k 使得 (|k*dx|/a)^n + ... = 1
            # k^n * value = 1  =>  k = (1 / value)^(1/n)
            # 注意：只计算 unsafe 的点
            
            # 避免 value 为 0 (正好在中心) 导致除零异常
            safe_value = torch.clamp(value, min=1e-6)
            scale_factor = torch.pow(1.0 / safe_value, 1.0 / n)
            
            # 5. 应用修正
            # 新位置 = 中心 + 旧偏移 * 缩放因子
            # 只有 unsafe 的点才会被更新
            
            # 这里的逻辑是：如果 unsafe，则投影；否则保持原样 (scale=1 或不更新)
            # 使用 mask 进行更新
            
            # 计算修正后的坐标
            # 注意：必须使用 mask 索引来只更新特定的点，否则会弄乱 safe 的点
            
            # 提取需要修正的 dx, dy
            dx_unsafe = dx[unsafe_mask]
            dy_unsafe = dy[unsafe_mask]
            s_unsafe = scale_factor[unsafe_mask]
            
            # 径向投影
            new_dx = dx_unsafe * s_unsafe
            new_dy = dy_unsafe * s_unsafe
            
            # 更新 x_proj
            x_proj[unsafe_mask, 0] = xc + new_dx
            x_proj[unsafe_mask, 1] = yc + new_dy
            
        return x_proj

    def GD(self, x, x_new, t, step_size=0.1, num_steps=1):
        """
        使用梯度，将越界点拉回安全区域
        策略：定义 Loss = sum(ReLU(1 - V))，对输入求导并更新。
        
        :param x: (batch_size, seq_length, x_dim) 原点 tensor
        :param x_new: (batch_size, seq_length, x_dim) 新点 tensor
        :param t: 当前时间步 (batch_size,)（未使用）tensor
        
        返回:
        - x_proj: (batch_size, seq_length, x_dim) 投影后的点
        """
        device = x_new.device
        obstacles = self.maze_obs.get_ellips_list()
        
        # 将障碍物参数转为 Tensor 以便进行向量化批处理 (Num_Obs, 5)
        obs_tensor = torch.tensor(obstacles, device=device, dtype=x_new.dtype)
        
        # 提取参数并调整维度以支持广播
        # shape: (1, 1, Num_Obs, 1)
        xc = obs_tensor[:, 0].view(1, 1, -1, 1)
        yc = obs_tensor[:, 1].view(1, 1, -1, 1)
        a  = obs_tensor[:, 2].view(1, 1, -1, 1)
        b  = obs_tensor[:, 3].view(1, 1, -1, 1)
        n  = obs_tensor[:, 4].view(1, 1, -1, 1)

        # 复制并开启梯度
        x_opt = x_new.detach().clone().requires_grad_(True)
        
        for _ in range(num_steps):
            # 扩展 x_opt 维度: (Batch, Seq, 1, 2)
            point = x_opt.unsqueeze(2)
            
            # 计算 dx, dy
            dx = point[..., 0:1] - xc
            dy = point[..., 1:2] - yc
            
            # 计算超椭圆方程值 V
            # V shape: (Batch, Seq, Num_Obs, 1)
            term_x = torch.pow(torch.abs(dx) / a, n)
            term_y = torch.pow(torch.abs(dy) / b, n)
            value = term_x + term_y
            
            # 定义 Loss: 我们希望 Value >= 1
            # 如果 Value < 1 (在内部)，Loss = 1 - Value
            # 使用 ReLU 截断，Safe 区域 Loss 为 0
            # 这是一个 Barrier Function
            violation = torch.relu(1.0 - value)
            
            # 总 Loss (对所有障碍物求和)
            # sum over obstacles (dim=2) and coordinates (dim=3 is 1)
            loss = violation.sum()
            
            # 如果没有违规，提前退出
            if loss.item() < 1e-6:
                break
                
            # 计算梯度
            grad = torch.autograd.grad(loss, x_opt)[0]
            
            # 梯度下降更新
            # 注意：Loss 是 (1 - V)，随着我们往外走，V 增大，Loss 减小
            # 所以我们沿梯度的反方向走 (Gradient Descent)
            with torch.no_grad():
                x_opt = x_opt - step_size * grad
                
        return x_opt.detach()


    def _create_binary_map(self):
        """
        创建二值化的迷宫地图，1 表示墙壁，0 表示通道
        """
        binary_map = np.zeros((self.rows, self.cols), dtype=np.int8)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze_map[r][c] == 1:
                    binary_map[r, c] = 1
                else:
                    binary_map[r, c] = 0
        return binary_map

    def _find_maximal_rectangles(self):
        """
        寻找网格中所有的最大空白矩形。
        这是一个经典算法问题，这里使用简化的全搜索或基于直方图的方法。
        考虑到地图较小(9x12)，我们暴力枚举所有可能的左上角和右下角，
        保留那些不能再扩张的矩形（即Maximal Rectangles）。
        """
        rects = []
        # 遍历所有可能的左上角 (r1, c1)
        for r1 in range(self.rows):
            for c1 in range(self.cols):
                if self.binary_map[r1, c1] == 1: continue
                
                # 遍历所有可能的右下角 (r2, c2)
                for r2 in range(r1, self.rows):
                    for c2 in range(c1, self.cols):
                        # 检查这个矩形区域是否全是0
                        sub_map = self.binary_map[r1:r2+1, c1:c2+1]
                        if np.any(sub_map == 1):
                            break # 这一行碰到墙了，不用再往右看了
                        
                        # 这是一个有效矩形，检查它是否是“最大”的
                        # 只有当它无法向上下左右四个方向扩张时，才是Maximal
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

        # 转换轨迹坐标到网格坐标 (row, col)
        # row=0 col=0 对应 maze 的左上角
        # row 变大 对应的是 y的负方向
        rect_bounds = []
        for rect in self.maximal_rects:
            # 左上角格子中心
            p_min = self.env_maze.cell_rowcol_to_xy(np.array([rect.r_min, rect.c_min]))
            # 右下角格子中心
            p_max = self.env_maze.cell_rowcol_to_xy(np.array([rect.r_max, rect.c_max]))

            half_scale = self.env_maze.maze_size_scaling * 0.5
            x_min, x_max = p_min[0] - half_scale + self.obs_expand_dis, p_max[0] + half_scale - self.obs_expand_dis
            y_min, y_max = p_max[1] - half_scale + self.obs_expand_dis, p_min[1] + half_scale - self.obs_expand_dis
            
            rect_bounds.append((x_min, x_max, y_min, y_max))

        return rect_bounds


    def safety_check(self, trajectories):
        """
        检查给定轨迹是否安全（不碰撞障碍物）
        
        :param trajectories: numpy 数组，形状为 (N, T, 2)，表示 N 条二维轨迹
        :return: list，长度为 N，每个元素为布尔值，表示对应轨迹是否安全
        """
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
        rows, cols = self.env_maze.map_length, self.env_maze.map_width

        scale = self.env_maze.maze_size_scaling

        # ==========================
        # 1. 绘制迷宫墙壁 (Background)
        # ==========================
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

        # ==========================
        # 2. 绘制椭圆障碍物 (Obstacles)
        # ==========================
        if plot_ellips:
            label_added = False
            ellips_list = self.maze_obs.get_ellips_list()
            for obs in ellips_list:
                xc, yc, a, b, n = obs
                # 仅给第一个障碍物加标签，避免图例重复
                lbl = 'CBF Obstacle' if not label_added else None

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
        if plot_ellips:
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

if __name__ == "__main__":
    env = Maze2DEnv()
    rects = env._find_maximal_rectangles()
    print("Found Maximal Rectangles:")
    for rect in rects:
        print(rect)