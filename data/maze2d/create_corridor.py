import minari
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from collections import namedtuple

# LARGE_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                 [1, 0, 'g', 0, 0, 1, 0, 0, 0, 0, 0, 1],
#                 [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#                 [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
#                 [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#                 [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
#                 [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
#                 [1, 0, 0, 1, 0, 0, 0, 1, 0, 'r', 0, 1],
#                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

LARGE_MAZE_TRUE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

USE_SMALL_CORRIDOR = True

if USE_SMALL_CORRIDOR:
    LARGE_MAZE =   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
else:
    LARGE_MAZE =   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]



# 定义矩形结构方便处理: (r_min, c_min, r_max, c_max) 包含边界
Rectangle = namedtuple('Rectangle', ['r_min', 'c_min', 'r_max', 'c_max'])

class MazeConstraintProcessor:
    def __init__(self, maze_map_str):
        """
        :param maze_map_str: 二维列表或数组，1为墙，0/str为通路
        """
        self.maze_map = np.array(maze_map_str)
        self.rows, self.cols = self.maze_map.shape
        # 将非墙区域标记为 0 (False), 墙标记为 1 (True)
        self.binary_map = np.where(
            (self.maze_map == 1) | (self.maze_map == '1'), 1, 0
        )
        self.maximal_rects = self._find_maximal_rectangles()
        
        # 定义约束矩阵模板 (Ax <= b)
        # 对应: x <= x_max, -x <= -x_min, y <= y_max, -y <= -y_min
        self.A_template = np.array([
            [1, 0],   # x positive
            [-1, 0],  # x negative
            [0, 1],   # y positive
            [0, -1]   # y negative
        ], dtype=np.float32)

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

    def find_closest_rect(self, x, y, rect_list):
        """
        找出离点 (x, y) 最近的所有矩形索引。
        
        :param x: 点的 x 坐标
        :param y: 点的 y 坐标
        :param rect_list: 矩形列表 [[x_min, x_max, y_min, y_max], ...]
        :return: 最近矩形的索引列表 (List[int])，例如 [0, 2]
        """
        min_indices = []
        min_dis_sq = float('inf')
        
        # 定义浮点数比较的容差 (如果坐标全是整数，可以忽略此项，直接用 ==)
        epsilon = 1e-9

        for i, rect in enumerate(rect_list):
            x_min, x_max, y_min, y_max = rect
            
            # 1. 计算 dx 和 dy (同上一步逻辑)
            dx = max(x_min - x, 0, x - x_max)
            dy = max(y_min - y, 0, y - y_max)
            
            # 2. 计算平方距离
            current_dis_sq = dx**2 + dy**2
            
            # 3. 比较逻辑
            if current_dis_sq < min_dis_sq - epsilon:
                # Case A: 发现了新的更小距离
                # 更新最小值，并重置索引列表
                min_dis_sq = current_dis_sq
                min_indices = [i]
                
            elif abs(current_dis_sq - min_dis_sq) <= epsilon:
                # Case B: 距离与当前最小值相等 (在容差范围内)
                # 追加索引
                min_indices.append(i)
                
        return min_indices

    def transform_b_to_rect_bound(self, b_vec):

        xmax, xmin_neg, ymax, ymin_neg = b_vec

        return [-xmin_neg, xmax, -ymin_neg, ymax] 

    def is_equal_rect_bound(self, rect1, rect2):

        flag = True
        for i in range(len(rect1)):
            if abs(rect1[i] - rect2[i]) > 1e-5:
                flag = False
                break
        return flag

    def get_trajectory_constraints(self, traj_xy, env_maze, obs_expand_dis=0.2):
        """
        为一条轨迹生成最优的约束序列。
        策略：贪心算法。在当前点，查看所有包含该点的矩形，
        选择那个能覆盖该点之后最长路径段的矩形。
        :param obs_expand_dis 障碍物膨胀距离
        """
        seq_len = len(traj_xy)
        traj_constraints_A = np.zeros((seq_len, 4, 2))
        traj_constraints_b = np.zeros((seq_len, 4))
        
        # 转换轨迹坐标到网格坐标 (row, col)
        # row=0 col=0 对应 maze 的左上角
        # row 变大 对应的是 y的负方向
        rect_bounds = []
        for rect in self.maximal_rects:
            # 左上角格子中心
            p_min = env_maze.cell_rowcol_to_xy(np.array([rect.r_min, rect.c_min]))
            # 右下角格子中心
            p_max = env_maze.cell_rowcol_to_xy(np.array([rect.r_max, rect.c_max]))

            half_scale = env_maze.maze_size_scaling * 0.5
            x_min, x_max = p_min[0] - half_scale + obs_expand_dis, p_max[0] + half_scale - obs_expand_dis
            y_min, y_max = p_max[1] - half_scale + obs_expand_dis, p_min[1] + half_scale - obs_expand_dis
            
            rect_bounds.append((x_min, x_max, y_min, y_max))

        current_idx = 0
        while current_idx < seq_len:
            # 1. 找出当前点 (current_idx) 在哪些矩形内
            valid_rect_indices = []
            cur_x, cur_y = traj_xy[current_idx]
            
            for i, (xmin, xmax, ymin, ymax) in enumerate(rect_bounds):
                if xmin <= cur_x <= xmax and ymin <= cur_y <= ymax:
                    valid_rect_indices.append(i)
            
            if not valid_rect_indices:
                # 异常处理：轨迹点跑到了墙里或者边界外
                # 策略：找最近的矩形, 如果有多个最近矩形，那么就选上一个轨迹点所在的
                idx_list = self.find_closest_rect(cur_x, cur_y, rect_list=rect_bounds)
                final_idx = -1
                if len(idx_list) > 1 and current_idx - 1 >= 0:
                    last_rect = self.transform_b_to_rect_bound(traj_constraints_b[current_idx-1])
                    for idx in idx_list:
                        if self.is_equal_rect_bound(rect1=last_rect, rect2=rect_bounds[idx]):
                            final_idx = idx
                            break
                    if final_idx == -1:
                        final_idx = idx_list[0]
                else:
                    final_idx = idx_list[0]


                best_rect_idx = final_idx
                max_reach = current_idx + 1


            else:
                # 2. 贪心选择：看哪个矩形能覆盖最远的未来
                best_rect_idx = -1
                max_reach = -1
                
                for r_idx in valid_rect_indices:
                    xmin, xmax, ymin, ymax = rect_bounds[r_idx]
                    # 向前看
                    reach = current_idx
                    while reach < seq_len:
                        tx, ty = traj_xy[reach]
                        if not (xmin <= tx <= xmax and ymin <= ty <= ymax):
                            break
                        reach += 1
                    
                    if reach > max_reach:
                        max_reach = reach
                        best_rect_idx = r_idx
                
            # 3. 填充约束直到 max_reach
            xmin, xmax, ymin, ymax = rect_bounds[best_rect_idx]
            
            # 构造 b 向量: [x_max, -x_min, y_max, -y_min]
            b_vec = np.array([xmax, -xmin, ymax, -ymin])
            
            for k in range(current_idx, max_reach):
                traj_constraints_A[k] = self.A_template
                traj_constraints_b[k] = b_vec
            
            current_idx = max_reach
                
        return traj_constraints_A, traj_constraints_b

def process_and_save_dataset(minari_dataset_id, output_path, target_seq_len=200, obs_expand_dis=0.2):
    # 1. 加载数据
    dataset = minari.load_dataset(minari_dataset_id)
    # 为了获取 Maze 对象和工具函数，我们需要临时创建一个环境
    # env = gym.make("PointMaze_Large-v3", maze_map=LARGE_MAZE)
    env = gym.make('PointMaze_Large-v3', maze_map=LARGE_MAZE, continuing_task=False, reset_target=False, max_episode_steps=1000,
                render_mode='rgb_array')
    maze = env.unwrapped.maze

    processor = MazeConstraintProcessor(LARGE_MAZE)
    
    all_trajs = []
    all_A = []
    all_b = []
    
    print(f"Processing {len(dataset)} trajectories...")
    
    length_list = []

    for episode in dataset:
        # 提取 XY 坐标 (在 observation 的前两维)
        # 原始数据 shape: (steps, obs_dim)
        obs = episode.observations['observation']
        xy_traj = obs[:, :2] 
        
        length_list.append(len(xy_traj))

        # 2. 重采样 (Resampling) 到固定长度
        # 使用线性插值
        orig_len = len(xy_traj)
        orig_indices = np.linspace(0, 1, orig_len)
        target_indices = np.linspace(0, 1, target_seq_len)
        
        resampled_traj = np.zeros((target_seq_len, 2))
        resampled_traj[:, 0] = np.interp(target_indices, orig_indices, xy_traj[:, 0])
        resampled_traj[:, 1] = np.interp(target_indices, orig_indices, xy_traj[:, 1])
        
        # 3. 生成约束
        single_A, single_b = processor.get_trajectory_constraints(resampled_traj, maze, obs_expand_dis=obs_expand_dis)
        
        all_trajs.append(resampled_traj)
        all_A.append(single_A)
        all_b.append(single_b)
        
    # 统计平均长度
    length_arr = np.array(length_list)
    print("length mean, max, min: ", np.mean(length_arr), np.max(length_arr), np.min(length_arr))

    # 转换为 Numpy 数组
    traj_dataset = np.array(all_trajs, dtype=np.float32) # (N, L, 2)
    single_A = np.array(all_A, dtype=np.float32)         # (N, L, 4, 2)
    single_b = np.array(all_b, dtype=np.float32)         # (N, L, 4)
    
    print(f"Saving to {output_path}")
    print(f"Traj Shape: {traj_dataset.shape}")
    print(f"Constraint A Shape: {single_A.shape}")
    
    np.savez(
        output_path, 
        traj_dataset=traj_dataset, 
        single_A=single_A, 
        single_b=single_b
    )

    visualize_trajectory_and_constraints(env_maze=maze, all_rects=processor.maximal_rects, traj=traj_dataset[0], b_seq=single_b[0], obs_expand_dis=obs_expand_dis)

    return 



# %%
# Visualization: Corridor Division
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 下面的代码用于可视化生成的最大矩形以及它们是如何覆盖路径的。

def visualize_trajectory_and_constraints(env_maze, all_rects, traj, b_seq, obs_expand_dis=0.2):
    """
    :param env_maze: env.unwrapped.maze 对象
    :param all_rects: 预处理得到的所有 maximal_rects (用于左图背景)
    :param traj: 轨迹数据 (T, 2)
    :param b_seq: 约束序列 (T, 4)，格式 [x_max, -x_min, y_max, -y_min]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    try:
        rows, cols = env_maze.map_length, env_maze.map_width
    except AttributeError:
        rows, cols = env_maze.maze_map.shape
        
    scale = env_maze.maze_size_scaling
    
    # ====================
    # Subplot 1: Global Map (上帝视角)
    # ====================
    ax1.set_title("Global Map: Maze & All Corridors (Red Outlines)")
    
    # 1.1 画墙 (使用 Native Tool)
    for r in range(rows):
        for c in range(cols):
            # 注意：需要在通过 env.unwrapped.maze 获取的对象上访问 maze_map
            if LARGE_MAZE_TRUE[r][c] == 1:
                center_xy = env_maze.cell_rowcol_to_xy((r, c))
                patch = patches.Rectangle(
                    (center_xy[0] - scale/2, center_xy[1] - scale/2), 
                    scale, scale, color='black'
                )
                ax1.add_patch(patch)
    
    # 1.2 画所有识别出的最大矩形 (修改为红色边框)
    for rect in all_rects:
        p1 = env_maze.cell_rowcol_to_xy((rect.r_min, rect.c_min))
        p2 = env_maze.cell_rowcol_to_xy((rect.r_max, rect.c_max))
        
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        x_min, x_max = min(xs) - scale/2 + obs_expand_dis, max(xs) + scale/2 - obs_expand_dis
        y_min, y_max = min(ys) - scale/2 + obs_expand_dis, max(ys) + scale/2 - obs_expand_dis
        
        # [修改点 1] 将 edgecolor 改为 red，增加一点线宽和不透明度
        ax1.add_patch(patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1.5, edgecolor='red', facecolor='none', linestyle=':', alpha=0.7
        ))

    # 1.3 画轨迹
    ax1.plot(traj[:, 0], traj[:, 1], color='blue', linewidth=2, label='Traj')
    ax1.scatter(traj[0, 0], traj[0, 1], c='green', s=60, zorder=10, label='Start')
    ax1.scatter(traj[-1, 0], traj[-1, 1], c='red', s=60, zorder=10, label='End')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    ax1.autoscale_view()


    # ====================
    # Subplot 2: Active Constraints (局部验证)
    # ====================
    ax2.set_title("Active Constraints Sequence (Points match Box Color)")
    
    # 2.1 提取唯一的约束框序列，并建立轨迹点到唯一约束索引的映射
    unique_constraints = []
    # 用于存储每个时间步对应的唯一约束 ID
    traj_constraint_indices = np.zeros(len(traj), dtype=int) 
    current_unique_idx = -1

    if len(b_seq) > 0:
        # 处理第一个点
        unique_constraints.append(b_seq[0])
        current_unique_idx = 0
        traj_constraint_indices[0] = current_unique_idx
        prev_b = b_seq[0]

        # 处理后续点
        for t in range(1, len(b_seq)):
            # 如果当前约束与上一个不同，说明切换到了新的约束框
            if not np.allclose(b_seq[t], prev_b):
                unique_constraints.append(b_seq[t])
                current_unique_idx += 1
                prev_b = b_seq[t]
            # [修改点 2] 记录当前时间步 t 对应的唯一约束索引
            traj_constraint_indices[t] = current_unique_idx

    num_unique = len(unique_constraints)

    # 生成颜色调色板
    # 使用 'tab10' 或 'Set1' 这种离散 colormap 通常比连续的 plasma 更好区分相邻的框
    # 如果框的数量特别多，可以回退到 continuous colormap
    if num_unique <= 10:
        cmap = cm.get_cmap('tab10')
        colors_palette = [cmap(i) for i in range(num_unique)]
    else:
        # 如果唯一的框太多，使用 plasma 并均匀采样
        colors_palette = cm.plasma(np.linspace(0, 0.9, num_unique))
    
    # 绘制约束框
    for i, b in enumerate(unique_constraints):
        x_max_val = b[0]
        x_min_val = -b[1]
        y_max_val = b[2]
        y_min_val = -b[3]
        
        width = x_max_val - x_min_val
        height = y_max_val - y_min_val
        
        # 稍微加一点偏移让重叠部分能看出来
        # margin = 0.03 * (i % 4) # 使用模运算防止偏移无限增大
        margin = 0.0
        
        rect_patch = patches.Rectangle(
            (x_min_val + margin, y_min_val + margin), 
            width - 2*margin, height - 2*margin,
            linewidth=3, edgecolor=colors_palette[i], facecolor='none', 
            label=f'Box {i}', alpha=0.8
        )
        ax2.add_patch(rect_patch)
        
        # 标号
        ax2.text(x_min_val + margin, y_max_val - margin, f"#{i}", 
                 color=colors_palette[i], fontweight='bold', ha='left', va='top')

    # 2.2 [修改点 3] 绘制轨迹点，颜色匹配其所在的约束框
    if num_unique > 0:
        # 根据索引从调色板中取出对应颜色
        point_colors = [colors_palette[idx] for idx in traj_constraint_indices]
        
        # 先画一条黑色虚线连接所有点，表示时序关系
        ax2.plot(traj[:, 0], traj[:, 1], color='black', linewidth=1, linestyle=':', alpha=0.5, zorder=4)
        
        # 再画彩色的散点
        ax2.scatter(traj[:, 0], traj[:, 1], c=point_colors, s=25, zorder=5, edgecolors='none')
        
        # 标出起点和终点以便定向
        ax2.scatter(traj[0, 0], traj[0, 1], c='green', s=80, zorder=10, marker='>', label='Start')
        ax2.scatter(traj[-1, 0], traj[-1, 1], c='red', s=80, zorder=10, marker='s', label='End')


    # 保持和左图一样的视野范围，方便对比
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_aspect('equal')
    # ax2.legend(loc='lower right') # 图例可能会太多遮挡视线，可以注释掉
    
    plt.tight_layout()
    plt.savefig("large_maze_data_check.png", dpi=150)
    # plt.show() # 如果在notebook中运行可以取消注释


# %%

# ================= 配置与执行 =================


# 假设你已经有了 minari 数据集，ID为 "pointmaze/custom-v0"
# 如果没有，请先运行你的 collect_data() 函数
# collect_data() 

# 运行处理
obs_expand_dis = 0.3
output_file = "large_maze_traj_data_expand_03.npz"
process_and_save_dataset("pointmaze/custom-v0", output_file, target_seq_len=300, obs_expand_dis=obs_expand_dis)

