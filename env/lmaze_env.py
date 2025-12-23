import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# x_min, x_max, y_min, y_max
corridor_1 = [0.0, 3.2, 0.8, 1.2]
corridor_2 = [2.8, 3.2, 0.8, 4.5]
corridors = [corridor_1, corridor_2]

class LMazeEnv:
    def __init__(self):
        
        
        # 定义走廊：[x_min, x_max, y_min, y_max]
        self.corridors_list = [
            [0.0, 3.2, 0.8, 1.2], # Corridor 1
            [2.8, 3.2, 0.8, 4.5]  # Corridor 2
        ]
        # 预先转换为 Tensor 方便后续计算 (num_corridors, 4)
        self.corridors_tensor = torch.tensor(self.corridors_list)

    def safety_check(self, trajectories):
        """
        检查轨迹是否安全（不碰撞）
        
        参数:
        - trajectories: (num_traj, seq_length, x_dim)
        
        返回:
        - list of bool: 每条轨迹是否安全
        """
        safe_list = []
        for i in range(trajectories.shape[0]):
            is_safe = True
            points_safe_list = []
            for j in range(trajectories.shape[1]):
                x, y = trajectories[i, j]
                point_safe = False
                for c in corridors:
                    if c[0] <= x <= c[1] and c[2] <= y <= c[3]:
                        point_safe = True
                        break
                points_safe_list.append(point_safe)
                if not point_safe:
                    is_safe = False
                    break
            safe_list.append(is_safe)
        return safe_list

    def Shield(self, x, x_new, t):
        """
        简单的投影函数，将越界点投影回最近的走廊内
        
        参数:
        - x: (batch_size, seq_length, x_dim) 当前点 tensor
        - x_new: (batch_size, seq_length, x_dim) 新点 tensor
        - t: 当前时间步 (batch_size,)（未使用） tensor
        
        返回:
        - x_proj: (batch_size, seq_length, x_dim) 投影后的点
        """
        # 确保数据在同一设备上
        device = x_new.device
        corridors = self.corridors_tensor.to(device) # (N_corr, 4)
        
        # 1. 准备广播维度
        # x_new: (B, S, 2) -> (B, S, 1, 2) 以便与 N_corr 广播
        point = x_new.unsqueeze(2) 
        
        # 提取边界: (1, 1, N_corr, 1)
        x_min = corridors[:, 0].view(1, 1, -1, 1)
        x_max = corridors[:, 1].view(1, 1, -1, 1)
        y_min = corridors[:, 2].view(1, 1, -1, 1)
        y_max = corridors[:, 3].view(1, 1, -1, 1)
        
        # 2. 对每个走廊分别进行 Clamping (截断)，找到该矩形上的最近点
        # proj_x shape: (B, S, N_corr, 1)
        proj_x = torch.clamp(point[..., 0:1], min=x_min, max=x_max)
        proj_y = torch.clamp(point[..., 1:2], min=y_min, max=y_max)
        
        # 合并为投影点候选集: (B, S, N_corr, 2)
        candidates = torch.cat([proj_x, proj_y], dim=-1)
        
        # 3. 计算原始点到每个候选点的欧氏距离平方
        # dists: (B, S, N_corr)
        dists_sq = torch.sum((candidates - point) ** 2, dim=-1)
        
        # 4. 找到距离最小的那个走廊的索引
        # min_indices: (B, S)
        min_indices = torch.argmin(dists_sq, dim=-1)
        
        # 5. Gather: 提取最优投影点
        # 扩展索引维度以匹配 candidates: (B, S, 1, 2)
        min_indices_expanded = min_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        
        # 最终投影点
        x_proj = torch.gather(candidates, 2, min_indices_expanded).squeeze(2)
        
        return x_proj
    
    def GD(self, x, x_new, t, step_size=0.1, num_steps=1):
        """
        简单的梯度下降投影函数，将越界点拉回走廊内
        
        参数:
        - x: (batch_size, seq_length, x_dim) 当前点 tensor
        - x_new: (batch_size, seq_length, x_dim) 新点 tensor
        - t: 当前时间步 (batch_size,)（未使用）tensor
        
        - step_size: 梯度下降步长（学习率）。
          如果设为 1.0 且 loss 为 0.5*dist^2，行为近似于 Shield。
          通常设小一点（如 0.1）用于软约束引导。

        返回:
        - x_proj: (batch_size, seq_length, x_dim) 投影后的点
        """
        device = x_new.device
        corridors = self.corridors_tensor.to(device)
        
        # 克隆并开启梯度计算，这是 GD 的核心
        # 我们不希望修改原始 x_new 的计算图，只希望计算当前的修正量
        x_opt = x_new.detach().clone().requires_grad_(True)
        
        # 这里必须显式开启 enable_grad，否则下面的计算不会建立计算图
        with torch.enable_grad():
            for _ in range(num_steps):
                # --- 1. 计算损失 (离最近走廊的距离) ---
                point = x_opt.unsqueeze(2) # (B, S, 1, 2)
                
                x_min = corridors[:, 0].view(1, 1, -1, 1)
                x_max = corridors[:, 1].view(1, 1, -1, 1)
                y_min = corridors[:, 2].view(1, 1, -1, 1)
                y_max = corridors[:, 3].view(1, 1, -1, 1)
                
                # 计算点到矩形的向量
                # 使用 maximum 确保只计算外部点的距离，内部点距离为 0
                d_x = torch.maximum(x_min - point[..., 0:1], torch.tensor(0., device=device)) + \
                      torch.maximum(point[..., 0:1] - x_max, torch.tensor(0., device=device))
                d_y = torch.maximum(y_min - point[..., 1:2], torch.tensor(0., device=device)) + \
                      torch.maximum(point[..., 1:2] - y_max, torch.tensor(0., device=device))
                
                # 每个走廊的距离平方
                dist_sq_per_corridor = d_x**2 + d_y**2
                
                # 取最近走廊的距离作为 Loss
                min_dist_sq, _ = torch.min(dist_sq_per_corridor, dim=-1)
                
                loss = min_dist_sq.sum()
                
                # --- 2. 计算梯度 ---
                if loss.item() < 1e-6:
                    break # 都在安全区，无需更新
                
                # 计算 loss 对 x_opt 的梯度
                grad = torch.autograd.grad(loss, x_opt)[0]
                
                # --- 3. 更新点 ---
                # 必须在 no_grad 下更新参数（避免更新步骤本身被计入计算图）
                with torch.no_grad():
                    # 梯度方向是 Loss 增加的方向（远离安全区），所以要减去梯度
                    x_opt = x_opt - step_size * grad
                    
                    # 更新后的 x_opt 需要重新开启梯度记录（如果还要进行下一轮循环）
                    # 实际上 x_opt 在上面减法后变成了新 tensor，这里不需要额外操作，
                    # 但如果在循环中复用，需要确保下一轮能求导。
                    # 上面的写法 x_opt = ... 会创建新节点，若要在循环中保持 requires_grad，
                    # 应该原地修改或者重新设 requires_grad
                    
                    # 修正更新逻辑以支持多步循环：
                    x_opt.requires_grad_(True)
                
        return x_opt.detach()

    def plot_trajectory_comparison(self, true_trajs, gene_trajs, plot_ellips=False, max_plot=100, save_path=None):
        """
        绘制真实轨迹与生成轨迹的对比图
        
        参数:
        - true_trajs: (num_traj, seq_length, x_dim)
        - gene_trajs: (num_traj, seq_length, x_dim)
        - plot_ellips: bool, 是否绘制障碍物膨胀椭圆
        - max_plot: int, 最大绘制轨迹数量
        - save_path: str or None, 保存路径，如果为 None 则不保存
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

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
        true_traj = np.array(true_trajs).reshape(true_trajs.shape[0], -1)
        gene_traj = np.array(gene_trajs).reshape(gene_trajs.shape[0], -1)
        n_true = min(len(true_trajs), max_plot)
        n_gene = min(len(gene_trajs), max_plot)
        true_traj = true_traj[:n_true]
        gene_traj = gene_traj[:n_gene]

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
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()