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