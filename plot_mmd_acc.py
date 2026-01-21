import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# ==========================================
# 1. 数据录入 (来源于你的 LaTeX 表格)
# ==========================================
#以此 Flow 的值为基准进行归一化
BASELINES = {
    'Hopper-Simple':   {'MMD': 2.79e-4, 'Acc': 0.010},
    'Hopper-Hard':     {'MMD': 2.79e-4, 'Acc': 0.010},
    'Walker2d-Simple': {'MMD': 1.01e-3, 'Acc': 0.034},
    'Walker2d-Hard':   {'MMD': 1.01e-3, 'Acc': 0.034},
    'Halfcheetah':     {'MMD': 4.98e-3, 'Acc': 3.229},
}

# 其他方法的原始数据
raw_data = [
    # --- Hopper-Simple ---
    {'Task': 'Hopper-Simple', 'Method': 'PolyFlow',  'MMD': 2.40e-4, 'Acc': 0.012},
    {'Task': 'Hopper-Simple', 'Method': 'SafeFlow',  'MMD': 2.80e-4, 'Acc': 0.010},
    {'Task': 'Hopper-Simple', 'Method': 'RoSD',      'MMD': 2.27e-3, 'Acc': 0.013},
    {'Task': 'Hopper-Simple', 'Method': 'GaugeFlow', 'MMD': 2.80e-4, 'Acc': 0.011},
    
    # --- Hopper-Hard ---
    {'Task': 'Hopper-Hard', 'Method': 'PolyFlow',  'MMD': 1.69e-4, 'Acc': 0.012},
    {'Task': 'Hopper-Hard', 'Method': 'SafeFlow',  'MMD': 3.03e-4, 'Acc': 0.011},
    {'Task': 'Hopper-Hard', 'Method': 'RoSD',      'MMD': 6.26e-3, 'Acc': 0.017},
    {'Task': 'Hopper-Hard', 'Method': 'GaugeFlow', 'MMD': 2.81e-4, 'Acc': 0.011},

    # --- Walker2d-Simple ---
    {'Task': 'Walker2d-Simple', 'Method': 'PolyFlow',  'MMD': 8.47e-4, 'Acc': 0.036},
    {'Task': 'Walker2d-Simple', 'Method': 'SafeFlow',  'MMD': 1.01e-3, 'Acc': 0.034},
    {'Task': 'Walker2d-Simple', 'Method': 'RoSD',      'MMD': 1.40e-3, 'Acc': 0.036},
    {'Task': 'Walker2d-Simple', 'Method': 'GaugeFlow', 'MMD': 9.16e-4, 'Acc': 0.034},

    # --- Walker2d-Hard ---
    {'Task': 'Walker2d-Hard', 'Method': 'PolyFlow',  'MMD': 8.12e-4, 'Acc': 0.036},
    {'Task': 'Walker2d-Hard', 'Method': 'SafeFlow',  'MMD': 1.01e-3, 'Acc': 0.034},
    {'Task': 'Walker2d-Hard', 'Method': 'RoSD',      'MMD': 1.37e-3, 'Acc': 0.036},
    {'Task': 'Walker2d-Hard', 'Method': 'GaugeFlow', 'MMD': 9.30e-4, 'Acc': 0.035},

    # --- Halfcheetah ---
    {'Task': 'Halfcheetah', 'Method': 'PolyFlow',  'MMD': 3.13e-2, 'Acc': 2.735},
    {'Task': 'Halfcheetah', 'Method': 'SafeFlow',  'MMD': 1.79e-2, 'Acc': 2.932},
    {'Task': 'Halfcheetah', 'Method': 'RoSD',      'MMD': 1.61e-2, 'Acc': 3.009},
    {'Task': 'Halfcheetah', 'Method': 'GaugeFlow', 'MMD': 2.93e-2, 'Acc': 2.742},
]

# ==========================================
# 2. 样式定义 (保持与你之前代码一致)
# ==========================================
PALETTE = {
    'PolyFlow': '#d62728',   # Red
    'SafeFlow': '#1f77b4',   # Blue
    'RoSD':     '#2ca02c',   # Green
    'GaugeFlow': "#8b2ca0",   # 
    'Flow':     '#7f7f7f'    # Gray (Baseline)
}

# MARKERS = {
#     'Hopper-Simple': 'o',
#     'Hopper-Hard': 'X',
#     'Walker2d-Simple': '^',
#     'Walker2d-Hard': 'P',
#     'Halfcheetah': 's'
# }

MARKERS = {'Hopper-Simple': 'o', 'Hopper-Hard': 's', 'Walker2d-Simple': 'D', 'Walker2d-Hard': '^', 'Halfcheetah': 'X'}

def preprocess_data(raw_data, baselines):
    processed = []
    # 添加 Baseline (Flow) 数据点 (即 1.0, 1.0)
    for task in baselines.keys():
        processed.append({
            'Task': task,
            'Method': 'Flow',
            'Norm_MMD': 1.0,
            'Norm_Acc': 1.0
        })
        
    # 处理其他方法
    for entry in raw_data:
        task = entry['Task']
        base_mmd = baselines[task]['MMD']
        base_acc = baselines[task]['Acc']
        
        # 归一化：值 / Flow的值
        # 注意：Acc 和 MMD 都是越小越好
        processed.append({
            'Task': task,
            'Method': entry['Method'],
            'Norm_MMD': entry['MMD'] / base_mmd,
            'Norm_Acc': entry['Acc'] / base_acc
        })
    return pd.DataFrame(processed)

def plot_acc_mmd_tradeoff(save_path='icml_acc_mmd.pdf'):
    # 设置 ICML 风格字体
    sns.set(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    df = preprocess_data(raw_data, BASELINES)
    
    # 过滤掉极值点以便更好展示 (例如 RoSD 在 Hopper-Hard 上的 MMD 极大)
    # 你可以根据需要注释掉这一行来显示所有点
    # df = df[df['Norm_MMD'] < 10.0] 

    # ICML 单栏宽度约为 3.25 英寸
    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    
    # 绘制散点
    sns.scatterplot(
        data=df,
        x='Norm_Acc',
        y='Norm_MMD',
        hue='Method',
        style='Task',
        palette=PALETTE,
        markers=MARKERS,
        s=40,               # 点的大小
        edgecolor='k',      # 黑色边缘增加对比度
        linewidth=0.5,
        alpha=0.85,
        ax=ax,
        legend=False        # 手动绘制图例
    )
    
    # 绘制基准线 (Flow = 1.0)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    # 标注 "Baseline"
    ax.text(1.02, 1.02, 'Baseline (Flow)', fontsize=6, color='gray', ha='left', va='bottom')

    # ========== 关键步骤：反转坐标轴 ==========

    # 1. 计算统一的范围 (Common Range)
    # 获取所有数据的最大值和最小值，并增加一点 buffer
    all_values = np.concatenate([df['Norm_Acc'].values, df['Norm_MMD'].values])
    # 过滤掉可能的 inf/nan
    all_values = all_values[np.isfinite(all_values)]
    
    # 确定边界：向外扩展 15% 以防点贴在边上
    # Log 空间下的扩展需要乘除
    d_min = all_values.min() - 0.1
    d_max = all_values.max() + 0.1
    
    # 如果基准线 1.0 不在范围内，强行包含 1.0
    d_min = min(d_min, 0.8) 
    d_max = max(d_max, 1.2)

    # 2. 设置 Log 轴
    # ax.set_xscale('log') 
    # ax.set_yscale('log')

    # 3. 设置统一的范围并反转 (大数在左/下)
    # 注意：invert_xaxis/yaxis 只是反转显示的顺序，set_xlim 需要传入 (max, min) 来配合
    # 或者先 set_xlim(min, max) 再 invert。这里推荐直接传入反转后的元组
    # ax.set_xlim(d_max, d_min) 
    # ax.set_ylim(d_max, d_min)

    # 4. 强制 1:1 比例 (Equal Aspect Ratio)
    # 在 Log 轴下，这意味着 X 轴的一个 decade (例如 1到10) 的物理长度等于 Y 轴的一个 decade
    # ax.set_aspect('equal', adjustable='box')

    # 5. 自定义刻度 (解决重叠问题)
    # 定义你想显示的刻度，例如 [0.5, 1, 2, 5, 10]
    major_ticks = [0.5, 1, 2, 5, 10]
    
    # X 轴格式化
    # ax.set_xticks(major_ticks)
    x_formatter = ticker.ScalarFormatter()
    x_formatter.set_scientific(False) 
    ax.get_xaxis().set_major_formatter(x_formatter)
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter()) # 隐藏次级刻度标签

    # Y 轴格式化 (与 X 轴完全保持一致)
    # ax.set_yticks(major_ticks)
    y_formatter = ticker.ScalarFormatter()
    y_formatter.set_scientific(False)
    ax.get_yaxis().set_major_formatter(y_formatter)
    ax.get_yaxis().set_minor_formatter(ticker.NullFormatter())

    # 坐标轴标签
    ax.set_xlabel(r"Normalized Acc ($\downarrow$)", fontsize=9)
    ax.set_ylabel(r"Normalized MMD ($\downarrow$)", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    
    # ==========================================
    # 自定义图例 (放顶部以节省单栏内部空间)
    # ==========================================
    # 方法图例
    method_handles = [Patch(facecolor=PALETTE[m], label=m) for m in ['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow']]
    # 任务图例
    task_handles = [Line2D([0], [0], marker=MARKERS[t], color='w', label=t.split('-')[0], markerfacecolor='k', markersize=5) for t in MARKERS.keys()]
    
    # 合并图例，放在图上方
    leg = fig.legend(
        handles=method_handles,
        loc='upper center',
        bbox_to_anchor=(0.55, 1.08),
        ncol=4,
        fontsize=7,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.0,
        handletextpad=0.4
    )
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # 给图例留空间
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_acc_mmd_tradeoff(save_path='plot_tradeoff_mmd_acc.png')