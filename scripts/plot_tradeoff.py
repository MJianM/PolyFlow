import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==========================================
# 1. 数据预处理 (保留 Safety 部分即可)
# ==========================================
def _preprocess_safety_data(raw_data_safety):
    plot_data = []
    methods_to_plot = ['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow']
    tasks = list(raw_data_safety.keys())
    EPSILON = 0.0008 
    for task in tasks:
        baseline_mag = raw_data_safety[task]['Flow'][0]
        baseline_ret = raw_data_safety[task]['Flow'][1]
        mag_denominator = max(baseline_mag, EPSILON)
        for method in methods_to_plot:
            if method not in raw_data_safety[task]: continue
            curr_mag = raw_data_safety[task][method][0]
            curr_ret = raw_data_safety[task][method][1]
            plot_data.append({
                'Method': method,
                'Task': task,
                'Normalized Violation Magnitude': curr_mag / mag_denominator,
                'Normalized Return': curr_ret / baseline_ret
            })
    return pd.DataFrame(plot_data)

# ==========================================
# 2. 样式定义
# ==========================================
PALETTE = {'PolyFlow': '#d62728', 'SafeFlow': '#1f77b4', 'RoSD': '#2ca02c', 'GaugeFlow': "#8b2ca0"}
MARKERS = {'Hopper-S': 'o', 'Hopper-H': 's', 'Walker2d-S': 'D', 'Walker2d-H': '^'}

def plot_icml_tradeoff_only(raw_data_safety, save_path='icml_tradeoff_0.7col.pdf'):
    # 设置全局字体
    sns.set(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    # ICML单栏宽度通常约为 3.25 英寸
    # 要求 0.7 倍单栏宽度 => 3.25 * 0.7 ≈ 2.275
    # 设置高度为 2.0 以保持良好的长宽比
    fig, ax = plt.subplots(figsize=(2.3, 2.0), dpi=300)
    
    df_safety = _preprocess_safety_data(raw_data_safety)

    # =======================================================
    # 绘图逻辑
    # =======================================================
    sns.scatterplot(
        data=df_safety,
        x='Normalized Violation Magnitude',
        y='Normalized Return',
        hue='Method', style='Task',
        palette=PALETTE, markers=MARKERS,
        s=45, edgecolor='k', linewidth=0.5, alpha=0.9,
        ax=ax, legend=False  # 关闭自动图例，后面手动添加
    )

    # 坐标轴设置：Symlog 使得 0 附近的点显示清晰，大的 Violation 被压缩
    ax.set_xscale('symlog', linthresh=1.0, linscale=0.5)
    ax.set_xticks([0, 1, 4, 8]) 
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # Baseline 参考线
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Baseline 文本标注 (放在左上象限比较安全，或者依据数据分布调整)
    # 这里放在 (0.8, 0.8) 这个相对坐标位置通常比较通用
    ax.text(0.7, 0.92, 'Baseline', color='gray', fontsize=6, fontstyle='italic', ha='right')

    # 标签与标题
    ax.set_xlabel(r"Norm. Max. Violation Mag.", fontsize=8)
    ax.set_ylabel(r"Norm. Return", fontsize=8)
    
    # 字体与刻度微调
    ax.tick_params(axis='both', labelsize=6, pad=0.5)
    
    # 限制范围 (可选，防止图例遮挡数据，或者根据数据自动调整)
    # 适当留白给右下角的图例
    # ax.set_xlim(right=...) 
    # ax.set_ylim(bottom=...)

    # =======================================================
    # 手动构建紧凑图例 (放置于右下角)
    # =======================================================
    # 1. Method Handles (Color)
    method_handles = [Patch(facecolor=PALETTE[m], edgecolor=None, label=m, alpha=0.8) 
                      for m in ['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow']]
    
    # 2. Task Handles (Shape) - 仅显示数据中存在的任务
    existing_tasks = df_safety['Task'].unique()
    task_handles = [Line2D([0], [0], marker=MARKERS[t], color='w', label=t, # 简化名称以节省空间 
                           markerfacecolor='gray', markersize=5) 
                    for t in MARKERS.keys() if t in existing_tasks]

    # 合并图例：为了节省垂直空间，建议将 Method 和 Task 分开列，或者混合在一起
    # 这里采用两列布局：左列是 Methods，右列是 Tasks
    
    # 如果数据点主要集中在左上（高回报低违规），右下角是空的，适合放图例
    leg = ax.legend(
        handles=method_handles + task_handles,
        loc='lower right',
        bbox_to_anchor=(1.0, 0.12),
        ncol=2,             # 分两列显示，更紧凑
        fontsize=5,         # 字体设小
        frameon=True,       # 显示边框背景
        framealpha=0.9,     # 背景半透明，防止遮挡可能的极端点
        edgecolor='gray',   
        borderpad=0.4,
        labelspacing=0.3,
        handletextpad=0.3,
        columnspacing=0.8
    )
    # 调整图例线宽
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Trade-off plot saved to {save_path}")

# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    # 仅提供 Safety 数据即可
    raw_data_safety = {
        'Hopper-S': {'Flow': [0.028, 2450], 'PolyFlow': [0.026, 3187], 'SafeFlow': [0.033, 2628], 'RoSD': [0.003, 961], 'GaugeFlow':[0.024, 2473]},
        'Hopper-H':   {'Flow': [0.456, 2450], 'PolyFlow': [0.137, 2949], 'SafeFlow': [0.751, 2712], 'RoSD': [0.256, 937],  'GaugeFlow':[0.228, 2851]},
        'Walker2d-S':{'Flow': [0.0, 5895],   'PolyFlow': [0.0, 6031],   'SafeFlow': [0.008, 5936], 'RoSD': [0.011, 5816], 'GaugeFlow':[0.001, 5974]},
        'Walker2d-H': {'Flow': [0.534, 5895], 'PolyFlow': [0.484, 5981], 'SafeFlow': [0.513, 5961], 'RoSD': [1.393, 5716], 'GaugeFlow':[0.892, 5783]},
    }
    plot_icml_tradeoff_only(raw_data_safety, save_path='scripts/icml_tradeoff.pdf')