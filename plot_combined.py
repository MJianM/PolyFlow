import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==========================================
# 1. 数据预处理
# ==========================================
def _preprocess_time_data(raw_data_time):
    plot_data = []
    methods_order = ['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow']
    tasks = list(raw_data_time.keys())
    for task in tasks:
        baseline_time = raw_data_time[task]['Flow']
        for method in methods_order:
            if method not in raw_data_time[task]: continue
            plot_data.append({
                'Method': method,
                'Task': task,
                'Normalized Time': raw_data_time[task][method] / baseline_time
            })
    return pd.DataFrame(plot_data)

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
MARKERS = {'Hopper-Simple': 'o', 'Hopper-Hard': 's', 'Walker2d-Simple': 'D', 'Walker2d-Hard': '^', 'HalfCheetah': 'X'}

def plot_icml_1x3(raw_data_time, raw_data_safety, save_path='icml_3_subplots.pdf'):
    # 设置全局字体
    sns.set(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    # 初始化画布: 1行3列
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    
    # 根据你提供的顺序：
    # axes[0]: Safety (左)
    # axes[1]: Boxplot (中)
    # axes[2]: Trend (右)
    ax_safe = axes[0]  
    ax_box = axes[1]   
    ax_line = axes[2]  

    df_time = _preprocess_time_data(raw_data_time)
    df_safety = _preprocess_safety_data(raw_data_safety)

    # =======================================================
    # Subplot 1 (Left): Safety Trade-off (Scatter)
    # =======================================================
    # 注意：根据你的变量分配，ax_safe现在是axes[0]
    sns.scatterplot(
        data=df_safety,
        x='Normalized Violation Magnitude',
        y='Normalized Return',
        hue='Method', style='Task',
        palette=PALETTE, markers=MARKERS,
        s=40, edgecolor='k', linewidth=0.5, alpha=0.85,
        ax=ax_safe, legend=False
    )

    ax_safe.set_xscale('symlog', linthresh=1.0, linscale=0.5)
    ax_safe.set_xticks([0, 1, 4, 8]) 
    ax_safe.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    # ax_safe.set_xlim(9.0, -0.2)
    # ax_safe.set_ylim(0.2, 1.25)

    ax_safe.axhline(y=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ax_safe.axvline(x=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    
    # 【修改】标注 Baseline
    # 位置选择在 (1.1, 0.25) 附近，即 Return 较低、Violation 接近 1 的空白处
    ax_safe.text(0.8, 0.8, 'Baseline', color='gray', fontsize=6, fontstyle='italic', ha='center')

    ax_safe.set_xlabel(r"Norm. Violation Mag", fontsize=8)
    ax_safe.set_ylabel(r"Norm. Return", fontsize=8)
    # 调整标题为 (a), 假设放在最左边
    ax_safe.set_title("(a) Safety-Return Trade-off", fontsize=9, pad=3)
    ax_safe.tick_params(axis='both', labelsize=7)

    # =======================================================
    # Subplot 2 (Middle): Inference Speed Distribution (Boxplot)
    # =======================================================
    sns.boxplot(
        data=df_time, x='Method', y='Normalized Time',
        hue='Method', palette=PALETTE,
        dodge=False, width=0.5, linewidth=0.8,
        showfliers=False, boxprops={'alpha': 0.4},
        ax=ax_box
    )
    sns.scatterplot(
        data=df_time, x='Method', y='Normalized Time',
        style='Task', markers=MARKERS,
        hue='Method', palette=PALETTE,
        s=20, linewidth=0, 
        ax=ax_box, zorder=10, legend=False
    )
    
    ax_box.set_yscale('log')
    ax_box.axhline(y=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ax_box.text(2.8, 0.8, 'Baseline', fontsize=6, color='gray', fontstyle='italic', va='top', ha='right')
    
    ax_box.set_yticks([0.5, 1, 2, 5, 10])
    ax_box.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_box.set_xlabel("")
    ax_box.set_xticklabels(['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow'], rotation=30, ha='right', fontsize=6.5)
    
    # 【修改】调小 labelpad, tick pad 和 fontsize
    ax_box.set_ylabel(r"Norm. Inference Time", fontsize=8, labelpad=0)
    ax_box.set_title("(b) Inference Time Dist.", fontsize=9, pad=3)
    # pad=1: 刻度数字离轴线更近; labelsize=6: 字体更小
    ax_box.tick_params(axis='y', labelsize=6, pad=1)
    ax_box.tick_params(axis='x', pad=1)

    # =======================================================
    # Subplot 3 (Right): Scalability Trend (Lineplot)
    # =======================================================
    task_order = ['Hopper-Simple', 'Hopper-Hard', 'Walker2d-Simple', 'Walker2d-Hard', 'HalfCheetah']
    df_time_trend = df_time.copy()
    df_time_trend['Task'] = pd.Categorical(df_time_trend['Task'], categories=task_order, ordered=True)
    
    sns.lineplot(
        data=df_time_trend, x='Task', y='Normalized Time',
        hue='Method', palette=PALETTE,
        style='Method', 
        markers=True, dashes=False,
        linewidth=1.2, markersize=5, alpha=0.9,
        ax=ax_line, legend=False
    )

    ax_line.set_yscale('log')
    ax_line.axhline(y=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    
    # 【修改】标注 Baseline
    # 位置设在最后一个任务索引处，y=0.8 (Baseline下方)
    ax_line.text(4.2, 0.8, 'Baseline', fontsize=6, color='gray', fontstyle='italic', va='top', ha='right')
    
    ax_line.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax_line.set_ylim(top=25)
    ax_line.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    
    ax_line.set_xlabel("")
    
    # 【修改】添加 Y Label 并调小 padding/font
    # 虽然和左图（中图）单位一样，但在多图并排时，有时候为了阅读方便也会加上
    # 这里使用简称 "Norm. Time" 以节省空间
    ax_line.set_ylabel("Norm. Inference Time", fontsize=8, labelpad=0)
    
    ax_line.set_title("(c) Time Complexity Curve", fontsize=9, pad=3)
    
    task_labels = ['Hopper-S', 'Hopper-H', 'Walker2d-S', 'Walker2d-H', 'HalfCheetah']
    ax_line.set_xticklabels(task_labels, rotation=30, ha='right', fontsize=6.5)
    ax_line.tick_params(axis='x', pad=0)
    
    # 【修改】调小 Y 轴刻度字体和距离
    ax_line.tick_params(axis='y', labelsize=6, pad=1)

    # =======================================================
    # 共享图例
    # =======================================================
    method_handles = [Patch(facecolor=PALETTE[m], edgecolor=None, label=m, alpha=0.8) for m in ['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow']]
    task_handles = [Line2D([0], [0], marker=MARKERS[t], color='w', label=t, markerfacecolor='k', markersize=6) for t in MARKERS.keys()]

    fig.legend(
        handles=method_handles + task_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15), 
        ncol=5,                     
        fontsize=7.5,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.3,
        borderpad=0
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.25, bottom=0.15) 

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Combined 1x3 plot saved to {save_path}")
# ==========================================
# 运行
# ==========================================
if __name__ == "__main__":
    raw_data_time = {
        'Hopper-Simple':   {'Flow': 0.898, 'PolyFlow': 0.654, 'SafeFlow': 1.089, 'RoSD': 2.278, 'GaugeFlow': 1.179},
        'Hopper-Hard':     {'Flow': 0.898, 'PolyFlow': 0.644, 'SafeFlow': 6.084, 'RoSD': 3.374, 'GaugeFlow': 1.350},
        'Walker2d-Simple': {'Flow': 0.531, 'PolyFlow': 0.606, 'SafeFlow': 0.684, 'RoSD': 2.445, 'GaugeFlow': 0.661},
        'Walker2d-Hard':   {'Flow': 0.589, 'PolyFlow': 0.592, 'SafeFlow': 8.578, 'RoSD': 1.902, 'GaugeFlow': 0.599},
        'HalfCheetah':     {'Flow': 0.531, 'PolyFlow': 0.424, 'SafeFlow': 8.039, 'RoSD': 2.125, 'GaugeFlow': 0.551}
    }
    raw_data_safety = {
        'Hopper-Simple': {'Flow': [0.028, 2450], 'PolyFlow': [0.026, 3187], 'SafeFlow': [0.033, 2628], 'RoSD': [0.003, 961], 'GaugeFlow':[0.024, 2473]},
        'Hopper-Hard':   {'Flow': [0.456, 2450], 'PolyFlow': [0.137, 2949], 'SafeFlow': [0.751, 2712], 'RoSD': [0.256, 937],  'GaugeFlow':[0.228, 2851]},
        'Walker2d-Simple':{'Flow': [0.0, 5895],   'PolyFlow': [0.0, 6031],   'SafeFlow': [0.008, 5936], 'RoSD': [0.011, 5816], 'GaugeFlow':[0.001, 5974]},
        'Walker2d-Hard': {'Flow': [0.534, 5895], 'PolyFlow': [0.484, 5981], 'SafeFlow': [0.513, 5961], 'RoSD': [1.393, 5716], 'GaugeFlow':[0.892, 5783]},
    }
    plot_icml_1x3(raw_data_time, raw_data_safety, save_path='combined.png')