import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from plot_tradeoff_mag_return import plot_violation_and_return

# ==========================================
# 通用配置与数据预处理
# ==========================================
def _preprocess_data(raw_data_time):
    """
    内部辅助函数：将字典转换为DataFrame，并计算相对于Flow的归一化时间
    """
    plot_data = []
    # 定义绘图顺序
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

def _set_style():
    """设置论文绘图风格"""
    sns.set(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# 统一颜色定义
PALETTE = {
    'PolyFlow': '#d62728', 
    'SafeFlow': '#1f77b4', 
    'RoSD': '#2ca02c', 
    'GaugeFlow': "#8b2ca0"
}

# ==========================================
# 函数 1: 箱线图 + 散点叠加 (分布对比)
# ==========================================
def plot_inference_boxplot(raw_data_time, save_path='inference_boxplot.pdf'):
    """
    绘制箱线图叠加散点图，用于展示不同方法的整体分布情况。
    """
    df = _preprocess_data(raw_data_time)
    _set_style()
    
    # 定义形状映射
    markers = {
        'Hopper-Simple': 'o', 'Hopper-Hard': 's',
        'Walker2d-Simple': 'D', 'Walker2d-Hard': '^',
        'HalfCheetah': 'X'
    }

    plt.figure(figsize=(3.5, 3.0))

    # Layer 1: Boxplot
    ax = sns.boxplot(
        data=df, x='Method', y='Normalized Time',
        hue='Method', palette=PALETTE,
        dodge=False, width=0.5, linewidth=1.0,
        showfliers=False, boxprops={'alpha': 0.4}
    )

    # Layer 2: Scatterplot
    sns.scatterplot(
        data=df, x='Method', y='Normalized Time',
        style='Task', markers=markers,
        hue='Method', palette=PALETTE,
        s=25, linewidth=0.5, # linewidth=0 去除边框
        ax=ax, zorder=10, legend=True
    )

    # 坐标轴设置
    ax.set_yscale('log')
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
    plt.text(2.3, 0.75, 'Baseline (Flow)', fontsize=8, color='gray', fontstyle='italic', va='bottom', ha='center')
    
    ax.set_yticks([0.5, 1, 2, 5, 10])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("")
    ax.set_ylabel(r"Norm. Inference Time ($\frac{\text{Method}}{\text{Flow}}$)", fontsize=9)

    # 图例处理：只保留Task
    handles, labels = ax.get_legend_handles_labels()
    final_handles, final_labels = [], []
    for h, l in zip(handles, labels):
        if l in markers.keys():
            final_handles.append(h)
            final_labels.append(l)

    plt.legend(
        final_handles, final_labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.25),
        ncol=3, frameon=False, fontsize=7, 
        columnspacing=0.8, handletextpad=0.1
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Boxplot saved to {save_path}")

# ==========================================
# 函数 2: 折线图 (趋势对比)
# ==========================================
def plot_inference_trend(raw_data_time, save_path='inference_trend.pdf'):
    """
    绘制折线图，横坐标为不同的任务，纵坐标为时间。
    用于对比不同方法在任务复杂度增加时的变化趋势。
    """
    df = _preprocess_data(raw_data_time)
    _set_style()

    # 关键步骤：指定 X 轴的顺序 (Simple -> Hard)
    # 这样折线的斜率才有物理意义（复杂度上升带来的开销）
    task_order = ['Hopper-Simple', 'Hopper-Hard', 'Walker2d-Simple', 'Walker2d-Hard', 'HalfCheetah']
    
    # 将 Task 列转换为 categorical 类型以固定排序
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)

    plt.figure(figsize=(4.0, 3.0)) # 稍微宽一点以容纳 X 轴标签

    # 绘制折线图
    # markers=True 会自动给每个点加标记
    # dashes=False 保证所有线都是实线
    ax = sns.lineplot(
        data=df,
        x='Task', 
        y='Normalized Time',
        hue='Method', 
        palette=PALETTE,
        style='Method',       # 让线型或标记也随方法变化，便于黑白打印识别
        markers=True,         # 显示数据点
        dashes=False,         # 禁用虚线，全部用实线
        linewidth=1.2,
        markersize=6,
        alpha=0.9
    )

    # 坐标轴设置
    ax.set_yscale('log')
    
    # 基准线
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=2.0, alpha=0.9)
    plt.text(len(task_order)-2, 0.75, 'Baseline', fontsize=8, color='gray', va='bottom', ha='right')

    # X轴标签调整
    ax.tick_params(axis='x', which='major', pad=1)
    # plt.xticks(rotation=20, ha='right', fontsize=6) # 稍微倾斜防止重叠
    plt.xticks(fontsize=6)
    plt.xlabel("")
    plt.ylabel(r"Norm. Inference Time (Log Scale)", fontsize=8, labelpad=3)
    
    # Y轴刻度
    ax.set_yticks([0.5, 1, 2, 5, 10, 15])
    plt.yticks(fontsize=6)
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # 图例设置
    plt.legend(
        title="",
        loc='upper left', 
        # bbox_to_anchor=(1.02, 1.0), # 放在图外右侧
        frameon=False,
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Trend plot saved to {save_path}")

# ==========================================
# 调用示例
# ==========================================
if __name__ == "__main__":
    # 原始数据
    raw_data_time = {
        'Hopper-Simple':   {'Flow': 0.898, 'PolyFlow': 0.654, 'SafeFlow': 1.089, 'RoSD': 2.278, 'GaugeFlow': 1.179},
        'Hopper-Hard':     {'Flow': 0.898, 'PolyFlow': 0.644, 'SafeFlow': 6.084, 'RoSD': 3.374, 'GaugeFlow': 1.350},
        'Walker2d-Simple': {'Flow': 0.531, 'PolyFlow': 0.606, 'SafeFlow': 0.684, 'RoSD': 2.445, 'GaugeFlow': 0.661},
        'Walker2d-Hard':   {'Flow': 0.589, 'PolyFlow': 0.592, 'SafeFlow': 8.578, 'RoSD': 1.902, 'GaugeFlow': 0.599},
        'HalfCheetah':     {'Flow': 0.531, 'PolyFlow': 1.175, 'SafeFlow': 8.039, 'RoSD': 2.125, 'GaugeFlow': 0.551}
    }

    # 1. 画箱线图
    plot_inference_boxplot(raw_data_time)

    # 2. 画趋势折线图
    plot_inference_trend(raw_data_time)


    # 画时间箱线图
    raw_data_time = {
        'Hopper-Simple':   {'Flow': 0.898, 'PolyFlow': 0.654, 'SafeFlow': 1.089, 'RoSD': 2.278, 'GaugeFlow': 1.179},
        'Hopper-Hard':     {'Flow': 0.898, 'PolyFlow': 0.644, 'SafeFlow': 6.084, 'RoSD': 3.374, 'GaugeFlow': 1.350},
        'Walker2d-Simple': {'Flow': 0.531, 'PolyFlow': 0.606, 'SafeFlow': 0.684, 'RoSD': 2.445, 'GaugeFlow': 0.661},
        'Walker2d-Hard':   {'Flow': 0.589, 'PolyFlow': 0.592, 'SafeFlow': 8.578, 'RoSD': 1.902, 'GaugeFlow': 0.599},
        'HalfCheetah':     {'Flow': 0.531, 'PolyFlow': 1.175, 'SafeFlow': 8.039, 'RoSD': 2.125, 'GaugeFlow': 0.551}
    }
    plot_inference_boxplot(raw_data_time)

    #画帕累托图
    raw_data = {
        'Hopper-Simple': {
            'Flow':     [0.031, 2545],
            'PolyFlow': [0.017, 2329],
            'SafeFlow': [0.020, 2291],
            'RoSD':     [0.033, 1226],
            'GaugeFlow':[0.017, 2481],
        },
        'Hopper-Hard': {
            'Flow':     [0.818, 2545],
            'PolyFlow': [0.338, 2627],
            'SafeFlow': [0.751, 2712],
            'RoSD':     [0.648, 695],
            'GaugeFlow':[0.142, 2704],
        },
        'Walker2d-Simple': {
            'Flow':     [0.0, 5861],    
            'PolyFlow': [0.0, 6039],
            'SafeFlow': [0.001, 5904],
            'RoSD':     [0.005, 5726],
            'GaugeFlow':[0.002, 5969]
        },
        'Walker2d-Hard': {
            'Flow':     [0.551, 5861],
            'PolyFlow': [0.467, 6004],
            'SafeFlow': [0.537, 5934],
            'RoSD':     [1.839, 5709],
            'GaugeFlow':[1.252, 5775]
        },

    }
    plot_violation_and_return(raw_data)