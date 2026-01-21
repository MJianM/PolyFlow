import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

# ==========================================
# 1. 数据录入 (保持不变)
# ==========================================
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

def plot_violation_and_return(raw_data):
    # ==========================================
    # 2. 数据处理 (保持不变)
    # ==========================================
    plot_data = []
    methods_to_plot = ['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow']
    tasks = list(raw_data.keys())
    EPSILON = 0.0008 

    for task in tasks:
        baseline_mag = raw_data[task]['Flow'][0]
        baseline_ret = raw_data[task]['Flow'][1]
        mag_denominator = max(baseline_mag, EPSILON)
        
        for method in methods_to_plot:
            tmp = raw_data[task].get(method, None)
            if tmp is None: continue
            curr_mag = raw_data[task][method][0]
            curr_ret = raw_data[task][method][1]
            
            ratio_mag = curr_mag / mag_denominator
            ratio_ret = curr_ret / baseline_ret
            
            plot_data.append({
                'Method': method,
                'Task': task,
                'Normalized Violation Magnitude': ratio_mag,
                'Normalized Return': ratio_ret
            })

    df = pd.DataFrame(plot_data)

    # ==========================================
    # 3. 绘图设置
    # ==========================================
    sns.set(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.figure(figsize=(3.5, 3.0)) 

    palette = {'PolyFlow': '#d62728', 'SafeFlow': '#1f77b4', 'RoSD': '#2ca02c', 'GaugeFlow': "#8b2ca0"}
    markers = {'Hopper-Simple': 'o', 'Hopper-Hard': 's', 'Walker2d-Hard': '^', 'Walker2d-Simple': 'D'}

    # ==========================================
    # 4. 绘制散点图
    # ==========================================
    ax = sns.scatterplot(
        data=df,
        x='Normalized Violation Magnitude',
        y='Normalized Return',
        hue='Method',
        style='Task',
        palette=palette,
        markers=markers,
        s=70, 
        edgecolor='k',
        linewidth=0.8,
        alpha=0.85
    )

    # ==========================================
    # 5. 坐标轴与辅助线处理 (修改重点)
    # ==========================================

    # 1. 设置 symlog 刻度
    # linthresh=1.0 表示：在 [0, 1] 范围内保持线性（拉伸），在 >1 后变为对数（压缩）
    # linscale=0.6 可以微调线性区域占整个轴的比例，值越小线性区域越宽
    ax.set_xscale('symlog', linthresh=1.0, linscale=0.4)

    # 2. 手动设置刻度 (因为 symlog 的默认刻度可能很奇怪)
    # 我们希望清楚地看到 0, 0.5, 1 (线性区) 以及 2, 3, 4 (压缩区)
    ax.set_xticks([0, 0.5, 1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter()) # 强制显示数字而非 10^0

    # 3. 反转坐标轴
    # 注意：反转需要在设置 scale 之后
    ax.set_xlim(8., -0.2) 

    # 添加基准线
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

    # 标注文字
    plt.text(1.1, 0.94, 'Baseline (Flow)', color='gray', fontsize=7, fontstyle='italic', ha='center')

    # ==========================================
    # 6. 标签与刻度
    # ==========================================
    plt.xlabel(r"Norm. Violation Mag. ($\frac{\text{Method}}{\text{Flow}}$)", fontsize=8)
    plt.ylabel(r"Norm. Return ($\frac{\text{Method}}{\text{Flow}}$)", fontsize=9)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.ylim(0.2, 1.2)

    # ==========================================
    # 7. 图例
    # ==========================================
    plt.legend(
        loc='lower left', 
        bbox_to_anchor=(0.0, 0.0), 
        ncol=1, 
        frameon=True, 
        framealpha=0.95, 
        borderpad=0.3, 
        labelspacing=0.2, 
        handletextpad=0.2, 
        fontsize=7,
        markerscale=0.7
    )

    plt.tight_layout(pad=0.2)
    plt.savefig('safety_tradeoff_plot.pdf', dpi=300, bbox_inches='tight')
    # plt.show()