import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ==========================================
# 1. 数据录入 (保持不变)
# ==========================================
raw_data = {
    'Hopper-Simple': {
        'Flow':     [0.031, 2545],
        'PolyFlow': [0.017, 2329],
        'SafeFlow': [0.020, 2291],
        'RoSD':     [0.033, 1226]
    },
    'Hopper-Hard': {
        'Flow':     [0.818, 2545],
        'PolyFlow': [0.338, 2627],
        'SafeFlow': [0.751, 2712],
        'RoSD':     [0.648, 695]
    },
    'Walker2d-Hard': {
        'Flow':     [0.551, 5861],
        'PolyFlow': [0.467, 6004],
        'SafeFlow': [0.537, 5934],
        'RoSD':     [1.839, 5709]
    },
    'Walker2d-Simple': {
        'Flow':     [0, 5861],
        'PolyFlow': [0, 6039],
        'SafeFlow': [0.001, 5904],
        'RoSD':     [0.005, 5726],
    }
}

# ==========================================
# 2. 数据处理 (保持不变)
# ==========================================
plot_data = []
methods_to_plot = ['PolyFlow', 'SafeFlow', 'RoSD']
tasks = list(raw_data.keys())

for task in tasks:
    baseline_mag = raw_data[task]['Flow'][0]
    baseline_ret = raw_data[task]['Flow'][1]
    
    for method in methods_to_plot:
        curr_mag = raw_data[task][method][0]
        curr_ret = raw_data[task][method][1]
        
        ratio_mag = curr_mag / baseline_mag
        ratio_ret = curr_ret / baseline_ret
        
        plot_data.append({
            'Method': method,
            'Task': task,
            'Normalized Violation Magnitude': ratio_mag,
            'Normalized Return': ratio_ret
        })

df = pd.DataFrame(plot_data)

# ==========================================
# 3. 绘图设置 (ICML 单栏优化版)
# ==========================================
# ICML 单栏宽度约为 3.25 英寸。
# 设置 figsize 为 (3.4, 2.8) 可以在插入时正好占满单栏且保持良好的纵横比。
sns.set(style="whitegrid", context="paper") 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif'] # 尝试匹配 LaTeX 字体
plt.figure(figsize=(3.4, 2.8)) 

# 颜色与标记
palette = {'PolyFlow': '#d62728', 'SafeFlow': '#1f77b4', 'RoSD': '#2ca02c'}
markers = {'Hopper-Simple': 'o', 'Hopper-Hard': 's', 'Walker2d-Hard': '^'}

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
    s=70,            # 减小点的大小以适应小图
    edgecolor='k',
    linewidth=0.8,   # 减小边缘线宽
    alpha=0.9
)

# ==========================================
# 5. 辅助元素
# ==========================================
# 基准线
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
# 字体调小
plt.text(1.1, 0.94, 'Baseline (Flow)', color='gray', fontsize=8, fontstyle='italic')

# ==========================================
# 6. 轴设置 (字体精细控制)
# ==========================================
# 使用 LaTeX 格式，字体设为 9 (适合 caption 大小)
plt.xlabel(r"Norm. Violation Mag. ($\frac{\text{Method}}{\text{Flow}}$)", fontsize=9)
plt.ylabel(r"Norm. Return ($\frac{\text{Method}}{\text{Flow}}$)", fontsize=9)

# 刻度字体设为 8
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# 范围微调
plt.xlim(0, 3.5)
plt.ylim(0.1, 1.25) # 稍微增加上限给上面留点呼吸空间

# ==========================================
# 7. 紧凑图例 (放在右下角)
# ==========================================
# 这里使用了极度紧凑的设置，以防图例占据过多空间
plt.legend(
    loc='lower right', 
    bbox_to_anchor=(1.0, 0.0), # 紧贴右下角
    ncol=1, 
    frameon=True, 
    framealpha=0.95, 
    borderpad=0.3,       # 减小边框内边距
    labelspacing=0.2,    # 减小标签行间距
    handletextpad=0.2,   # 减小符号和文字的间距
    fontsize=7,          # 字体设小
    markerscale=0.7      # 减小图例中的符号大小
)

# 调整布局，防止标签被切
plt.tight_layout(pad=0.2)

# ==========================================
# 8. 保存
# ==========================================
plt.savefig('safety_tradeoff_plot.png', dpi=300, bbox_inches='tight')
plt.show()