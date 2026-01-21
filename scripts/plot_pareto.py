import matplotlib.pyplot as plt
import numpy as np

# 设置符合学术论文（如ICML）的字体风格
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "figure.titlesize": 10
})

# === 数据准备 ===
# 数据结构: {Method: {'scores': [list of dismatching scores], 'smoothness': [list of raw smoothness]}}
# 任务顺序对应: [Hopper-S, Hopper-C, Walker-S, Walker-C, HalfCheetah]

# Flow 的原始 Smoothness 值，用于归一化
flow_smoothness_bases = np.array([0.0102, 0.0102, 0.0338, 0.0338, 3.2337])

data = {
    'PolyFlow': {
        'd_scores': [0.956, 1.080, 0.899, 0.891, 5.256],
        'raw_smooth': [0.0113, 0.0117, 0.0353, 0.0354, 2.7081]
    },
    'SafeFlow': {
        'd_scores': [0.984, 1.134, 1.000, 0.998, 3.177],
        'raw_smooth': [0.0104, 0.0106, 0.0338, 0.0338, 2.9385]
    },
    'RoSD': {
        'd_scores': [35.276, 27.519, 1.236, 1.237, 3.100],
        'raw_smooth': [0.0129, 0.0167, 0.0363, 0.0362, 3.0107]
    },
    'GaugeFlow': {
        'd_scores': [1.000, 1.289, 1.048, 1.012, 4.335],
        'raw_smooth': [0.0111, 0.0110, 0.0340, 0.0347, 2.7590]
    }
}

PALETTE = {'PolyFlow': '#d62728', 'SafeFlow': '#1f77b4', 'RoSD': '#2ca02c', 'GaugeFlow': "#8b2ca0"}
MARKERS = ['o', 's', '^', 'D', 'v'] # 分别代表5个不同任务，增加信息密度
TASKS = ['Hopper-S', 'Hopper-C', 'Walker-S', 'Walker-C', 'HalfCheetah']

# === 绘图 ===

# ICML单栏宽度约3.25英寸，0.8倍宽度约为2.6英寸
# 高度设为2.0英寸以保持紧凑
fig, ax = plt.subplots(figsize=(2.6, 2.0))

# 绘制 Flow Baseline 参考点 (1, 1)
ax.scatter(1, 1, color='black', marker='*', s=60, zorder=10, label='Flow (Base)')
ax.axvline(x=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# 循环绘制各方法的数据
for method_name, method_data in data.items():
    # 计算归一化 Smoothness: Raw / Flow_Base
    norm_smoothness = np.array(method_data['raw_smooth']) / flow_smoothness_bases
    x = method_data['d_scores']
    y = norm_smoothness
    
    # 绘制散点
    # 使用单一marker或者根据任务区分marker。为了清晰展示Pareto，这里统一用圆点，
    # 但如果您希望区分任务，可以解开下方的循环注释。
    ax.scatter(x, y, c=PALETTE[method_name], label=method_name, s=25, alpha=0.8, edgecolors='white', linewidth=0.5)

# === 轴设置 ===
ax.set_xlabel('Dismatching Score ($\downarrow$)', labelpad=2)
ax.set_ylabel('Norm. Acc. Smoothness ($\downarrow$)', labelpad=2)

# 设置X轴为对数坐标，因为RoSD的值(35.27)远大于其他方法，线性坐标会挤压左下角的重要区域
# ax.set_xscale('log') 

# 调整刻度显示，使其看起来更自然
from matplotlib.ticker import ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())
# ax.set_xticks([0.9, 1, 2, 5, 10, 30])
ax.set_xlim([0.5, 2])

# 移除上方和右侧的边框 (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加网格
ax.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)

# === 图例设置 ===
# 将图例放在图内空白处或上方，这里选择放在右上方，尽量紧凑
legend = ax.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='none', handletextpad=0.1, borderpad=0.2)

# 紧凑布局
plt.tight_layout(pad=0.2)

# 保存或显示
plt.savefig('scripts/pareto_frontier.png', dpi=300) 
# plt.show()