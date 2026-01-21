import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. 全局绘图风格设置 (ICML 风格) ---
sns.set_theme(style="white", rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5
})

# --- 2. 数据准备 ---
# 原始 Wasserstein 数据 (来自表格)
# 顺序: Hopper-S, Hopper-H, Walker-S, Walker-H, HalfCheetah
envs = ['Hopper-S', 'Hopper-H', 'Walker-S', 'Walker-H', 'HalfCheetah']
raw_data = {
    'Flow (Baseline)': [1.184, 1.184, 3.641, 3.641, 0.705],
    'PolyFlow':        [1.106, 1.085, 3.429, 3.407, 0.832],
    'SafeFlow':        [1.184, 1.181, 3.641, 3.640, 0.799],
    'RoSD':            [1.832, 2.033, 3.780, 3.776, 0.796],
    'GaugeFlow':       [1.169, 1.165, 3.615, 3.624, 0.815]
}

# --- 3. 数据处理：构建归一化的 DataFrame ---
rows = []
baseline_vals = raw_data['Flow (Baseline)']
methods_to_plot = ['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow']

for i, env in enumerate(envs):
    base_val = baseline_vals[i]
    for method in methods_to_plot:
        # 计算归一化值 (Value / Baseline)
        norm_val = raw_data[method][i] / base_val
        rows.append({
            'Environment': env,
            'Method': method,
            'Normalized W': norm_val
        })

df = pd.DataFrame(rows)

# --- 4. 颜色定义 ---
PALETTE = {
    'PolyFlow': '#d62728',   # 红色
    'SafeFlow': '#1f77b4',   # 蓝色
    'RoSD':     '#2ca02c',   # 绿色
    'GaugeFlow': "#8b2ca0"   # 紫色
}

# --- 5. 绘图 ---
# 设置画布大小 (ICML单栏宽度建议在 3.25~4 英寸之间，这里设宽一点为了清晰)
plt.figure(figsize=(7, 3.0), dpi=300)

# 绘制柱状图
ax = sns.barplot(
    data=df,
    x='Environment',
    y='Normalized W',
    hue='Method',
    palette=PALETTE,
    edgecolor='white', # 柱子边缘颜色
    linewidth=0.8,     # 柱子边缘宽度
    width=0.75,        # 柱子组的总宽度
    saturation=0.8     # 防止 seaborn 自动降低饱和度
)

# --- 6. 细节修饰 ---

# 添加基准线 (y=1.0)
ax.axhline(1.0, color='#555555', linestyle='--', linewidth=2.5, alpha=0.8, zorder=0)
# 在图的最右侧添加文字标注
ax.text(2.45, 1.1, 'Flow Baseline', va='center', ha='left', 
        fontsize=10, color='#555555', style='italic')

# 坐标轴设置
ax.set_ylabel('Normalized Wasserstein Dis.', fontsize=11)
ax.set_xlabel('') # 移除 X 轴标题，因为标签已经很清楚了
ax.set_ylim(0, 1.9) # 根据数据调整，给图例留出空间

# 优化图例 (放在上方，水平排列)
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(0.5, 1.02), # 放在图表上方
    ncol=4, 
    title=None, 
    frameon=False,
    fontsize=12
)

# 去除多余边框 (左、右、上)
sns.despine(left=False, bottom=False, top=True, right=True)

# 调整布局
plt.tight_layout()

# --- 7. 保存 ---
plt.savefig('w_compare.pdf', bbox_inches='tight', dpi=300)
# plt.show()






# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# # --- 1. 全局绘图风格设置 ---
# sns.set_theme(style="white", rc={
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "axes.grid": True,
#     "grid.linestyle": "--",
#     "grid.alpha": 0.5,
#     "axes.linewidth": 0.8
# })

# # --- 2. 数据准备 ---
# envs = ['Hopper-S', 'Hopper-H', 'Walker-S', 'Walker-H', 'HalfCheetah']
# methods_to_plot = ['PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow']

# # 原始数据 (来自表格)
# raw_data_w = {
#     'Flow (Baseline)': [1.184, 1.184, 3.641, 3.641, 0.705],
#     'PolyFlow':        [1.106, 1.085, 3.429, 3.407, 0.832],
#     'SafeFlow':        [1.184, 1.181, 3.641, 3.640, 0.799],
#     'RoSD':            [1.832, 2.033, 3.780, 3.776, 0.796],
#     'GaugeFlow':       [1.169, 1.165, 3.615, 3.624, 0.815]
# }

# raw_data_acc = {
#     'Flow (Baseline)': [0.010, 0.010, 0.034, 0.034, 3.229],
#     'PolyFlow':        [0.012, 0.012, 0.036, 0.036, 2.735], # HalfCheetah中PolyFlow更平滑
#     'SafeFlow':        [0.010, 0.011, 0.034, 0.034, 2.932],
#     'RoSD':            [0.013, 0.017, 0.036, 0.036, 3.009],
#     'GaugeFlow':       [0.011, 0.011, 0.034, 0.035, 2.742]
# }

# # --- 3. 数据归一化处理函数 ---
# def create_norm_df(raw_data, value_name):
#     rows = []
#     baseline_vals = raw_data['Flow (Baseline)']
#     for i, env in enumerate(envs):
#         base_val = baseline_vals[i]
#         for method in methods_to_plot:
#             norm_val = raw_data[method][i] / base_val
#             rows.append({
#                 'Environment': env,
#                 'Method': method,
#                 value_name: norm_val
#             })
#     return pd.DataFrame(rows)

# df_w = create_norm_df(raw_data_w, 'Normalized W')
# df_acc = create_norm_df(raw_data_acc, 'Normalized Acc')

# # --- 4. 颜色定义 ---
# PALETTE = {
#     'PolyFlow': '#d62728',   # 红色
#     'SafeFlow': '#1f77b4',   # 蓝色
#     'RoSD':     '#2ca02c',   # 绿色
#     'GaugeFlow': "#8b2ca0"   # 紫色
# }

# # --- 5. 绘图 (2行1列) ---
# # 增加高度以容纳两个子图
# fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), dpi=300, sharex=True)

# # 通用绘图参数
# bar_params = dict(
#     x='Environment',
#     hue='Method',
#     palette=PALETTE,
#     edgecolor='white',
#     linewidth=0.8,
#     width=0.75,
#     saturation=1.0
# )

# # === 子图 1: Wasserstein Distance ===
# sns.barplot(data=df_w, y='Normalized W', ax=axes[0], **bar_params)

# # 修饰子图 1
# axes[0].set_ylabel('Norm. Wasserstein ($\downarrow$)', fontsize=10)
# axes[0].set_xlabel('')
# axes[0].set_title('(a) Distribution Matching Quality (Wasserstein)', fontsize=10, pad=5)
# axes[0].axhline(1.0, color='#555555', linestyle='--', linewidth=1, alpha=0.8)
# axes[0].text(4.45, 1.0, 'Baseline', va='bottom', ha='right', fontsize=7, color='#555555')
# axes[0].set_ylim(0, 1.9) # 根据 W 数据范围调整
# axes[0].legend_.remove() # 移除默认图例，后面统一加

# # === 子图 2: Acc (Smoothness) ===
# sns.barplot(data=df_acc, y='Normalized Acc', ax=axes[1], **bar_params)

# # 修饰子图 2
# axes[1].set_ylabel('Norm. Smoothness (Acc $\downarrow$)', fontsize=10)
# axes[1].set_xlabel('') # 最下方不需要Label，Tick足矣，或者留空
# axes[1].set_title('(b) Trajectory Smoothness (Acceleration)', fontsize=10, pad=5)
# axes[1].axhline(1.0, color='#555555', linestyle='--', linewidth=1, alpha=0.8)
# axes[1].text(4.45, 1.0, 'Baseline', va='bottom', ha='right', fontsize=7, color='#555555')
# # Acc在Hopper上略有增加(1.2左右)，在HalfCheetah减少(0.8左右)，调整ylim
# axes[1].set_ylim(0, 1.8) 
# axes[1].legend_.remove()

# # --- 6. 全局布局调整 ---

# # 统一图例 (获取第一个子图的句柄和标签)
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
#            ncol=4, frameon=False, fontsize=10)

# # 去除多余边框
# for ax in axes:
#     sns.despine(ax=ax, left=False, bottom=False, top=True, right=True)

# # 调整子图间距
# plt.tight_layout()
# # 留出顶部空间给图例
# plt.subplots_adjust(top=0.9)

# # --- 7. 保存与显示 ---
# plt.savefig('w_acc_comparison.png', bbox_inches='tight', dpi=300)
# # plt.show()