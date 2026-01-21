import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog

def plot_polytope_3d(A, b, bounds_limit=10.0):
    """
    绘制由 Ax <= b 定义的 3D 多面体，并判断其是否有界。
    
    参数:
        A: (num_cons, 3) 约束矩阵
        b: (num_cons,) 约束向量
        bounds_limit: float, 用于可视化无界多面体时的截断范围 [-limit, limit]
    """
    
    # --- 第一步：寻找内部可行点 (Chebyshev Center) ---
    # 我们需要找到一个点 x 和半径 r，使得以 x 为球心、r 为半径的球体完全在多面体内。
    # 最大化 r，使得 a_i^T * x + ||a_i|| * r <= b_i
    # scipy.linprog 是求最小化，所以我们要最小化 -r
    
    norm_A = np.linalg.norm(A, axis=1)
    # 构造线性规划矩阵: 变量为 [x0, x1, x2, r]
    # 约束形式: A*x + ||A||*r <= b
    c = np.array([0, 0, 0, -1])  # 目标函数: max r -> min -r
    A_lp = np.column_stack((A, norm_A))
    
    # 求解
    res = linprog(c, A_ub=A_lp, b_ub=b, bounds=(None, None), method='highs')
    
    if not res.success or res.x[3] <= 1e-9:
        print("错误: 该区域为空集 (Infeasible) 或没有严格内部点 (Degenerate)。无法绘图。")
        return

    interior_point = res.x[:3]
    
    # --- 第二步：判断是否封闭 (Boundedness Check) ---
    # 我们沿 6 个轴向方向 (±x, ±y, ±z) 检查线性规划是否有界
    is_bounded = True
    directions = np.eye(3)
    directions = np.vstack((directions, -directions)) # 6个方向
    
    for direction in directions:
        # 只需要判断是否有解，目标是最小化 direction * x
        # 注意：linprog 返回 status 3 代表 unbounded
        res_bound = linprog(c=direction, A_ub=A, b_ub=b, bounds=(None, None), method='highs')
        if res_bound.status == 3: # 3 means Unbounded in scipy
            is_bounded = False
            break
            
    status_str = "封闭 (Bounded)" if is_bounded else "无界 (Unbounded)"
    print(f"多面体状态: {status_str}")

    # --- 第三步：准备绘图数据 ---
    # 如果是无界的，或者是为了防止绘图范围过大，我们添加人为的包围盒限制
    # 添加 ±bounds_limit 的限制: x <= L, x >= -L, ...
    
    # 原始半空间: Ax - b <= 0. Scipy 需要 [A, -b] 格式
    halfspaces = np.column_stack((A, -b))
    
    # 添加包围盒约束 (Box constraints)
    box_halfspaces = np.array([
        [1, 0, 0, -bounds_limit],  # x <= L
        [-1, 0, 0, -bounds_limit], # -x <= L -> x >= -L
        [0, 1, 0, -bounds_limit],
        [0, -1, 0, -bounds_limit],
        [0, 0, 1, -bounds_limit],
        [0, 0, -1, -bounds_limit],
    ])
    
    # 将原始约束和包围盒约束合并
    combined_halfspaces = np.vstack((halfspaces, box_halfspaces))
    
    # --- 第四步：计算半空间交集 ---
    try:
        hs = HalfspaceIntersection(combined_halfspaces, interior_point)
        verts = hs.intersections
    except Exception as e:
        print(f"计算交集时出错 (可能内部点选取得不够好或包围盒太小): {e}")
        return

    # --- 第五步：利用凸包提取面并绘图 ---
    # HalfspaceIntersection 给出了顶点，但没有给出面的连接顺序
    # 使用 ConvexHull 对顶点进行三角剖分
    
    # 可能会产生非常接近的点，去重
    # verts = np.unique(verts.round(decimals=5), axis=0) # 可选
    
    if len(verts) < 4:
        print("顶点数量不足以构成3D多面体。")
        return

    hull = ConvexHull(verts)
    
    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制面
    # hull.simplices 包含了构成凸包表面的三角形顶点索引
    triangles = verts[hull.simplices]
    
    poly3d = Poly3DCollection(triangles, alpha=0.5, edgecolor='k', linewidths=0.5)
    poly3d.set_facecolor('cyan')
    ax.add_collection3d(poly3d)
    
    # 绘制内部点（参考）
    ax.scatter(interior_point[0], interior_point[1], interior_point[2], c='r', marker='o', s=50, label='Interior Point')
    
    # 设置坐标轴范围
    margin = 1.0
    ax.set_xlim(verts[:,0].min()-margin, verts[:,0].max()+margin)
    ax.set_ylim(verts[:,1].min()-margin, verts[:,1].max()+margin)
    ax.set_zlim(verts[:,2].min()-margin, verts[:,2].max()+margin)
    
    ax.set_title(f"Polytope Visualization ({status_str})\nBox Limit: [-{bounds_limit}, {bounds_limit}]")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    # plt.show()
    plt.savefig("polytope.png")

# --- 测试用例 ---

# 案例 1: 一个封闭的立方体 (Bounded)
# x <= 1, -x <= 1 (即 x >= -1), 同理 y, z
print("--- 案例 1: 封闭立方体 ---")
A1 = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1]
])
b1 = np.array([1, 1, 1, 1, 1, 1])
plot_polytope_3d(A1, b1)

# # 案例 2: 一个无界的锥体 (Unbounded)
# # z >= 0, z >= |x| + |y| 的某种近似线性形式
# # 例如: -z <= 0, x - z <= 0, -x - z <= 0, y - z <= 0, -y - z <= 0
# print("\n--- 案例 2: 无界金字塔锥 ---")
# A2 = np.array([
#     [0, 0, -1],  # -z <= 0 -> z >= 0
#     [1, 0, -1],  # x - z <= 0 -> z >= x
#     [-1, 0, -1], # -x - z <= 0 -> z >= -x
#     [0, 1, -1],  # y - z <= 0 -> z >= y
#     [0, -1, -1]  # -y - z <= 0 -> z >= -y
# ])
# b2 = np.array([0, 0, 0, 0, 0])
# # 这是一个倒立的无限金字塔，只在Z正方向无限延伸
# plot_polytope_3d(A2, b2, bounds_limit=5.0)