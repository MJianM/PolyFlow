from typing import List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ot
import time

class RayShootingLayer(nn.Module):
    """
    对应图中中间上方: ray-shooting 计算边界点
    数学原理:
    给定中心 c, 方向 d。射线为 r = c + t * d (t > 0)。
    需要满足 A(c + t*d) <= b  =>  A*c + t*(A*d) <= b  => t*(A*d) <= b - A*c
    令 slack = b - A*c (即当前点距离边界的余量，若 c 在可行域内，slack >= 0)
    令 projection = A*d
    则需满足 t * projection <= slack
    对于每个约束 i:
       1. 若 projection_i <= 0: 射线背离或平行于约束，距离为无穷大。
       2. 若 projection_i > 0: 射线朝向约束，t <= slack_i / projection_i
    最终 t = min_i (slack_i / projection_i)，仅考虑 projection_i > 0 的情况。
    """
    def forward(self, c, directions, A, b):
        # c (center): [batch, n]
        # directions: [batch, k, n] (k = n+1)
        # A: [batch, M, n]
        # b: [batch, M]
        
        # 1. 计算 Slack (余量): s = b - Ac
        # [batch, M]
        Ax = torch.einsum('bmn,bn->bm', A, c)
        slack = b - Ax 
        
        # 2. 计算投影: Ad = A * d
        # [batch, k, M]
        Ad = torch.einsum('bmn,bkn->bkm', A, directions)
        
        # 3. 计算 intersection times (t)
        # 为了数值稳定性，加一个小 epsilon
        EPS = 1e-8
        
        # 我们只关心 Ad > 0 的约束（即射线朝向的墙）
        # t = slack / Ad
        # 这里需要广播: slack [batch, 1, M], Ad [batch, k, M]
        t = slack.unsqueeze(1) / (Ad + EPS)
        
        # 创建掩码: 只有 Ad > 0 的才是有意义的物理碰撞
        mask = Ad > 1e-6
        
        # 将无效的 t (Ad <= 0) 设为无穷大，这样 min 的时候会被忽略
        t_masked = t.clone()
        t_masked[~mask] = float('inf')
        
        # 4. 找到最近的边界 (Ray Shooting 核心)
        # alpha: [batch, k, 1]
        alpha, _ = t_masked.min(dim=-1, keepdim=True)
        
        # 处理数值问题：如果点在多面体外或者数值误差导致没有正向交点，alpha 可能会是 inf
        # 在 Flow Matching 训练中通常假设 x_t 在多面体内。
        # 这里做一个 clamp 防止梯度爆炸
        alpha = torch.clamp(alpha, min=0.0, max=1e5) 
        
        # 5. 计算边界点 V = c + alpha * d
        # directions: [batch, k, n], alpha: [batch, k, 1]
        # boundary_points = c.unsqueeze(1) + alpha * directions
        boundary_points = alpha * directions  # 输出变化量
        
        return boundary_points


# class EfficientRayShootingLayer(nn.Module):
#     """
#     针对块对角约束优化的 Ray Shooting
#     输入维度不再是展平的 2T，而是保持 (Batch, T, 2)
#     """
#     def __init__(self, method='hard'):
#         super().__init__()

#         self.method = method

#         assert self.method in ['hard', 'softmin', 'boltzmann']

#     def forward(self, c, directions, A, b):

#         if self.method == 'hard':
#             return self.forward_hard(c, directions, A, b)
#         elif self.method == 'softmin':
#             return self.forward_softmin(c, directions, A, b)
#         elif self.method == 'boltzmann':
#             return self.forward_boltzmann(c, directions, A, b)
#         else:
#             raise NotImplementedError

#     def forward_hard(self, c, directions, A, b):
#         """
#         c (center): [batch, T, x_dim]  <-- 也就是当前的 trajectory x_t
#         directions: [batch, T, k, x_dim] (k = num_vertices)
#         A: [batch, T, m, x_dim] <-- 这里的 m 是单步的约束数 (例如4)
#         b: [batch, T, m]
#         """
#         # 1. 计算 Slack: s = b - Ac
#         # Einsum: B(atch), T(ime), M(constraints), D(im)
#         # Ac: [batch, T, m]
#         Ax = torch.einsum('btmd,btd->btm', A, c)
#         slack = b - Ax 
        
#         # 2. 计算投影: Ad = A * d
#         # Ad: [batch, T, k, m]
#         Ad = torch.einsum('btmd,btkd->btkm', A, directions)
        
#         # 3. 计算 intersection times (alpha)
#         EPS = 1e-6
        
#         # 我们只关心 Ad > 0 的约束 (射线朝向的墙)
#         # alpha = slack / Ad
#         # Broadcast slack: [batch, T, 1, m]
#         t = slack.unsqueeze(2) / (Ad + EPS)
        
#         # 掩码: 只有 Ad > EPS 的才是有效的物理碰撞
#         mask = Ad > EPS
        
#         # 将无效的 t 设为无穷大
#         t_masked = t.clone()
#         t_masked[~mask] = float('inf')
        
#         # 4. 找到最近的边界 (Ray Shooting 核心)
#         # alpha: [batch, T, k, 1]
#         alpha, _ = t_masked.min(dim=-1, keepdim=True)
        
#         # 数值稳定性处理：防止 alpha 为 inf (射线在多面体内部没碰到墙，虽然理论上不可能)
#         # 或者点已经在多面体外导致 slack < 0
#         alpha = torch.clamp(alpha, min=0.0, max=1e3) 
        
#         # 5. 计算边界点向量 (相对于中心的位移)
#         # directions: [batch, T, k, x_dim], alpha: [batch, T, k, 1]
#         boundary_vectors = alpha * directions 
        
#         return boundary_vectors
    
#     def forward_softmin(self, c, directions, A, b):
#         pass

#     def forward_boltzmann(self, c, directions, A, b):
#         pass



class EfficientRayShootingLayer(nn.Module):
    """
    """
    def __init__(self, method='hard', beta=50.0):
        """
        Args:
            method: 'hard', 'softmin', or 'boltzmann'
            beta: 温度系数 (Temperature scaling). 
                  用于控制 softmin 和 boltzmann 的平滑程度。
                  Beta 越大 -> 越接近 hard min (梯度越陡)。
                  Beta 越小 -> 越平滑 (梯度越丰富，但数值误差变大)。
        """
        super().__init__()
        self.method = method
        self.beta = beta # 建议范围: 10.0 ~ 100.0

        assert self.method in ['hard', 'softmin', 'boltzmann']

    def forward(self, c, directions, A, b):
        if self.method == 'hard':
            return self.forward_hard(c, directions, A, b)
        elif self.method == 'softmin':
            return self.forward_softmin(c, directions, A, b)
        elif self.method == 'boltzmann':
            return self.forward_boltzmann(c, directions, A, b)
        else:
            raise NotImplementedError

    def _compute_raw_t(self, c, directions, A, b):
        """
        辅助函数：计算所有平面的物理碰撞时间 t 和有效性 mask
        这是所有方法共用的第一步
        Returns:
            t: [batch, T, k, m] (未处理的原始时间，分母加了EPS)
            mask: [batch, T, k, m] (布尔值，True表示有效碰撞)
        """
        # 1. 计算 Slack: s = b - Ac -> [batch, T, m]
        Ax = torch.einsum('btmd,btd->btm', A, c)
        slack = b - Ax 
        
        # 2. 计算投影: Ad = A * d -> [batch, T, k, m]
        Ad = torch.einsum('btmd,btkd->btkm', A, directions)
        
        # 3. 计算 intersection times (t)
        EPS = 1e-6
        # Broadcast slack: [batch, T, 1, m]
        t = slack.unsqueeze(2) / (Ad + EPS)
        
        # 4. 掩码: 只有 Ad > 0 (射线朝向墙) 才是有效的物理碰撞
        mask = Ad > EPS
        
        return t, mask

    def forward_hard(self, c, directions, A, b):
        t, mask = self._compute_raw_t(c, directions, A, b)
        
        # 将无效的 t 设为无穷大，以便 min 操作忽略它们
        t_masked = t.clone()
        t_masked[~mask] = float('inf')
        
        # t_masked = torch.where(mask, t, torch.tensor(1e9, device=t.device))

        # 找到最近的边界 (Min)
        # alpha: [batch, T, k, 1]
        alpha, _ = t_masked.min(dim=-1, keepdim=True)
        
        # 数值稳定性处理
        alpha = torch.clamp(alpha, min=0.0, max=1e3) 
        
        # 计算位移
        return alpha * directions 
    
    def forward_softmin(self, c, directions, A, b):
        """
        Method 1: SoftMin (LogSumExp)
        公式: alpha = - (1/beta) * log( sum( exp(-beta * t_i) ) )
        优点: 全局光滑近似 Min 操作，梯度在顶点处连续。
        """
        t, mask = self._compute_raw_t(c, directions, A, b)
        
        # 1. Masking
        # 对于无效的墙 (Ad <= 0)，物理上 t 应该是无穷大。
        # 在 SoftMin 中，exp(-beta * inf) = 0，这意味着它们不会贡献到 sum 中。
        # 我们用一个很大的数 (1e9) 代替 inf，保证数值稳定。
        t_masked = t.clone()
        t_masked[~mask] = 1e9 
        
        # 2. LogSumExp 计算
        # 注意: LSE(x) 计算 log(sum(exp(x)))
        # 我们需要计算 -1/beta * log(sum(exp(-beta * t)))
        # 所以传入 LSE 的应该是 -beta * t
        neg_scaled_t = -self.beta * t_masked
        
        # dim=-1 是约束维度 (m)
        # alpha: [batch, T, k, 1]
        alpha = -torch.logsumexp(neg_scaled_t, dim=-1, keepdim=True) / self.beta
        
        # 3. 数值稳定性 (Softmin 可能会因为近似产生微小的负数或过大值)
        alpha = torch.clamp(alpha, min=0.0, max=1e3)
        
        return alpha * directions

    def forward_boltzmann(self, c, directions, A, b):
        """
        Method 2: Boltzmann / Softmax Weighted Average
        公式: alpha = sum( w_i * t_i ), where w_i = Softmax(-beta * t_i)
        优点: 这是基于概率的期望值，不仅仅关注最近的墙，而是对所有墙的距离做加权平均。
        """
        t, mask = self._compute_raw_t(c, directions, A, b)
        
        # 1. Masking
        # 同样，对于无效墙，我们需要它们的权重 w_i 接近 0。
        # Softmax(-beta * 1e9) ≈ 0
        t_masked = t.clone()
        t_masked[~mask] = 1e9
        
        # 2. 计算权重
        # weights: [batch, T, k, m]
        # 注意：这里 dim=-1，在约束维度上做归一化
        weights = F.softmax(-self.beta * t_masked, dim=-1)
        
        # 3. 加权求和
        # alpha = sum(w_i * t_i)
        # 注意：这里我们使用原始的 t 进行相乘，还是 t_masked？
        # 理论上 w_i 对应的无效位置已经是 0 了，所以用 t_masked (含有1e9) 可能会导致 0 * 1e9 的数值误差。
        # 最安全的方法是再乘一次 mask 确保无效项绝对为 0，或者直接用原始 t (只要 t 不含 nan/inf)
        # 由于 _compute_raw_t 中分母加了 EPS，原始 t 是安全的数值。
        
        # 此外，为了完全排除无效墙的干扰（防止 t 是负数的情况干扰结果），
        # 我们最好只对有效部分求和。
        # 虽然 weights 已经处理了，但双重保险：
        t_safe = t.clone()
        t_safe[~mask] = 0.0 # 设为0，反正权重也是0，相乘不影响结果
        
        alpha = (weights * t_safe).sum(dim=-1, keepdim=True)
        
        # 4. 数值稳定性
        alpha = torch.clamp(alpha, min=0.0, max=1e3)
        
        return alpha * directions


def create_block_diagonal_mask(T, block_size=4, device='cpu'):
    """
    创建块对角掩码，每个块大小为block_size×block_size
    总序列长度 = T * block_size
    
    Args:
        T: 块的数量
        block_size: 每个块的大小（默认4）
        device: 设备
    
    Returns:
        mask: (L, L)的布尔掩码，False表示允许注意力
    """
    L = T * block_size
    mask = torch.ones(L, L, dtype=torch.bool, device=device)
    
    # 为每个块创建允许的注意力区域
    for i in range(T):
        start = i * block_size
        end = (i + 1) * block_size
        mask[start:end, start:end] = False  # 块内允许注意力
    
    return mask


def create_block_cross_attention_mask(query_len, key_len, n, m, T, device='cuda'):
    """
    创建块状交叉注意力掩码
    
    Args:
        query_len: n*T, query序列的总长度
        key_len: m*T, key/value序列的总长度
        n: 每个query块的长度（不含额外token）
        m: 每个key/value块的长度
        T: 块的数量
    """
    # 验证输入尺寸
    assert query_len == n * T, f"query_len should be (n+1)*T = {n*T}, but got {query_len}"
    assert key_len == m * T, f"key_len should be m*T = {m*T}, but got {key_len}"
    
    # 创建初始掩码（True表示被屏蔽，False表示允许注意力）
    attn_mask = torch.ones(query_len, key_len, dtype=torch.bool, device=device)
    
    # 为每个块设置允许的注意力区域
    for t in range(T):
        # 第t个query块的范围
        query_start = t * n
        query_end = (t + 1) * n
        
        # 第t个key/value块的范围
        key_start = t * m
        key_end = (t + 1) * m
        
        # 允许当前query块关注对应的key块
        attn_mask[query_start:query_end, key_start:key_end] = False
    
    return attn_mask

def ot_minibatch_coupling(x0, x1):
    """
    使用 POT 库计算 Batch 内的最优传输配对。
    
    Args:
        x0: Source samples (Noise), [Batch, Dim]
        x1: Target samples (Data), [Batch, Dim]
        
    Returns:
        x0_new: 重排后的 x0 (或者保持 x0 不变，重排 x1，效果一样)
        x1_new: 重排后的 x1，使得 (x0, x1_new) 构成最优传输对
    """
    # POT 通常在 CPU 上运行 (虽然也支持 backend，但 batch 较小时 CPU 够快且稳定)
    device = x0.device
    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()
    
    batch_size = x0.shape[0]
    
    # 1. 构造均匀分布权重 (Uniform weights)
    # 假设每个样本权重相等 a = b = 1/batch_size
    a = np.ones((batch_size,)) / batch_size
    b = np.ones((batch_size,)) / batch_size
    
    # 2. 计算代价矩阵 (Cost Matrix)
    # 使用欧氏距离平方: ||x - y||^2
    M = ot.dist(x0_np, x1_np, metric='sqeuclidean')
    
    # 3. 求解 EMD (Earth Mover's Distance)
    # 返回的是传输矩阵 gamma (G)
    # G[i, j] 非 0 表示 x0[i] 应该移动到 x1[j]
    G = ot.emd(a, b, M)
    
    # 4. 获取配对索引
    # 由于权重是均匀的，对于每个 i，G[i, :] 中只有一个 j 是非零的 (值为 1/N)
    # 使用 argmax 找到这个 j
    # axis=1 表示对每一行(x0的每个点)找对应的列(x1的索引)
    pair_indices = np.argmax(G, axis=1)
    
    # 5. 转换为 Tensor 并重排数据
    pair_indices = torch.from_numpy(pair_indices).to(device)
    
    # 保持 x0 不变，重排 x1
    x1_aligned = x1[pair_indices]
    
    return x0, x1_aligned


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation_class: Type[nn.Module] = nn.ReLU,
    dropout: float = 0.0,
    use_bias: bool = True
) -> nn.Sequential:
    """
    构建一个多层感知机 (MLP) 并返回 nn.Sequential 容器。
    """
    layers = []
    current_dim = input_dim
    
    # 构建隐藏层
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim, bias=use_bias))
        layers.append(activation_class())  # 添加激活函数
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = h_dim
        
    # 构建输出层 (通常输出层不加激活函数和 Dropout)
    layers.append(nn.Linear(current_dim, output_dim, bias=use_bias))
    
    return nn.Sequential(*layers)

def set_all_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def benchmark_ray_shooting(device='cuda', n_runs=1000):
    """
    Args:
        device: 'cuda' or 'cpu'
        n_runs: 重复推断次数用于计算平均值
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable, switching to CPU.")
        device = 'cpu'
    
    print(f"--- Starting Benchmark on {device.upper()} ---")

    # 1. 模拟数据参数
    # 假设场景：Batch=64, 轨迹长度T=20, 32条射线, 12个平面(例如一个凸多面体), 3维空间
    B, T, K, M, D = 64, 20, 32, 12, 3
    
    print(f"Input Shapes: [B={B}, T={T}, K={K}, M={M}, D={D}]")
    print(f"Total Rays per pass: {B*T*K:,}")
    
    # 2. 创建 Dummy Tensors
    torch.manual_seed(42)
    c = torch.randn(B, T, D).to(device)
    directions = torch.randn(B, T, K, D).to(device)
    directions = F.normalize(directions, dim=-1) # 归一化方向
    A = torch.randn(B, T, M, D).to(device)
    b = torch.abs(torch.randn(B, T, M)).to(device) # 保证 b 为正，尽量让原点在内部

    methods = ['hard', 'softmin', 'boltzmann']
    
    print(f"{'Method':<15} | {'Mean Time (ms)':<15} | {'Std Dev (ms)':<15} | {'FPS (batches/s)':<15}")
    print("-" * 70)

    for method in methods:
        # 初始化模型
        model = EfficientRayShootingLayer(method=method, beta=50.0).to(device)
        model.eval()

        # 3. 预热 (Warm-up)
        # GPU 需要预热来分配缓存和进行初始的编译优化
        with torch.no_grad():
            for _ in range(50):
                _ = model(c, directions, A, b)
        
        # 4. 计时
        timings = []
        
        # 使用 CUDA Event 进行精确计时
        if device == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            with torch.no_grad():
                for _ in range(n_runs):
                    start_event.record()
                    _ = model(c, directions, A, b)
                    end_event.record()
                    # 必须等待 GPU 完成
                    torch.cuda.synchronize() 
                    timings.append(start_event.elapsed_time(end_event)) # 返回毫秒
        else:
            # CPU 计时
            with torch.no_grad():
                for _ in range(n_runs):
                    start = time.perf_counter()
                    _ = model(c, directions, A, b)
                    end = time.perf_counter()
                    timings.append((end - start) * 1000) # 转换为毫秒

        # 5. 统计
        timings = np.array(timings)
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        fps = 1000 / mean_time # 1000ms / 平均耗时

        print(f"{method:<15} | {mean_time:.4f} ms       | {std_time:.4f} ms       | {fps:.2f}")


if __name__=='__main__':

    # mask = create_block_diagonal_mask(T=2, block_size=3)
    # print(mask)

    benchmark_ray_shooting(device='cuda:0', n_runs=1000)