import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# 假设这些类都在当前命名空间或已正确导入
from .polyflow_block import (
    ConstraintEncoder, 
    TrajEncoder, 
    CrossAttentionBlock, 
    EfficientRayShootingLayer, 
    OneWeightDecoder, 
    SinusoidalTimeEmb,
    create_block_cross_attention_mask
)

class PartiallyConstrainedFlowModel(nn.Module):
    """
    部分受限的 Flow Model (支持共享轨迹编码器)。
    
    参数 share_traj_encoder 控制编码策略：
    - True: 使用一个共享的 TrajEncoder 处理完整的 x (full_x_dim)。
            优势：受限流和自由流能看到彼此的信息（例如位置求解可以看到速度信息）。
    - False: 受限部分只看 x_c，自由部分只看 x_u，使用两个独立的 Encoder。
            优势：特征完全解耦，互不干扰。
    """
    def __init__(self, 
                 full_x_dim: int,
                 constrained_idxs: List[int],
                 num_cons: int, 
                 max_seq: int, 
                 embed_dim: int = 128, 
                 share_traj_encoder: bool = True,  # <--- 新增参数
                 cons_begin_seq_idx: int = 1, # <--- 开始受到约束的horizon索引
                 time_invariance_cons: bool = False, # <--- 约束A，b 是否不随horizon变化
                 ray_shooting_method: str = 'hard',
                 ray_shooting_beta: float = 50,
                 # 约束部分特定的参数
                 num_rays: int = 1,
                 num_heads_cons: int = 4, 
                 num_layers_cons: int = 2, 
                 num_heads_weight: int = 4, 
                 num_layers_weight: int = 2,
                 # 通用/轨迹参数
                 num_heads_traj: int = 4, 
                 num_layers_traj: int = 2, 
                 time_embed_scale: float = 1000.0, 
                 use_block_mask_cons: bool = True, 
                 use_block_mask_cross: bool = True, 
                 device: str = 'cuda'):
        
        super().__init__()
        
        self.full_x_dim = full_x_dim
        self.constrained_idxs = constrained_idxs
        self.share_traj_encoder = share_traj_encoder
        self.device = device
        self.cons_begin_seq_idx = cons_begin_seq_idx # 保存起始索引
        self.time_invariance_cons = time_invariance_cons
        self.ray_shooting_method = ray_shooting_method
        self.ray_shooting_beta = ray_shooting_beta
        print(f"PartialPolyFlow Using time invariance cons: {self.time_invariance_cons}   Ray shooting method: {self.ray_shooting_method}  Ray shooting beta: {self.ray_shooting_beta}")
        
        # 1. 维度计算与索引处理
        # ------------------------------------------------------------
        self.c_idxs = torch.tensor(constrained_idxs, dtype=torch.long, device=device)
        
        # 找出未受限的索引
        all_idxs = torch.arange(full_x_dim, device=device)
        mask = torch.ones(full_x_dim, dtype=torch.bool, device=device)
        mask[self.c_idxs] = False
        self.u_idxs = all_idxs[mask]
        
        self.dim_c = len(self.c_idxs)
        self.dim_u = len(self.u_idxs)
        
        self.num_cons = num_cons
        self.max_seq = max_seq
        self.embed_dim = embed_dim
        self.num_rays = num_rays 

        print(f"[Model Init] Full: {full_x_dim} | Cons: {self.dim_c} | Uncons: {self.dim_u} | SharedEncoder: {share_traj_encoder}")

        # 2. 时间编码 (共享)
        # ------------------------------------------------------------
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(dim=embed_dim, scale=time_embed_scale),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(device)

        # 3. 轨迹编码器初始化逻辑 (核心修改)
        # ------------------------------------------------------------
        if self.share_traj_encoder:
            # === 模式 A: 共享编码器 ===
            # 输入为全维度 full_x_dim，产出通用的 latent
            self.traj_encoder = TrajEncoder(
                x_dim=self.full_x_dim,  # 输入完整维度
                embed_dim=embed_dim,
                max_seq=max_seq,
                num_heads=num_heads_traj,
                num_layers=num_layers_traj,
                device=device
            )
            # 为了代码统一，若处于共享模式，就不初始化 separate encoders
            self.traj_encoder_c = None
            self.traj_encoder_u = None
        else:
            # === 模式 B: 独立编码器 ===
            # 分别对 x_c 和 x_u 进行编码
            if self.dim_c > 0:
                self.traj_encoder_c = TrajEncoder(
                    x_dim=self.dim_c, # 只输入受限维度
                    embed_dim=embed_dim,
                    max_seq=max_seq,
                    num_heads=num_heads_traj,
                    num_layers=num_layers_traj,
                    device=device
                )
            if self.dim_u > 0:
                self.traj_encoder_u = TrajEncoder(
                    x_dim=self.dim_u, # 只输入自由维度
                    embed_dim=embed_dim,
                    max_seq=max_seq,
                    num_heads=num_heads_traj,
                    num_layers=num_layers_traj,
                    device=device
                )

        # 4. 构建【受限部分】的后处理组件
        # ------------------------------------------------------------
        if self.dim_c > 0:
            remain_length = self.max_seq - self.cons_begin_seq_idx
            if self.time_invariance_cons:
                # 因为约束形式不随horizon变化，因此没有必要在horizon上进行attn
                constrained_horizon_length = 1  
            else:
                constrained_horizon_length = remain_length

            self.constraint_encoder = ConstraintEncoder(
                x_dim=self.dim_c, 
                embed_dim=embed_dim, 
                num_cons=num_cons,
                max_seq=constrained_horizon_length, 
                num_heads=num_heads_cons, 
                num_layers=num_layers_cons,
                use_block_mask=use_block_mask_cons, 
                device=device
            )

            # Cross Attention
            self.cross_attn = CrossAttentionBlock(
                embed_dim=embed_dim, num_heads=num_heads_traj
            ).to(device)
            
            # 预计算完整的 Mask
            if use_block_mask_cross:
                self.full_cross_attn_mask = create_block_cross_attention_mask(
                    query_len=max_seq, key_len=max_seq*num_cons, n=1, m=num_cons, T=max_seq, device=device
                )
            else:
                self.full_cross_attn_mask = None

            # Heads
            self.ray_mlp = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, self.num_rays * self.dim_c)
            ).to(device)

            self.ray_shooter = EfficientRayShootingLayer(method=self.ray_shooting_method, beta=self.ray_shooting_beta).to(device)

            self.weight_decoder = OneWeightDecoder(
                x_dim=self.dim_c, 
                embed_dim=embed_dim, 
                max_seq=remain_length, 
                num_rays=self.num_rays,
                num_heads=num_heads_weight, 
                num_layers=num_layers_weight, 
                device=device
            )

        # 5. 构建【自由部分】的输出头
        # ------------------------------------------------------------
        if self.dim_u > 0:
            self.unconstrained_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, self.dim_u)
            ).to(device)

    def forward(self, x, cond, t, A, b):
        """
        x: [B, S, full_x_dim] 
        t: [B,]
        A: [B, S, num_cons, dim_c] 位置与索引参数对应
        b: [B, S, num_cons,]
        return:
            [B, S, full_x_dim]
            [B, S, num_rays, dim_c]
            [B, S, num_rays]
        """
        batch_size = x.size(0)
        start_idx = self.cons_begin_seq_idx
        
        # 0. 形状预处理
        if x.dim() == 2:
            x = x.view(batch_size, self.max_seq, self.full_x_dim)
            
        # 1. 数据拆分 (无论是否共享encoder，RayShooting都需要物理上的 x_c)
        x_c = x.index_select(-1, self.c_idxs)
        x_u = x.index_select(-1, self.u_idxs)
        
        t_emb = self.time_mlp(t)

        # 2. 轨迹特征编码 (Encoding)
        # ------------------------------------------------------------
        if self.share_traj_encoder:
            # [共享模式]
            # 输入完整的 x，得到包含全局信息的 latent
            # latent shape: [B, S, embed_dim]
            global_lat = self.traj_encoder(x, t_emb)
            
            # 两个流使用相同的 latent
            traj_lat_c = global_lat
            traj_lat_u = global_lat
        else:
            # [独立模式]
            # 分别编码
            traj_lat_c = self.traj_encoder_c(x_c, t_emb) if self.dim_c > 0 else None
            traj_lat_u = self.traj_encoder_u(x_u, t_emb) if self.dim_u > 0 else None

        # 2. 准备输出容器 (Padding with Zeros)
        # ------------------------------------------------------------
        
        delta_x_c_future = None
        boundary_vectors_full = torch.zeros(batch_size, self.max_seq, self.num_rays, self.dim_c, device=self.device)
        weights_full = torch.zeros(batch_size, self.max_seq, self.num_rays, device=self.device)
        
        # 3. 受限部分处理 (Sliced Execution)
        # ------------------------------------------------------------
        if self.dim_c > 0:
            # === [Slicing] ===
            # 只取出需要生成的未来时间步
            # Shape变为: [B, S_future, ...]
            traj_lat_c_active = traj_lat_c[:, start_idx:, :]
            A_active = A[:, start_idx:, :, :]
            b_active = b[:, start_idx:, :]
            x_c_active = x_c[:, start_idx:, :] # Ray Shooting 需要未来的中心点

            # 3.1 编码约束 (只编码 active 部分，节省计算)
            # 因为 ConstraintEncoder 内部是 Block-Diagonal 的，所以切片输入是安全的
            # 输出: [B, S_future, num_cons, embed_dim]
            if self.time_invariance_cons:
                # 只输入一个horizon长度即可, 返回 (batch, 1, cons_num, embed)
                cons_lat_active = self.constraint_encoder(A_active[:, 0:1, :, :], b_active[:, 0:1, :])
            else:
                cons_lat_active = self.constraint_encoder(A_active, b_active)
            
            if self.time_invariance_cons:
                # 运行 Attention
                # Query: [B, S_future, E], Key: [B, M, E]
                cons_lat_flat = cons_lat_active.reshape(batch_size, self.num_cons, self.embed_dim)
                e_c_active, _ = self.cross_attn(
                    query=traj_lat_c_active, 
                    key_value=cons_lat_flat, 
                )
            else:
                # 3.2 Cross Attention (融合)
                # 展平 Constraint: [B, S_future * num_cons, embed_dim]
                S_future = traj_lat_c_active.size(1)
                cons_lat_flat = cons_lat_active.reshape(batch_size, S_future * self.num_cons, self.embed_dim)
                
                # 处理 Mask: 切割预计算的 Mask
                if self.full_cross_attn_mask is not None:
                    # Query 维度: start ~ end
                    # Key 维度: start*num_cons ~ end*num_cons
                    # 注意：create_block_cross_attention_mask 必须是 block 对齐的
                    active_mask = self.full_cross_attn_mask[
                        start_idx : , 
                        start_idx * self.num_cons : 
                    ]
                else:
                    active_mask = None

                # 运行 Attention
                # Query: [B, S_future, E], Key: [B, S_future*M, E]
                e_c_active, _ = self.cross_attn(
                    query=traj_lat_c_active, 
                    key_value=cons_lat_flat, 
                    attn_mask=active_mask
                )
            
            S_future = traj_lat_c_active.size(1)
            # 3.3 Ray Generation & Shooting (仅对未来)
            rays_active = self.ray_mlp(e_c_active).view(batch_size, S_future, self.num_rays, self.dim_c)
            rays_active = F.normalize(rays_active, p=2, dim=-1)
            
            # 几何求交 (Expensive Operation)
            # 这里的 A_active 和 x_c_active 都是切片后的，因此 RayShooting 仅计算未来的碰撞
            bound_vec_active = self.ray_shooter(x_c_active, rays_active, A_active, b_active)
            
            # 3.4 Weights & Aggregation
            # WeightDecoder 内部可能含 Self-Attention (AdaLNBlock)，输入切片后的 latent 是合理的
            # 如果 WeightDecoder 用了 PE (Positional Encoding)，可能需要注意位置偏移?
            # 这里的 OneWeightDecoder 里 input_proj 后加了 traj_embed (fusion)。
            # 注意：OneWeightDecoder 里的 attn_blocks 使用 AdaLN，没有显式的绝对位置编码(PE是在TrajEncoder加的)。
            # 只要 TrajEncoder 传递过来的 embeddings 包含了正确的位置信息(它确实包含)，这里就没问题。
            weights_active = self.weight_decoder(bound_vec_active, traj_embed=e_c_active, t_embed=t_emb)
            
            # 计算 delta
            delta_x_c_active = torch.einsum('bsk,bskd->bsd', weights_active, bound_vec_active)
            
            # === [Fill Back] ===
            # 将计算结果填回 full sequence 容器
            boundary_vectors_full[:, start_idx:, :, :] = bound_vec_active
            weights_full[:, start_idx:, :] = weights_active
            delta_x_c_future = delta_x_c_active # 暂存 slicing 部分

        # 4. 自由部分处理 (Sliced Execution)
        # ------------------------------------------------------------
        delta_x_u_future = None
        if self.dim_u > 0:
            # 切片 Latent
            traj_lat_u_active = traj_lat_u
            # MLP
            delta_x_u_future = self.unconstrained_head(traj_lat_u_active)


        # 5. 最终重组 (Recombination)
        # ------------------------------------------------------------
        # 结果张量初始化为全 0
        delta_x_final = torch.zeros(batch_size, self.max_seq, self.full_x_dim, device=self.device)
        
        
        if self.dim_c > 0:
            # 只有 start_idx 之后的部分有值
            delta_x_final[:, start_idx:, :].index_copy_(-1, self.c_idxs, delta_x_c_future)
        
        if self.dim_u > 0:
            # 所有 horizon 都有值
            delta_x_final[:, :, :].index_copy_(-1, self.u_idxs, delta_x_u_future)


        return delta_x_final, boundary_vectors_full, weights_full