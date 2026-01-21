import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

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

class QuadrupedConstrainedFlowModel(nn.Module):
    """
    专为四足机器人设计的受限 Flow Model。
    
    特性:
    1. 混合输出: 针对 Horizon=0 且 Contact=1 的腿部动作应用几何约束求解；其他情况使用无约束预测。
    2. 高效计算: 利用 Batch * 4 的方式并行处理四条腿的约束，共享主干 TrajEncoder。
    """
    def __init__(self, 
                 act_dim: int, 
                 obs_dim: int,
                 leg_act_dim: int = 3, # 每条腿的动作维度 (例如 x,y,z 力或位置)
                 num_legs: int = 4,
                 num_cons_per_leg: int = 5, # 每条腿的最大约束面数量
                 max_seq: int = 100, 
                 embed_dim: int = 128,
                 # 约束部分参数
                 ray_shooting_method: str = 'hard',
                 ray_shooting_beta: float = 50,
                 num_rays: int = 1,
                 num_heads_cons: int = 4, 
                 num_layers_cons: int = 2, 
                 num_heads_weight: int = 4, 
                 num_layers_weight: int = 2,
                 # 轨迹编码参数
                 num_heads_traj: int = 4, 
                 num_layers_traj: int = 4, 
                 time_embed_scale: float = 1000.0, 
                 device: str = 'cuda'):
        
        super().__init__()
        
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.full_dim = act_dim + obs_dim
        self.leg_act_dim = leg_act_dim
        self.num_legs = num_legs
        self.num_cons = num_cons_per_leg
        self.max_seq = max_seq
        self.embed_dim = embed_dim
        self.num_rays = num_rays
        self.device = device
        
        # 假设动作向量的前 (num_legs * leg_act_dim) 维对应四条腿
        # 例如 12维 action: [leg1_x, leg1_y, leg1_z, leg2_x, ...]
        self.total_leg_dim = num_legs * leg_act_dim
        assert self.total_leg_dim <= act_dim, "Leg dimensions exceed total action dimensions"

        # 1. 全局轨迹编码器 (共享)
        # ------------------------------------------------------------
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(dim=embed_dim, scale=time_embed_scale),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(device)

        self.traj_encoder = TrajEncoder(
            x_dim=self.full_dim, 
            embed_dim=embed_dim,
            max_seq=max_seq,
            num_heads=num_heads_traj,
            num_layers=num_layers_traj,
            device=device
        )

        # 2. 无约束输出头 (负责 Horizon > 0 的部分 以及 Contact=0 的腿)
        # ------------------------------------------------------------
        self.unconstrained_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.full_dim)
        ).to(device)

        # 3. 受限部分组件 (仅针对 Horizon=0, Contact=1)
        # ------------------------------------------------------------
        # 注意：这里的 x_dim 是单条腿的维度 (通常为3)
        self.constraint_encoder = ConstraintEncoder(
            x_dim=self.leg_act_dim, 
            embed_dim=embed_dim, 
            num_cons=num_cons_per_leg,
            max_seq=1, # 约束求解只发生在 Horizon 0, 这里的 seq 为 1
            num_heads=num_heads_cons, 
            num_layers=num_layers_cons,
            use_block_mask=True, # 必须为True，因为我们要在 batch*4 维度下处理
            device=device
        )

        # 融合 Traj 特征和 Constraint 特征
        self.cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim, num_heads=num_heads_traj
        ).to(device)

        self.ray_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.num_rays * self.leg_act_dim)
        ).to(device)

        self.ray_shooter = EfficientRayShootingLayer(
            method=ray_shooting_method, beta=ray_shooting_beta
        ).to(device)

        # 计算几何权重
        self.weight_decoder = OneWeightDecoder(
            x_dim=self.leg_act_dim, 
            embed_dim=embed_dim, 
            max_seq=1, 
            num_rays=self.num_rays,
            num_heads=num_heads_weight, 
            num_layers=num_layers_weight, 
            device=device
        )

        # === [新增] 腿部身份编码 ===
        # 用于区分当前处理的是哪一条腿 (FL, FR, RL, RR)
        self.leg_embedding = nn.Embedding(num_legs, embed_dim).to(device)
        # 初始化权重 (可选，通常正态分布初始化有助于收敛)
        nn.init.normal_(self.leg_embedding.weight, std=0.02)

    def forward(self, x, cond, t, A, b, contact):
        """
        Args:
            x: [B, S, act_dim + obs_dim]
            t: [B,]
            A: [B, S, 4, num_cons, 3] - 4表示四条腿，3表示约束空间维度
            b: [B, S, 4, num_cons]
            contact: [B, S, 4] - 1 表示触地(受限), 0 表示腾空(不受限)
        
        Returns:
            delta_x_final: [B, S, full_dim]
            boundary_vectors: [B, 4, num_rays, 3] (仅返回 horizon=0 的受限边界供vis)
            weights: [B, 4, num_rays]
        """
        batch_size = x.size(0)
        
        # 0. 基础编码与无约束预测
        # ------------------------------------------------------------
        if x.dim() == 2:
            x = x.view(batch_size, self.max_seq, self.full_dim)
            
        t_emb = self.time_mlp(t) # [B, embed]
        
        # [B, S, embed]
        traj_lat = self.traj_encoder(x, t_emb)
        
        # [B, S, full_dim] -> 这是我们的 Base Output
        # 如果腿部腾空或 horizon > 0，直接使用此值
        delta_x_uncons = self.unconstrained_head(traj_lat)
        
        # 初始化最终输出为无约束结果
        delta_x_final = delta_x_uncons.clone()
        
        # 1. 准备受限计算 (Horizon = 0 且仅针对腿部动作)
        # ------------------------------------------------------------
        # 我们使用 "Batch Expansion" 技巧：将 Batch 中的 4 条腿视为 4 * Batch 个独立样本
        
        # 取出 t=0 的 Traj Latent: [B, 1, embed]
        traj_lat_0 = traj_lat[:, 0:1, :]
        
        # 复制 4 份，适配 4 条腿: [B, 4, embed] -> [B*4, 1, embed]
        # 物理含义：每条腿都拥有全身的轨迹上下文信息
        traj_lat_legs = traj_lat_0.expand(-1, 4, -1).reshape(batch_size * 4, 1, self.embed_dim)
        
        # 取出 t=0 的约束: 
        # A: [B, 0, 4, Nc, 3] -> [B, 4, Nc, 3] -> [B*4, 1, Nc, 3]
        # b: [B, 0, 4, Nc]    -> [B, 4, Nc]    -> [B*4, 1, Nc]
        A_0 = A[:, 0, ...].reshape(batch_size * 4, 1, self.num_cons, self.leg_act_dim)
        b_0 = b[:, 0, ...].reshape(batch_size * 4, 1, self.num_cons)
        contact_t0 = contact[:, 0, :] 
        
        # 2. 展平以匹配 A_0/b_0 的 Batch*4 维度: [B*4]
        contact_mask_flat = contact_t0.reshape(-1).float()
        # A_0 需要: [B*4, 1, 1, 1]
        mask_A = contact_mask_flat.view(-1, 1, 1, 1)
        # b_0 需要: [B*4, 1, 1]
        mask_b = contact_mask_flat.view(-1, 1, 1)
        # (Contact=0 的部分，A和b变为全0)
        A_0 = A_0 * mask_A
        b_0 = b_0 * mask_b


        # 取出 t=0 的腿部物理坐标 (用于 Ray Shooting 中心): 
        # 假设 x 的前12维是腿: [B, 12] -> [B, 4, 3] -> [B*4, 1, 3]
        x_legs_0 = x[:, 0, :self.total_leg_dim].reshape(batch_size, 4, self.leg_act_dim)
        x_legs_0_flat = x_legs_0.reshape(batch_size * 4, 1, self.leg_act_dim)
        
        # 扩展 time embedding: [B, embed] -> [B*4, embed]
        t_emb_legs = t_emb.repeat_interleave(4, dim=0)

        # 2. 约束编码与 Cross Attention
        # ------------------------------------------------------------
        # 编码约束: [B*4, 1, num_cons, embed]
        cons_lat = self.constraint_encoder(A_0, b_0)
        
        # === 注入 Leg Condition ===
        # 生成腿部索引: [0, 1, 2, 3, 0, 1, 2, 3, ...]
        # 对应之前的 reshape 逻辑 (batch_size, 4, ...) -> (batch_size*4, ...)
        leg_indices = torch.arange(self.num_legs, device=self.device).repeat(batch_size) # Shape: [B*4]
        # 查找 Embedding: [B*4, embed]
        leg_emb_val = self.leg_embedding(leg_indices)
        # 广播相加:
        # cons_lat:    [B*4, 1, num_cons, embed]
        # leg_emb_val: [B*4, 1, 1,        embed] (需要 unsqueeze)
        cons_lat = cons_lat + leg_emb_val.view(-1, 1, 1, self.embed_dim)


        # 展平用于 Attention: [B*4, num_cons, embed]
        cons_lat_flat = cons_lat.view(batch_size * 4, self.num_cons, self.embed_dim)
        
        # Cross Attn: Query 是轨迹特征，Key/Value 是约束特征
        # query: [B*4, 1, embed], key: [B*4, num_cons, embed]
        e_legs, _ = self.cross_attn(
            query=traj_lat_legs,
            key_value=cons_lat_flat
        )
        
        # 3. Ray Shooting & 几何解算
        # ------------------------------------------------------------
        # 生成射线: [B*4, 1, rays, 3]
        rays = self.ray_mlp(e_legs).view(batch_size * 4, 1, self.num_rays, self.leg_act_dim)
        rays = F.normalize(rays, p=2, dim=-1)
        
        # 几何求交: 
        # x_legs_0_flat (中心), rays (方向), A_0, b_0 (平面)
        # 输出: [B*4, 1, rays, 3]
        bound_vecs = self.ray_shooter(x_legs_0_flat, rays, A_0, b_0)
        
        # 计算权重: [B*4, 1, rays]
        weights = self.weight_decoder(bound_vecs, traj_embed=e_legs, t_embed=t_emb_legs)
        
        # 聚合得到受限的 delta x: [B*4, 1, 3]
        delta_legs_constrained = torch.einsum('bsk,bskd->bsd', weights, bound_vecs)
        
        # 4. 结果合并 (Contact Logic)
        # ------------------------------------------------------------
        # 恢复形状: [B, 4, 3]
        delta_legs_constrained = delta_legs_constrained.view(batch_size, 4, self.leg_act_dim)
        
        # 获取接触掩码: contact [B, 0, 4] -> mask [B, 4, 1]
        # 1 为触地 (需要约束), 0 为腾空 (无需约束)
        mask = contact[:, 0, :].unsqueeze(-1).float() 
        
        # 获取原始无约束预测的腿部部分: [B, 12] -> [B, 4, 3]
        delta_legs_uncons = delta_x_uncons[:, 0, :self.total_leg_dim].view(batch_size, 4, self.leg_act_dim)
        
        # 混合逻辑: 
        # Final = Mask * Constrained + (1 - Mask) * Unconstrained
        delta_legs_final = mask * delta_legs_constrained + (1.0 - mask) * delta_legs_uncons
        
        # 5. 填回最终张量
        # ------------------------------------------------------------
        # 将处理好的腿部动作填回 horizon=0 的前12维
        delta_x_final[:, 0, :self.total_leg_dim] = delta_legs_final.view(batch_size, -1)
        
        # 为可视化整理返回值 (B, 4, rays, 3)
        bound_vecs_reshaped = bound_vecs.view(batch_size, 4, self.num_rays, self.leg_act_dim)
        weights_reshaped = weights.view(batch_size, 4, self.num_rays)

        return delta_x_final, bound_vecs_reshaped, weights_reshaped