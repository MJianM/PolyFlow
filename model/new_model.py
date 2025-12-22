import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import build_mlp, create_block_diagonal_mask, create_block_cross_attention_mask, EfficientRayShootingLayer

class SequencePositionalEncoding(nn.Module):
    """
    标准的正弦位置编码 (Sinusoidal Positional Encoding)
    用于给序列中的 Token 增加位置信息 (Sequence Order)
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建一个足够长的 PE 矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [1, max_len, d_model] 以便广播
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer，不是模型参数，不参与梯度更新，但随模型保存
        self.register_buffer('pe', pe)

    def forward(self, x, step_repeat=1):
        """
        Args:
            x: 输入 tensor，主要用于获取 device 和 dtype，形状为 (B, T*step_repeat, embed_dim)
            step_repeat: 每个时间步重复多少次 (Block Size)
                         如果 step_repeat > 1，则输出形式为 [pos0, pos0, pos1, pos1, ...]
        Returns:
            Sliced PE tensor matched to sequence length
        """
        # 假设需要的序列长度是 encoded length / step_repeat
        # 例如输入 x 长度 400, step_repeat=4, 则实际的时间步 T=100
        seq_len = x.size(1) // step_repeat
        
        # 获取前 seq_len 个位置编码: [1, T, d_model]
        pe_slice = self.pe[:, :seq_len, :]
        
        if step_repeat > 1:
            # 在 dim=1 重复每个位置编码 step_repeat 次
            # 结果形状: [1, T * step_repeat, d_model]
            pe_slice = pe_slice.repeat_interleave(step_repeat, dim=1)
            
        return pe_slice

class CrossAttentionBlock(nn.Module):
    """使用 PyTorch 内置 MultiheadAttention 实现的 Cross-Attention"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        
        # Cross-Attention: query 来自一个序列，key/value 来自另一个序列
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # PyTorch 2.0+ 支持
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: [batch_size, target_len, embed_dim] - 来自一个序列
            key_value: [batch_size, source_len, embed_dim] - 来自另一个序列
            attn_mask: 注意力掩码
            key_padding_mask: key 的填充掩码
        """
        # 保存残差连接
        residual = query
        
        # Cross-Attention
        # query 来自第一个序列，key/value 来自第二个序列
        attn_output, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        
        # 残差连接 + LayerNorm
        query = self.norm1(residual + self.dropout(attn_output))
        
        # Feed-Forward Network
        residual = query
        query = self.ffn(query)
        query = self.norm2(residual + query)
        
        return query, attn_weights
    
class SinusoidalTimeEmb(nn.Module):
    """
    针对 [0, 1] 浮点数输入的正弦位置编码
    """
    def __init__(self, dim, scale=1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale # 新增缩放因子

    def forward(self, x):
        """
        x: [batch_size], dtype=torch.float32, range=[0, 1]
        """
        device = x.device
        
        # 1. 缩放输入：将 [0, 1] 映射到 [0, scale]
        # 这样做的目的是防止输入过小导致 sin/cos 函数处于线性区（失去区分度）
        x = x * self.scale 
        
        # 2. 计算频率项 (与之前逻辑一致)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # 3. 计算正弦特征
        # x: [B], emb: [D/2] -> [B, D/2]
        emb = x[:, None] * emb[None, :]
        
        # 4. 拼接 sin 和 cos
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLNBlock(nn.Module):
    """
    支持时间条件注入的 Transformer Block (基于 AdaLN)。
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        # 1. Self-Attention 模块
        self.attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False) # 关闭默认的 affine 参数，由 AdaLN 接管
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 2. Feed Forward 模块
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)  # 关闭默认的 affine 参数
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 3. AdaLN 调制层 (Modulation)
        # 它的作用是：根据 time_emb 预测出用于调节 LayerNorm 的 scale 和 shift 参数
        # 我们需要预测 4 个参数：
        # - shift_msa, scale_msa (用于 Attention 前的 Norm)
        # - shift_mlp, scale_mlp (用于 FFN 前的 Norm)
        # 所以输出维度是 4 * hidden_dim
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )

    def forward(self, x, time_emb, attn_mask=None, key_padding_mask=None):
        """
        x: [batch_size, seq_len, hidden_dim] - 序列输入 (例如动作序列或图像Patch)
        time_emb: [batch_size, hidden_dim] - 时间嵌入向量
        attn_mask: [seq_len, seq_len] 或 [batch_size * num_heads, seq_len, seq_len]
                   用于屏蔽特定位置的注意力（如因果掩码）。
        key_padding_mask: [batch_size, seq_len]
                   用于屏蔽 Padding Token (True 表示被屏蔽/不参与计算)。
        """
        # 1. 计算调制参数 (Scale & Shift)
        # shift_msa, scale_msa, shift_mlp, scale_mlp 形状均为 [batch_size, hidden_dim]
        # chunk(4, dim=1) 将长向量切分成4份
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(time_emb).chunk(4, dim=1)
        
        # 注意：需要 unsqueeze(1) 以便广播到 seq_len 维度: [batch_size, 1, hidden_dim]
        shift_msa, scale_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)
        shift_mlp, scale_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)

        # 2. Attention 块 (应用 AdaLN)
        # 公式: x = x * (1 + scale) + shift
        # 这里的 modulate 操作替代了标准的 LayerNorm 可学习参数
        x_norm = self.attn_norm(x)
        x_modulated = x_norm * (1 + scale_msa) + shift_msa
        
        attn_out, _ = self.attn(
            query=x_modulated, 
            key=x_modulated, 
            value=x_modulated, 
            attn_mask=attn_mask,           
            key_padding_mask=key_padding_mask 
        )
        x = x + attn_out # 残差连接

        # 3. Feed Forward 块 (应用 AdaLN)
        x_norm = self.ffn_norm(x)
        x_modulated = x_norm * (1 + scale_mlp) + shift_mlp
        
        ffn_out = self.ffn(x_modulated)
        x = x + ffn_out # 残差连接

        return x

class ConstraintEncoder(nn.Module):
    """
    处理 A, b 每行组成 token -> multi-head attention block
    这是 Set Transformer 的一部分，用于处理 permutation-invariant 的约束集合。
    """
    def __init__(self, x_dim, embed_dim, num_cons, max_seq, num_heads, num_layers=2, use_block_mask=True, device='cuda'):
        """
        Args:
            x_dim: 网络输入维度其实是 x_dim (A的列数) + 1 (b的值)
            embed_dim: Embedding 维度
            num_cons: 每个点约束个数
            max_seq: 轨迹点个数
            num_heads: Transformer Heads
            num_layers: Transformer Layers
            use_block_mask: 是否只对每个点内的约束组进行self-attn
        """
        super().__init__()
        self.num_cons = num_cons
        self.max_seq = max_seq
        self.x_dim = x_dim
        self.embed_dim = embed_dim
        self.use_block_mask = use_block_mask
        self.device = device

        if self.use_block_mask:
            self.mask = create_block_diagonal_mask(
                T=self.max_seq, block_size=self.num_cons, device=self.device
            )
        else:
            self.mask = None

        # 输入维度是 x_dim (A的列数) + 1 (b的值)
        self.input_proj = build_mlp(input_dim=x_dim+1, hidden_dims=[4*embed_dim], output_dim=embed_dim).to(self.device)
        
        # 使用 Transformer Encoder 处理集合输入 (无位置编码，因为约束顺序无关)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

    def forward(self, A, b):
        """
        Docstring for forward
        
        :param self: Description
        :param A: 形状 (B, S, num_cons, x_dim)
        :param b: 形状 (B, S, num_cons)
        :return 形状 (B, S, num_cons, embed_dim)
        """
        batch_size = A.shape[0]

        # 拼接 -> [B, S, num_cons, x_dim+1]
        constraints = torch.cat([A, b.unsqueeze(-1)], dim=-1)
        tokens = self.input_proj(constraints) # [batch, S, num_cons, embed_dim]
        
        # flatten
        tokens = tokens.reshape(batch_size, self.max_seq*self.num_cons, self.embed_dim)
        encoded_constraints = self.transformer_encoder(tokens, mask=self.mask)

        # reshape (B, S, num_cons, embed_dim)
        encoded_constraints = encoded_constraints.reshape(batch_size, self.max_seq, self.num_cons, self.embed_dim)
        return encoded_constraints
    
class TrajEncoder(nn.Module):
    def __init__(self, x_dim, embed_dim, max_seq, num_heads, num_layers=2, device='cuda'):
        '''
        Docstring for __init__
        
        :param self: Description
        :param x_dim: Description
        :param embed_dim: Description
        :param max_seq: Description
        :param num_heads: Description
        :param num_layers: 
        :param device: Description
        '''
        super().__init__()
        self.x_dim = x_dim
        self.embed_dim = embed_dim
        self.max_seq = max_seq
        self.device = device

        # 在轨迹序列维度进行时序编码
        self.positional_embedding = SequencePositionalEncoding(embed_dim, max_seq).to(device)

        self.input_proj = build_mlp(x_dim, hidden_dims=[4*embed_dim], output_dim=embed_dim).to(device)

        self.attn_blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)
        ]).to(device)

    def forward(self, x, t_embed, attn_mask=None, key_padding_mask=None):
        """
        Docstring for forward
        
        :param self: Description
        :param x: [B, S, x_dim]
        :param t: [B, embed_dim]
        """

        tokens = self.input_proj(x)
        # 添加轨迹时序维度的pe
        pe = self.positional_embedding(x, step_repeat=1)
        tokens = tokens + pe

        # t_embed = self.time_mlp(t)

        for block in self.attn_blocks:
            tokens = block(tokens, t_embed, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return tokens
        
class OneWeightDecoder(nn.Module):
    def __init__(self, x_dim, embed_dim, max_seq, num_rays, num_heads, num_layers, device='cuda'):
        super().__init__()
        self.x_dim = x_dim
        self.embed_dim = embed_dim
        self.max_seq = max_seq
        self.num_rays = num_rays
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        assert self.num_rays == 1

        # input [batch_size, seq_length, x_dim+embed_dim] 最后一维表示 方向+模长 + 轨迹embed信息

        self.input_proj = build_mlp(x_dim, hidden_dims=[embed_dim], output_dim=embed_dim).to(device)

        self.attn_blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)
        ]).to(device)

        self.final_norm = nn.LayerNorm(embed_dim).to(device)
        self.output_proj = nn.Linear(embed_dim, 1).to(device)

    def forward(self, rays, traj_embed, t_embed, attn_mask=None, key_padding_mask=None):
        """
        Docstring for forward
        
        :param self: Description
        :param rays: (B, S, num_rays, x_dim)
        :param traj_embed: (B, S, embed_dim)
        :param t_embed: (B)
        :param attn_mask: (S*num_rays, S*num_rays)
        :param key_padding_mask: Description
        """
        batch_size = rays.size(0)

        # (B, S*num_rays, embed_dim)
        tokens = self.input_proj(rays)
        tokens = tokens.reshape(batch_size, self.max_seq*self.num_rays, self.embed_dim)

        tokens = tokens + traj_embed  # fusion 融合轨迹信息和边界信息

        for block in self.attn_blocks:
            tokens = block(tokens, t_embed, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        tokens = self.final_norm(tokens)

        # (B, S, num_rays)
        weight = self.output_proj(tokens).reshape(batch_size, self.max_seq, self.num_rays)
        weight = F.sigmoid(weight)

        return weight

class WeightDecoder(nn.Module):
    def __init__(self, x_dim, embed_dim, max_seq, num_rays, num_heads, num_layers, device='cuda'):
        super().__init__()
        self.x_dim = x_dim
        self.embed_dim = embed_dim
        self.max_seq = max_seq
        self.num_rays = num_rays
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        self.input_proj = build_mlp(x_dim, hidden_dims=[4*embed_dim], output_dim=embed_dim).to(device)

        self.attn_blocks = nn.ModuleList([
            AdaLNBlock(hidden_dim=embed_dim, num_heads=num_heads) for _ in range(num_layers)
        ]).to(device)

        self.final_norm = nn.LayerNorm(embed_dim).to(device)
        self.output_proj = nn.Linear(embed_dim, 1).to(device)

    def forward(self, rays, traj_embed, t_embed, attn_mask=None, key_padding_mask=None):
        """
        Docstring for forward
        
        :param self: Description
        :param rays: (B, S, num_rays, x_dim)
        :param traj_embed (B, S, embed_dim)
        :param t_embed: (B)
        :param attn_mask: (S*num_rays, S*num_rays)
        :param key_padding_mask: Description
        """
        batch_size = rays.size(0)

        
        tokens = self.input_proj(rays)
        tokens = tokens + traj_embed.unsqueeze(2) # (B, S, num_rays, embed_dim)
        tokens = tokens.reshape(batch_size, self.max_seq*self.num_rays, self.embed_dim) # (B, S*num_rays, embed_dim)

        for block in self.attn_blocks:
            tokens = block(tokens, t_embed, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        tokens = self.final_norm(tokens)

        # (B, S, num_rays)
        weight = self.output_proj(tokens).reshape(batch_size, self.max_seq, self.num_rays)
        if self.num_rays == 1:
            weight = F.sigmoid(weight)
        else:
            weight = F.softmax(weight, dim=-1)

        return weight


class PolytopeConstrainedFlowModel(nn.Module):
    def __init__(self, x_dim, num_cons, num_rays, max_seq, embed_dim=128,
                 num_heads_cons=4, num_layers_cons=2, 
                 num_heads_traj=4, num_layers_traj=2, 
                 num_heads_weight=4, num_layers_weight=2,
                 time_embed_scale=1000, use_block_mask_cons=True,
                 use_block_mask_cross=True, use_block_mask_weight=True,
                 device='cuda'):
        super().__init__()

        assert num_rays >= 1

        self.x_dim = x_dim
        self.num_cons = num_cons
        self.num_rays = num_rays
        self.max_seq = max_seq
        self.embed_dim = embed_dim
        self.use_block_mask_cons = use_block_mask_cons
        self.device = device

        self.constraint_encoder = ConstraintEncoder(
            x_dim=x_dim, embed_dim=embed_dim, num_cons=num_cons,
            max_seq=max_seq, num_heads=num_heads_cons, num_layers=num_layers_cons,
            use_block_mask=use_block_mask_cons, device=device
        )

        self.traj_encoder = TrajEncoder(
            x_dim=x_dim, embed_dim=embed_dim, max_seq=max_seq,
            num_heads=num_heads_traj, num_layers=num_layers_traj,
            device=device
        )

        # 对flow的时间输入进行编码
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(dim=embed_dim, scale=time_embed_scale),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(device)

        # Cross Attention Block (Fusion)
        # Q: Traj Point, K,V: Constraints
        self.cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim, num_heads=num_heads_traj
        ).to(device)
        if use_block_mask_cross:
            self.cross_attn_mask = create_block_cross_attention_mask(
                query_len=self.max_seq, key_len=self.max_seq*self.num_cons, n=1, m=self.num_cons, T=self.max_seq, device=device
            )
        else:
            self.cross_attn_mask = None

        # 输出rays头
        self.ray_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_rays*x_dim)
        ).to(device)

        # 计算边界点
        self.ray_shooter = EfficientRayShootingLayer().to(device)

        # 计算weight
        self.weight_decoder = WeightDecoder(
            x_dim=x_dim, embed_dim=embed_dim, max_seq=max_seq, num_rays=num_rays,
            num_heads=num_heads_weight, num_layers=num_layers_weight, device=device
        ).to(device)
        if use_block_mask_weight:
            self.weight_attn_mask = create_block_diagonal_mask(T=self.max_seq, block_size=self.num_rays, device=device)
        else:
            self.weight_attn_mask = None

    def forward(self, x, t, A, b):
        """
        Docstring for forward
        
        :param self: Description
        :param x: [B, x_dim*S]
        :param t: [B]
        :param A: [B, S, num_cons, x_dim]
        :param b: [B, S, num_cons]
        """

        batch_size = x.size(0)

        # 数据整形
        x_reshaped = x.view(batch_size, self.max_seq, self.x_dim)

        # 构造时间embed (B, embed_dim)
        t_emb = self.time_mlp(t)

        # (B, S, embed_dim)
        traj_lat = self.traj_encoder(x_reshaped, t_emb)

        # (B, S, num_cons, embed_dim)
        cons_lat = self.constraint_encoder(A, b)

        # Q: (B, S, embed_dim), K: (B, S*num_cons, embed_dim)
        cons_lat = cons_lat.reshape(batch_size, self.max_seq*self.num_cons, self.embed_dim)
        e, _ = self.cross_attn(query=traj_lat, key_value=cons_lat, attn_mask=self.cross_attn_mask)

        # ray (B, S, num_rays, x_dim)
        rays = self.ray_mlp(e).view(batch_size, self.max_seq, self.num_rays, self.x_dim)
        rays = F.normalize(rays, p=2, dim=-1)

        # ray shooting (B, S, num_rays, x_dim)
        boundary_vectors = self.ray_shooter(x_reshaped, rays, A, b)

        # weight (B, S, num_rays)
        if self.num_rays == 1:
            weights = self.weight_decoder(boundary_vectors, e, t_emb)
        else:
            weights = self.weight_decoder(boundary_vectors, e, t_emb, attn_mask=self.weight_attn_mask)


        # (B, S, x_dim)
        v_geometric = torch.einsum('bsk,bskd->bsd', weights, boundary_vectors)

        delta_x_reshaped = v_geometric

        # (B, S*x_dim)
        delta_x = delta_x_reshaped.reshape(batch_size, -1)

        return delta_x, boundary_vectors, weights


class PolytopeConstrainedOneRayFlowModel(nn.Module):
    def __init__(self, 
                 x_dim, 
                 num_cons, 
                 num_rays, 
                 max_seq, 
                 embed_dim=128, 
                 num_heads_cons=4, num_layers_cons=2, 
                 num_heads_traj=4, num_layers_traj=2, 
                 num_heads_weight=4, num_layers_weight=2, 
                 time_embed_scale=1000, 
                 use_block_mask_cons=True, use_block_mask_cross=True, use_block_mask_weight=True, 
                 device='cuda'):
        super().__init__()

        assert num_rays == 1

        self.x_dim = x_dim
        self.num_cons = num_cons
        self.num_rays = num_rays
        self.max_seq = max_seq
        self.embed_dim = embed_dim
        self.use_block_mask_cons = use_block_mask_cons
        self.device = device

        self.constraint_encoder = ConstraintEncoder(
            x_dim=x_dim, embed_dim=embed_dim, num_cons=num_cons,
            max_seq=max_seq, num_heads=num_heads_cons, num_layers=num_layers_cons,
            use_block_mask=use_block_mask_cons, device=device
        )

        self.traj_encoder = TrajEncoder(
            x_dim=x_dim, embed_dim=embed_dim, max_seq=max_seq,
            num_heads=num_heads_traj, num_layers=num_layers_traj,
            device=device
        )

        # 对flow的时间输入进行编码
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(dim=embed_dim, scale=time_embed_scale),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(device)

        # Cross Attention Block (Fusion)
        # Q: Traj Point, K,V: Constraints
        self.cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim, num_heads=num_heads_traj
        ).to(device)
        if use_block_mask_cross:
            self.cross_attn_mask = create_block_cross_attention_mask(
                query_len=self.max_seq, key_len=self.max_seq*self.num_cons, n=1, m=self.num_cons, T=self.max_seq, device=device
            )
        else:
            self.cross_attn_mask = None

        # 输出rays头
        self.ray_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_rays*x_dim)
        ).to(device)

        # 计算边界点
        self.ray_shooter = EfficientRayShootingLayer().to(device)

        # 计算weight
        # self.weight_decoder = WeightDecoder(
        #     x_dim=x_dim, embed_dim=embed_dim, max_seq=max_seq, num_rays=num_rays,
        #     num_heads=num_heads_weight, num_layers=num_layers_weight, device=device
        # ).to(device)
        # if use_block_mask_weight:
        #     self.weight_attn_mask = create_block_diagonal_mask(T=self.max_seq, block_size=self.num_rays, device=device)
        # else:
        #     self.weight_attn_mask = None
        self.weight_decoder: OneWeightDecoder = OneWeightDecoder(
            x_dim=x_dim, embed_dim=embed_dim, max_seq=max_seq, num_rays=1,
            num_heads=num_heads_weight, num_layers=num_layers_weight, device=device
        )

    def forward(self, x, t, A, b):
        """
        Docstring for forward
        
        :param self: Description
        :param x: [B, x_dim*S]
        :param t: [B]
        :param A: [B, S, num_cons, x_dim]
        :param b: [B, S, num_cons]
        """

        batch_size = x.size(0)

        # 数据整形
        x_reshaped = x.view(batch_size, self.max_seq, self.x_dim)

        # 构造时间embed (B, embed_dim)
        t_emb = self.time_mlp(t)

        # (B, S, embed_dim)
        traj_lat = self.traj_encoder(x_reshaped, t_emb)

        # (B, S, num_cons, embed_dim)
        cons_lat = self.constraint_encoder(A, b)

        # Q: (B, S, embed_dim), K: (B, S*num_cons, embed_dim)
        cons_lat = cons_lat.reshape(batch_size, self.max_seq*self.num_cons, self.embed_dim)
        e, _ = self.cross_attn(query=traj_lat, key_value=cons_lat, attn_mask=self.cross_attn_mask)

        # ray (B, S, num_rays, x_dim)
        rays = self.ray_mlp(e).view(batch_size, self.max_seq, self.num_rays, self.x_dim)
        rays = F.normalize(rays, p=2, dim=-1)

        # ray shooting (B, S, num_rays, x_dim)
        boundary_vectors = self.ray_shooter(x_reshaped, rays, A, b)

        # weight (B, S, num_rays)
        weights = self.weight_decoder(boundary_vectors, traj_embed=e, t_embed=t_emb)

        # (B, S, x_dim)
        v_geometric = torch.einsum('bsk,bskd->bsd', weights, boundary_vectors)

        delta_x_reshaped = v_geometric

        # (B, S*x_dim)
        delta_x = delta_x_reshaped.reshape(batch_size, -1)

        return delta_x, boundary_vectors, weights
    
