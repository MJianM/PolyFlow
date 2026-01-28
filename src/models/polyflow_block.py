import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .block import build_mlp, create_block_diagonal_mask, create_block_cross_attention_mask, EfficientRayShootingLayer

class SequencePositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x, step_repeat=1):

        seq_len = x.size(1) // step_repeat
        
        pe_slice = self.pe[:, :seq_len, :]
        
        if step_repeat > 1:
            pe_slice = pe_slice.repeat_interleave(step_repeat, dim=1)
            
        return pe_slice

class CrossAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  
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
            query: [batch_size, target_len, embed_dim]
            key_value: [batch_size, source_len, embed_dim]
            attn_mask: 
            key_padding_mask:
        """

        residual = query
        
        attn_output, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        
        query = self.norm1(residual + self.dropout(attn_output))
        
        residual = query
        query = self.ffn(query)
        query = self.norm2(residual + query)
        
        return query, attn_weights
    
class SinusoidalTimeEmb(nn.Module):

    def __init__(self, dim, scale=1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale 

    def forward(self, x):
        """
        x: [batch_size], dtype=torch.float32, range=[0, 1]
        """
        device = x.device
        
        x = x * self.scale 
        
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # x: [B], emb: [D/2] -> [B, D/2]
        emb = x[:, None] * emb[None, :]
        
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLNBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        self.attn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )

    def forward(self, x, time_emb, attn_mask=None, key_padding_mask=None):
        """
        x: [batch_size, seq_len, hidden_dim]
        time_emb: [batch_size, hidden_dim] 
        attn_mask: [seq_len, seq_len] 或 [batch_size * num_heads, seq_len, seq_len]     
        key_padding_mask: [batch_size, seq_len]
                   
        """

        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(time_emb).chunk(4, dim=1)
        
        shift_msa, scale_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)
        shift_mlp, scale_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)

        x_norm = self.attn_norm(x)
        x_modulated = x_norm * (1 + scale_msa) + shift_msa
        
        attn_out, _ = self.attn(
            query=x_modulated, 
            key=x_modulated, 
            value=x_modulated, 
            attn_mask=attn_mask,           
            key_padding_mask=key_padding_mask 
        )
        x = x + attn_out 

        x_norm = self.ffn_norm(x)
        x_modulated = x_norm * (1 + scale_mlp) + shift_mlp
        
        ffn_out = self.ffn(x_modulated)
        x = x + ffn_out 

        return x

class ConstraintEncoder(nn.Module):

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
        :param A: (B, S, num_cons, x_dim)
        :param b: (B, S, num_cons)
        :return (B, S, num_cons, embed_dim)
        """
        batch_size = A.shape[0]

        # [B, S, num_cons, x_dim+1]
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

        super().__init__()
        self.x_dim = x_dim
        self.embed_dim = embed_dim
        self.max_seq = max_seq
        self.device = device

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

        tokens = tokens + traj_embed  # fusion 

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

        x_reshaped = x.view(batch_size, self.max_seq, self.x_dim)

        # embed (B, embed_dim)
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
    