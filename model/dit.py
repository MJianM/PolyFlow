import torch
import torch.nn as nn
import math

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

class SinusoidalTimeDiffusionEmb(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device) / half_dim
        )

        args = timesteps[:, None].float() * freqs[None, :]

        # [B, D/2] -> [B, D]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [Batch, Seq_Len, Dim]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class AdaLN(nn.Module):
    def __init__(self, hidden_size, time_emb_dim):
        super().__init__()
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x, t_emb):
        # x: [B, S, D], t_emb: [B, time_emb_dim]
        # shift, scale: [B, D] -> [B, 1, D]
        scale_shift = self.emb_layer(t_emb).unsqueeze(1) 
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, time_emb_dim, mlp_ratio=4.0):
        super().__init__()
        self.ada_ln1 = AdaLN(hidden_size, time_emb_dim)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
        self.ada_ln2 = AdaLN(hidden_size, time_emb_dim)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

    def forward(self, x, t_emb):
        # Self-Attention Branch
        # x: [B, Seq_Len, Hidden]
        x_norm1 = self.ada_ln1(x, t_emb)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + attn_out
        
        # MLP Branch
        x_norm2 = self.ada_ln2(x, t_emb)
        x = x + self.mlp(x_norm2)
        return x

class TrajectoryDiT(nn.Module):
    """
    Diffusion Transformer for Trajectories
    input: x (B, Horizon, x_dim), t (B,)
    output: v (B, Horizon, x_dim)
    """
    def __init__(self, 
                 x_dim=2, 
                 max_horizon=100, 
                 hidden_dim=128, 
                 depth=4, 
                 num_heads=4, 
                 time_embed_dim=128):
        super().__init__()
        
        # 1. Input Embedding (State -> Hidden)
        self.input_proj = nn.Linear(x_dim, hidden_dim)
        
        # 2. Time Embedding (Diffusion Time t)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 3. Positional Embedding (Sequence Position s)
        self.pos_embed = SinusoidalPositionalEncoding(hidden_dim, max_len=max_horizon)

        # 4. Transformer Backbone
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, time_embed_dim)
            for _ in range(depth)
        ])

        # 5. Output Head (Hidden -> Vector Field)
        # Final AdaLN usually helps convergence
        self.final_ada_ln = AdaLN(hidden_size=hidden_dim, time_emb_dim=time_embed_dim)
        self.output_proj = nn.Linear(hidden_dim, x_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        
        # Zero-out output layers 
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, x, t):
        B, H, D = x.shape
        
        # Embed Inputs
        h = self.input_proj(x) # [B, H, Hidden]
        
        # Add Positional Embeddings 
        h = self.pos_embed(h)
        
        # Embed Diffusion Time
        t_emb = self.time_mlp(t) # [B, Time_Dim]
        
        # Process through DiT Blocks
        for block in self.blocks:
            h = block(h, t_emb)
            
        # Final Layer
        h = self.final_ada_ln(h, t_emb)
        out = self.output_proj(h) # [B, H, Dim]
        
        return out

class TrajectoryDiTConditional(TrajectoryDiT):
    def __init__(self, 
                 x_dim=2, 
                 cond_dim=2,
                 max_horizon=100, 
                 hidden_dim=128, 
                 depth=4, 
                 num_heads=4, 
                 time_embed_dim=128):
        super().__init__(x_dim, max_horizon, hidden_dim, depth, num_heads, time_embed_dim)
        
        # 2. Time Embedding (Diffusion Time t)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeDiffusionEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x, cond, t):
        """
        x: [Batch, Horizon, Dim] 
        cond: [Batch, cond_dim]
        t: [Batch,]
        """
        B, H, D = x.shape

        return super().forward(x, t)

    

