import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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

    def __init__(self, 
                 full_x_dim: int,
                 constrained_idxs: List[int],
                 num_cons: int, 
                 max_seq: int, 
                 embed_dim: int = 128, 
                 share_traj_encoder: bool = True,  
                 cons_begin_seq_idx: int = 1, 
                 time_invariance_cons: bool = False, 
                 ray_shooting_method: str = 'hard',
                 ray_shooting_beta: float = 50,
                 # 
                 num_rays: int = 1,
                 num_heads_cons: int = 4, 
                 num_layers_cons: int = 2, 
                 num_heads_weight: int = 4, 
                 num_layers_weight: int = 2,
                 # 
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
        self.cons_begin_seq_idx = cons_begin_seq_idx 
        self.time_invariance_cons = time_invariance_cons
        self.ray_shooting_method = ray_shooting_method
        self.ray_shooting_beta = ray_shooting_beta
        print(f"PartialPolyFlow Using time invariance cons: {self.time_invariance_cons}   Ray shooting method: {self.ray_shooting_method}  Ray shooting beta: {self.ray_shooting_beta}")
        
        self.c_idxs = torch.tensor(constrained_idxs, dtype=torch.long, device=device)
        
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


        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(dim=embed_dim, scale=time_embed_scale),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ).to(device)


        if self.share_traj_encoder:
            self.traj_encoder = TrajEncoder(
                x_dim=self.full_x_dim,  
                embed_dim=embed_dim,
                max_seq=max_seq,
                num_heads=num_heads_traj,
                num_layers=num_layers_traj,
                device=device
            )
            self.traj_encoder_c = None
            self.traj_encoder_u = None
        else:
            if self.dim_c > 0:
                self.traj_encoder_c = TrajEncoder(
                    x_dim=self.dim_c, 
                    embed_dim=embed_dim,
                    max_seq=max_seq,
                    num_heads=num_heads_traj,
                    num_layers=num_layers_traj,
                    device=device
                )
            if self.dim_u > 0:
                self.traj_encoder_u = TrajEncoder(
                    x_dim=self.dim_u, 
                    embed_dim=embed_dim,
                    max_seq=max_seq,
                    num_heads=num_heads_traj,
                    num_layers=num_layers_traj,
                    device=device
                )

        if self.dim_c > 0:
            remain_length = self.max_seq - self.cons_begin_seq_idx
            if self.time_invariance_cons:
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
            
            # Mask
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

        if self.dim_u > 0:
            self.unconstrained_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, self.dim_u)
            ).to(device)

    def forward(self, x, cond, t, A, b):
        """
        x: [B, S, full_x_dim] 
        t: [B,]
        A: [B, S, num_cons, dim_c]
        b: [B, S, num_cons,]
        return:
            [B, S, full_x_dim]
            [B, S, num_rays, dim_c]
            [B, S, num_rays]
        """
        batch_size = x.size(0)
        start_idx = self.cons_begin_seq_idx
        
        if x.dim() == 2:
            x = x.view(batch_size, self.max_seq, self.full_x_dim)
            
        x_c = x.index_select(-1, self.c_idxs)
        x_u = x.index_select(-1, self.u_idxs)
        
        t_emb = self.time_mlp(t)

        if self.share_traj_encoder:
            # latent shape: [B, S, embed_dim]
            global_lat = self.traj_encoder(x, t_emb)
            
            traj_lat_c = global_lat
            traj_lat_u = global_lat
        else:

            traj_lat_c = self.traj_encoder_c(x_c, t_emb) if self.dim_c > 0 else None
            traj_lat_u = self.traj_encoder_u(x_u, t_emb) if self.dim_u > 0 else None

        
        delta_x_c_future = None
        boundary_vectors_full = torch.zeros(batch_size, self.max_seq, self.num_rays, self.dim_c, device=self.device)
        weights_full = torch.zeros(batch_size, self.max_seq, self.num_rays, device=self.device)
        
        if self.dim_c > 0:
            traj_lat_c_active = traj_lat_c[:, start_idx:, :]
            A_active = A[:, start_idx:, :, :]
            b_active = b[:, start_idx:, :]
            x_c_active = x_c[:, start_idx:, :] 

            if self.time_invariance_cons:
                # (batch, 1, cons_num, embed)
                cons_lat_active = self.constraint_encoder(A_active[:, 0:1, :, :], b_active[:, 0:1, :])
            else:
                cons_lat_active = self.constraint_encoder(A_active, b_active)
            
            if self.time_invariance_cons:
                # Attention
                # Query: [B, S_future, E], Key: [B, M, E]
                cons_lat_flat = cons_lat_active.reshape(batch_size, self.num_cons, self.embed_dim)
                e_c_active, _ = self.cross_attn(
                    query=traj_lat_c_active, 
                    key_value=cons_lat_flat, 
                )
            else:
                # Cross Attention
                # Constraint: [B, S_future * num_cons, embed_dim]
                S_future = traj_lat_c_active.size(1)
                cons_lat_flat = cons_lat_active.reshape(batch_size, S_future * self.num_cons, self.embed_dim)
                
                if self.full_cross_attn_mask is not None:
                    # Query : start ~ end
                    # Key : start*num_cons ~ end*num_cons
                    active_mask = self.full_cross_attn_mask[
                        start_idx : , 
                        start_idx * self.num_cons : 
                    ]
                else:
                    active_mask = None

                # Attention
                # Query: [B, S_future, E], Key: [B, S_future*M, E]
                e_c_active, _ = self.cross_attn(
                    query=traj_lat_c_active, 
                    key_value=cons_lat_flat, 
                    attn_mask=active_mask
                )
            
            S_future = traj_lat_c_active.size(1)

            rays_active = self.ray_mlp(e_c_active).view(batch_size, S_future, self.num_rays, self.dim_c)
            rays_active = F.normalize(rays_active, p=2, dim=-1)
            
            bound_vec_active = self.ray_shooter(x_c_active, rays_active, A_active, b_active)
            
            weights_active = self.weight_decoder(bound_vec_active, traj_embed=e_c_active, t_embed=t_emb)
            
            # delta
            delta_x_c_active = torch.einsum('bsk,bskd->bsd', weights_active, bound_vec_active)
            

            boundary_vectors_full[:, start_idx:, :, :] = bound_vec_active
            weights_full[:, start_idx:, :] = weights_active
            delta_x_c_future = delta_x_c_active 


        delta_x_u_future = None
        if self.dim_u > 0:
            traj_lat_u_active = traj_lat_u
            delta_x_u_future = self.unconstrained_head(traj_lat_u_active)


        delta_x_final = torch.zeros(batch_size, self.max_seq, self.full_x_dim, device=self.device)
        
        
        if self.dim_c > 0:

            delta_x_final[:, start_idx:, :].index_copy_(-1, self.c_idxs, delta_x_c_future)
        
        if self.dim_u > 0:

            delta_x_final[:, :, :].index_copy_(-1, self.u_idxs, delta_x_u_future)


        return delta_x_final, boundary_vectors_full, weights_full