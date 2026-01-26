from typing import List, Type
import torch
import torch.nn as nn
import numpy as np
import ot

class RayShootingLayer(nn.Module):
    def forward(self, c, directions, A, b):
        # c (center): [batch, n]
        # directions: [batch, k, n] (k = n+1)
        # A: [batch, M, n]
        # b: [batch, M]
        
        # Slack : s = b - Ac
        # [batch, M]
        Ax = torch.einsum('bmn,bn->bm', A, c)
        slack = b - Ax 
        
        # Ad = A * d
        # [batch, k, M]
        Ad = torch.einsum('bmn,bkn->bkm', A, directions)
        
        EPS = 1e-8
        
        # t = slack / Ad
        # slack [batch, 1, M], Ad [batch, k, M]
        t = slack.unsqueeze(1) / (Ad + EPS)
        
        mask = Ad > 1e-6

        t_masked = t.clone()
        t_masked[~mask] = float('inf')
        
        # alpha: [batch, k, 1]
        alpha, _ = t_masked.min(dim=-1, keepdim=True)
        
        alpha = torch.clamp(alpha, min=0.0, max=1e5) 
        
        # directions: [batch, k, n], alpha: [batch, k, 1]
        # boundary_points = c.unsqueeze(1) + alpha * directions
        boundary_points = alpha * directions  
        
        return boundary_points


class EfficientRayShootingLayer(nn.Module):
    def forward(self, c, directions, A, b):
        """
        c (center): [batch, T, x_dim]  
        directions: [batch, T, k, x_dim] (k = num_vertices)
        A: [batch, T, m, x_dim] 
        b: [batch, T, m]
        """
        # Slack: s = b - Ac
        # Einsum: B(atch), T(ime), M(constraints), D(im)
        # Ac: [batch, T, m]
        Ax = torch.einsum('btmd,btd->btm', A, c)
        slack = b - Ax 
        
        #  Ad = A * d
        # Ad: [batch, T, k, m]
        Ad = torch.einsum('btmd,btkd->btkm', A, directions)
        
        EPS = 1e-6
        
        # alpha = slack / Ad
        # Broadcast slack: [batch, T, 1, m]
        t = slack.unsqueeze(2) / (Ad + EPS)
        
        mask = Ad > EPS
        
        t_masked = t.clone()
        t_masked[~mask] = float('inf')

        # alpha: [batch, T, k, 1]
        alpha, _ = t_masked.min(dim=-1, keepdim=True)

        alpha = torch.clamp(alpha, min=0.0, max=1e3) 
        
        # directions: [batch, T, k, x_dim], alpha: [batch, T, k, 1]
        boundary_vectors = alpha * directions 
        
        return boundary_vectors

def create_block_diagonal_mask(T, block_size=4, device='cpu'):
    L = T * block_size
    mask = torch.ones(L, L, dtype=torch.bool, device=device)
    
    for i in range(T):
        start = i * block_size
        end = (i + 1) * block_size
        mask[start:end, start:end] = False  
    
    return mask


def create_block_cross_attention_mask(query_len, key_len, n, m, T, device='cuda'):
    assert query_len == n * T, f"query_len should be (n+1)*T = {n*T}, but got {query_len}"
    assert key_len == m * T, f"key_len should be m*T = {m*T}, but got {key_len}"
    
    attn_mask = torch.ones(query_len, key_len, dtype=torch.bool, device=device)
    
    for t in range(T):
        query_start = t * n
        query_end = (t + 1) * n
        
        key_start = t * m
        key_end = (t + 1) * m
        
        attn_mask[query_start:query_end, key_start:key_end] = False
    
    return attn_mask

def ot_minibatch_coupling(x0, x1):
    """
    Args:
        x0: Source samples (Noise), [Batch, Dim]
        x1: Target samples (Data), [Batch, Dim]
    """
    device = x0.device
    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()
    
    batch_size = x0.shape[0]
    
    a = np.ones((batch_size,)) / batch_size
    b = np.ones((batch_size,)) / batch_size

    M = ot.dist(x0_np, x1_np, metric='sqeuclidean')
    
    G = ot.emd(a, b, M)

    pair_indices = np.argmax(G, axis=1)

    pair_indices = torch.from_numpy(pair_indices).to(device)
    
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
    layers = []
    current_dim = input_dim
    
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim, bias=use_bias))
        layers.append(activation_class()) 
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = h_dim
        
    layers.append(nn.Linear(current_dim, output_dim, bias=use_bias))
    
    return nn.Sequential(*layers)

def set_all_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__=='__main__':

    mask = create_block_diagonal_mask(T=2, block_size=3)
    print(mask)