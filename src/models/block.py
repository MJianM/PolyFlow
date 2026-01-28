from typing import List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ot
import time

class RayShootingLayer(nn.Module):
    """
    Given center c, direction d, the ray is r = c + t * d (t > 0)。
    A(c + t*d) <= b  =>  A*c + t*(A*d) <= b  => t*(A*d) <= b - A*c
    slack = b - A*c
    projection = A*d
    t * projection <= slack
    For every contraint i:
       1. if projection_i <= 0: ignore
       2. if projection_i > 0: t <= slack_i / projection_i
    t = min_i (slack_i / projection_i)
    """
    def forward(self, c, directions, A, b):
        # c (center): [batch, n]
        # directions: [batch, k, n] (k = n+1)
        # A: [batch, M, n]
        # b: [batch, M]
        
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
    """
    """
    def __init__(self, method='hard', beta=50.0):
        """
        Args:
            method: 'hard', 'softmin', or 'boltzmann'
            beta: Temperature scaling
        """
        super().__init__()
        self.method = method
        self.beta = beta # 10.0 ~ 100.0

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
        Returns:
            t: [batch, T, k, m] 
            mask: [batch, T, k, m] 
        """
        # Slack: s = b - Ac -> [batch, T, m]
        Ax = torch.einsum('btmd,btd->btm', A, c)
        slack = b - Ax 
        
        # Ad = A * d -> [batch, T, k, m]
        Ad = torch.einsum('btmd,btkd->btkm', A, directions)
        
        # intersection times (t)
        EPS = 1e-6
        # Broadcast slack: [batch, T, 1, m]
        t = slack.unsqueeze(2) / (Ad + EPS)
        
        mask = Ad > EPS
        
        return t, mask

    def forward_hard(self, c, directions, A, b):
        t, mask = self._compute_raw_t(c, directions, A, b)
        
        t_masked = t.clone()
        t_masked[~mask] = float('inf')
        
        # t_masked = torch.where(mask, t, torch.tensor(1e9, device=t.device))

        # alpha: [batch, T, k, 1]
        alpha, _ = t_masked.min(dim=-1, keepdim=True)
        
        alpha = torch.clamp(alpha, min=0.0, max=1e3) 
        
        return alpha * directions 
    
    def forward_softmin(self, c, directions, A, b):
        """
        SoftMin (LogSumExp)
        alpha = - (1/beta) * log( sum( exp(-beta * t_i) ) )
        """
        t, mask = self._compute_raw_t(c, directions, A, b)
        

        t_masked = t.clone()
        t_masked[~mask] = 1e9 
        
        neg_scaled_t = -self.beta * t_masked
        
        # alpha: [batch, T, k, 1]
        alpha = -torch.logsumexp(neg_scaled_t, dim=-1, keepdim=True) / self.beta
        
        alpha = torch.clamp(alpha, min=0.0, max=1e3)
        
        return alpha * directions

    def forward_boltzmann(self, c, directions, A, b):
        """
        Boltzmann / Softmax Weighted Average
        alpha = sum( w_i * t_i ), where w_i = Softmax(-beta * t_i)
        """
        t, mask = self._compute_raw_t(c, directions, A, b)
        

        t_masked = t.clone()
        t_masked[~mask] = 1e9
        

        weights = F.softmax(-self.beta * t_masked, dim=-1)
        

        t_safe = t.clone()
        t_safe[~mask] = 0.0 
        
        alpha = (weights * t_safe).sum(dim=-1, keepdim=True)
        
        alpha = torch.clamp(alpha, min=0.0, max=1e3)
        
        return alpha * directions


def create_block_diagonal_mask(T, block_size=4, device='cpu'):
    """
    Args:
        T: number of blocks
        block_size: size of each block
        device
    
    Returns:
        mask: (L, L)
    """
    L = T * block_size
    mask = torch.ones(L, L, dtype=torch.bool, device=device)
    
    for i in range(T):
        start = i * block_size
        end = (i + 1) * block_size
        mask[start:end, start:end] = False 
    
    return mask


def create_block_cross_attention_mask(query_len, key_len, n, m, T, device='cuda'):
    """
    
    Args:
        query_len: n*T, query length
        key_len: m*T, key/value length
        n: size of each query block
        m: size of each key/value block
        T: number of blocks
    """
    # check dimensions
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

