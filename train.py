import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
import tqdm 

from dataset import TrajDataset, ConstantConsTrajDataset
from new_model import PolytopeConstrainedFlowModel
from utils import ot_minibatch_coupling, set_all_seed


def train_discrete_delta(dataset: TrajDataset, iteration=2000, lr=1e-4, batch_size=50, steps=10, use_ot=True, device="cuda:0"):
    """
    训练模型直接预测两个离散时间点之间的差值 (Delta x)
    Target = x_{k+1} - x_k
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Prediction Target: Delta x (Steps={steps})")
    
    # load dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    max_seq = dataset.seq_length
    num_cons = dataset.num_cons
    x_dim = dataset.x_dim

    model = PolytopeConstrainedFlowModel(
        x_dim=x_dim, num_cons=num_cons, num_rays=x_dim+1, max_seq=max_seq,
        use_block_mask_cons=True, use_block_mask_cross=True,
        use_block_mask_weight=True,
        device=device
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iteration, eta_min=1e-6)
    
    model.train()
    loss_history = []
    
    t_iter = tqdm.trange(iteration)
    data_iter = iter(dataloader)
    
    for i in t_iter:
        try:
            batch_data_dict = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_data_dict = next(data_iter)

        batch_traj = batch_data_dict['traj'].float().to(device) # [batch_size, seq_length*x_dim]
        batch_A = batch_data_dict['A'].float().to(device) # [batch_size, seq_length, num_cons, x_dim]
        batch_b = batch_data_dict['b'].float().to(device) # [batch_size, seq_length, num_cons]

        x1 = batch_traj
        x0, _, _ = dataset.generate_prior_data(batch_size=x1.shape[0], A=batch_A, b=batch_b)
        x0 = x0.float().to(device)
        
        if use_ot:
            x0, x1 = ot_minibatch_coupling(x0, x1)

        # === 1. 采样离散时间步 k ===
        # 范围是 [0, steps-1]
        k = torch.randint(0, steps, (x1.size(0),), device=device)
        
        # === 2. 计算当前时刻 t_k 和 下一时刻 t_{k+1} ===
        t_curr = (k.float() / steps).unsqueeze(1)       # [B, 1]
        t_next = ((k.float() + 1) / steps).unsqueeze(1) # [B, 1]
        
        # === 3. 构造两个时刻的坐标 (Ground Truth) ===
        # 使用线性插值 (Optimal Transport Path)
        x_curr = (1 - t_curr) * x0 + t_curr * x1
        x_next = (1 - t_next) * x0 + t_next * x1
        
        # === 4. 计算训练目标: 离散增量 Delta ===
        target_delta = x_next - x_curr
        
        # === 5. 模型预测 ===
        # 输入当前位置 x_curr 和当前时间 t_curr
        # 输出直接拟合 delta
        pred_delta, _, _ = model(x_curr, t_curr.squeeze(-1), batch_A, batch_b)
        
        loss = F.mse_loss(pred_delta, target_delta)
        
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪可选，防止 Ray Shooting 在极其靠近边界时产生大梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        t_iter.set_description(f"Loss: {loss.item():.4f}")

    return model, loss_history


@torch.no_grad()
def sample_discrete_delta(model: PolytopeConstrainedFlowModel, dataset: TrajDataset, n_samples=10, steps=10):
    """
    采样过程: x_{k+1} = x_k + Model(x_k, t_k)
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 1. 初始化噪声和约束condition
    # x [B, seq_length*x_dim]
    # A [B, seq_length, num_cons, x_dim]
    # b [B, seq_length, num_cons]
    x, A, b = dataset.generate_prior_data(batch_size=n_samples)
    x, A, b = x.float().to(device), A.float().to(device), b.float().to(device)
    
    traj_history = [x.cpu().numpy()]
    
    print(f"Sampling with Delta Prediction ({steps} steps)...")
    
    for k in range(steps):
        # 构造当前时间 t (归一化到 [0, 1])
        t_curr = torch.ones(n_samples, device=device) * (k / steps)
        
        # 2. 模型预测 Delta
        pred_delta, _, _ = model(x, t_curr, A, b)
        
        # 3. 更新位置 (直接相加)
        x = x + pred_delta
        
        # 记录轨迹
        traj_history.append(x.cpu().numpy())
        
    return np.array(traj_history)

