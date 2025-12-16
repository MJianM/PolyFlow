import numpy as np
import torch

from utils import set_all_seed
from dataset import ConstantConsTrajDataset
from train import train_discrete_delta, sample_discrete_delta
from visual import plot_results, plot_simple_loss


def ceil(num, scale=100):
    scaled_num = num * scale
    ceiled_num = np.ceil(scaled_num)
    return ceiled_num / scale

def floor(num, scale=100):
    scaled_num = num * scale
    ceiled_num = np.floor(scaled_num)
    return ceiled_num / scale

def save_dataset_reshaped(data):
    # 1. 获取基本维度信息
    num_trajs, seq_length, x_dim = data.shape
    M, T, D = num_trajs, seq_length, x_dim
    
    SPLIT_TIME = 200
    
    # 2. 计算边界 (逻辑保持不变，计算随时间变化的包围盒)
    # up_b 和 low_b 的形状均为 (T, D)
    up_b = np.zeros((T, D))
    low_b = np.zeros((T, D))
    
    # --- 阶段 1: t < 200 ---
    time_mask_1 = np.arange(T) < SPLIT_TIME
    # 计算该阶段所有数据的全局最大/最小值
    up_bound_1 = ceil(np.max(data[:, :SPLIT_TIME], axis=(0, 1)))
    low_bound_1 = floor(np.min(data[:, :SPLIT_TIME], axis=(0, 1)))
    # 填充到对应时间步
    up_b[time_mask_1, :] = up_bound_1
    low_b[time_mask_1, :] = low_bound_1

    # --- 阶段 2: t >= 200 ---
    time_mask_2 = np.arange(T) >= SPLIT_TIME
    # 计算该阶段所有数据的全局最大/最小值
    up_bound_2 = ceil(np.max(data[:, SPLIT_TIME:], axis=(0, 1)))
    low_bound_2 = floor(np.min(data[:, SPLIT_TIME:], axis=(0, 1)))
    # 填充到对应时间步
    up_b[time_mask_2, :] = up_bound_2
    low_b[time_mask_2, :] = low_bound_2

    # 3. 构建 b 矩阵 (num_trajs, seq_length, num_cons)
    # ---------------------------------------------------------
    # 对于每个维度 d，我们有两个约束：
    # 1. x_d <= up_b_d
    # 2. x_d >= low_b_d  <=>  -x_d <= -low_b_d
    # 我们将它们堆叠起来。
    # b_seq 形状: (T, 2*D) -> 前 D 个是上界，后 D 个是负下界
    b_seq = np.concatenate([up_b, -low_b], axis=-1) 
    
    # 扩展到所有轨迹 (Broadcast)
    # b shape: (M, T, 2*D)
    b = np.tile(b_seq[None, :, :], (M, 1, 1))

    # 4. 构建 A 矩阵 (num_trajs, seq_length, num_cons, x_dim)
    # ---------------------------------------------------------
    # 对应于 b 的结构，A 应该是 [I; -I] 的形式
    # I 对应 x <= U
    # -I 对应 -x <= -L
    
    I = np.eye(D)      # (D, D)
    A_local = np.concatenate([I, -I], axis=0) # (2*D, D)
    
    # 扩展到时间维度 (T, 2*D, D) - A 矩阵结构在所有时间步是一样的
    A_seq = np.tile(A_local[None, :, :], (T, 1, 1))
    
    # 扩展到轨迹维度 (M, T, 2*D, D)
    A = np.tile(A_seq[None, :, :, :], (M, 1, 1, 1))

    # 5. 保存
    print(f"Data Shape: {data.shape}")
    print(f"A Shape: {A.shape}  (M, T, 2D, D)")
    print(f"b Shape: {b.shape}  (M, T, 2D)")
    
    # 保存时，single_A 和 single_b 现在的维度包含了 Batch 信息
    np.savez_compressed('bound_hopper_traj_data.npz', traj_dataset=data, single_A=A, single_b=b)
    
    return A, b

def train():

    seed = 42
    iters = 100
    file_path = "bound_hopper_traj_data.npz"
    device = "cuda:4"

    set_all_seed(seed)

    # 原来轨迹长度是800, 把轨迹降采样到200
    dataset = ConstantConsTrajDataset(file_path=file_path, seq_length=200) # 把轨迹降采样到200

    model, loss_history = train_discrete_delta(
        dataset=dataset, iteration=iters, batch_size=100, lr=1e-4, steps=10, use_ot=True, device=device
    )
    
    plot_simple_loss(loss_history, save_path='hopper_train_loss.png')
    torch.save(model, "hopper_final_model.pt")

    generated_traj = sample_discrete_delta(model, dataset, n_samples=10, steps=10)[-1] # (n_samples, seq_length*x_dim)

    # eva
    #TODO
    #使用gym mujoco render功能，不进行仿真step,只做可视化
    #TODO
    #把高度曲线画出来

if __name__=="__main__":

    train()