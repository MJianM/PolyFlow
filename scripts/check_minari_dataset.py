import minari
import numpy as np
import os

def analyze_dataset_returns(dataset_id):
    """
    加载本地 Minari 数据集并计算 Return 的均值、方差、最大值和最小值。
    """
    
    # 1. 检查数据集是否在本地存在
    # Minari 默认将数据存储在本地的根目录下。
    # 这里我们直接尝试加载，如果不存在 catch 异常会更稳健。
    print(f"--- 正在尝试加载数据集: {dataset_id} ---")
    
    try:
        # 加载数据集
        dataset = minari.load_dataset(dataset_id)
    except ValueError:
        print(f"错误: 找不到数据集 '{dataset_id}'。")
        print("请使用 `minari list remote` 查看可用数据，并使用 `minari download <dataset_id>` 先下载到本地。")
        return
    except Exception as e:
        print(f"发生未知错误: {e}")
        return

    # 2. 遍历所有 episode 并计算 Return
    # Return (G) 通常定义为一个 episode 中所有 reward 的总和
    all_returns = []
    
    # Minari dataset 是一个可迭代对象，包含 Episode 对象
    for i, episode in enumerate(dataset):
        # episode.rewards 是一个 numpy 数组
        episode_return = np.sum(episode.rewards)
        all_returns.append(episode_return)

    # 转换为 numpy 数组以便计算统计量
    all_returns = np.array(all_returns)
    
    # 3. 计算统计量
    if len(all_returns) == 0:
        print("警告: 数据集中没有 episode 数据。")
        return

    mean_ret = np.mean(all_returns)
    std_ret = np.std(all_returns)
    max_ret = np.max(all_returns)
    min_ret = np.min(all_returns)

    # 4. 打印结果
    print(f"\n数据集分析结果 ({dataset_id}):")
    print(f"总 Episode 数量: {len(all_returns)}")
    print("-" * 30)
    print(f"Return 均值 (Mean):     {mean_ret:.4f}")
    print(f"Return 标准差 (Std):    {std_ret:.4f}")
    print(f"Return 最大值 (Max):    {max_ret:.4f}")
    print(f"Return 最小值 (Min):    {min_ret:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    # === 在这里修改你要加载的数据集名称 ===

    TARGET_DATASET_ID = "mujoco/halfcheetah/medium-v0" 
    analyze_dataset_returns(TARGET_DATASET_ID)