import csv
import os
import numpy as np

def flatten_metrics(metrics, horizon_list):
    """
    将包含列表的字典拍平成标量字典
    输入: {'mmd': [0.1, 0.2], 'safety': 0.9}
    输出: {'mmd_t0': 0.1, 'mmd_t5': 0.2, 'safety': 0.9}
    """
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (list, np.ndarray)):
            # 如果是列表，按时间步拆分
            for i, t in enumerate(horizon_list):
                flat[f"{k}_t{t}"] = float(v[i])
        else:
            flat[k] = float(v)
    return flat

def save_csv_native(metrics_dict, save_path="metrics.csv"):
    file_exists = os.path.isfile(save_path)
    fieldnames = list(metrics_dict.keys())

    with open(save_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # 如果文件不存在，先写表头
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics_dict)

class CSVLogger:
    def __init__(self, log_dir, filename="progress.csv"):
        self.save_path = os.path.join(log_dir, filename)
        self.headers = None
        self.file = None
        
    def log(self, metrics_dict):
        """
        metrics_dict: key-value 形式的标量字典
        """
        # 第一次调用时，确定表头并创建文件
        if self.headers is None:
            self.headers = list(metrics_dict.keys())
            file_exists = os.path.isfile(self.save_path)
            
            # 打开文件准备追加
            self.file = open(self.save_path, 'a', newline='')
            self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
            
            if not file_exists:
                self.writer.writeheader()
        
        # 写入数据
        self.writer.writerow(metrics_dict)
        self.file.flush() # 立即写入磁盘，防止程序崩溃数据丢失

    def close(self):
        if self.file:
            self.file.close()

# === 在 train.py 中使用 ===
# logger = CSVLogger(log_dir="outputs/...")
# for i in range(100):
#     logger.log({"iter": i, "loss": 0.5})
# logger.close()