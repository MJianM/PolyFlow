import csv
import os
import numpy as np

def flatten_metrics(metrics, horizon_list):
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, (list, np.ndarray)):
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

        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics_dict)

class CSVLogger:
    def __init__(self, log_dir, filename="progress.csv"):
        self.save_path = os.path.join(log_dir, filename)
        self.headers = None
        self.file = None
        
    def log(self, metrics_dict):
        if self.headers is None:
            self.headers = list(metrics_dict.keys())
            file_exists = os.path.isfile(self.save_path)
            
            self.file = open(self.save_path, 'a', newline='')
            self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
            
            if not file_exists:
                self.writer.writeheader()
        
        self.writer.writerow(metrics_dict)
        self.file.flush() 

    def close(self):
        if self.file:
            self.file.close()