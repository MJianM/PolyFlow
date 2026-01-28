import os
import json
import csv
import numpy as np

def flatten_metrics(metrics, horizon_list):
    """
    输入: {'mmd': [0.1, 0.2], 'safety': 0.9}
    输出: {'mmd_t0': 0.1, 'mmd_t5': 0.2, 'safety': 0.9}
    """
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
        """
        metrics_dict: key-value 形式的标量字典
        """
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



class Logger:

    def __init__(self, renderer, logpath, vis_freq=10, max_render=8):
        self.renderer = renderer
        self.savepath = logpath
        self.vis_freq = vis_freq
        self.max_render = max_render

    def log(self, t, samples, state, rollout=None, diffusion = None):
        if t % self.vis_freq != 0:
            return

        ## render image of plans
        self.renderer.composite(
            os.path.join(self.savepath, f'{t}.png'),
            samples.observations,
        )

        ## render video of plans
        self.renderer.render_plan(
            os.path.join(self.savepath, f'{t}_plan.mp4'),
            samples.actions[:self.max_render],
            samples.observations[:self.max_render],
            state,
        )

        if rollout is not None:
            ## render video of rollout thus far
            self.renderer.render_rollout(
                os.path.join(self.savepath, f'rollout.mp4'),
                rollout,
                fps=80,
            )

        # if diffusion is not None:
        #     ## render video of diffusion step
        #     self.renderer.render_diffusion_samp(
        #         self.savepath+"/png/",
        #         diffusion,
        #         fps = 200,
        #     )
        # import pdb;
        # pdb.set_trace()

        if diffusion is not None:
            ## render video of diffusion step
            self.renderer.render_diffusion_samp_c(
                os.path.join(self.savepath, f'diffusion.mp4'),
                diffusion,
                fps = 200,
            )
        import pdb;
        pdb.set_trace()

    def finish(self, t, score, total_reward, terminal, diffusion_experiment, value_experiment):
        json_path = os.path.join(self.savepath, 'rollout.json')
        json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
            'epoch_diffusion': diffusion_experiment.epoch, 'epoch_value': value_experiment.epoch}
        json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
        print(f'[ utils/logger ] Saved log to {json_path}')
