from collections import namedtuple
import torch
import einops
import pdb
import time

import src.utils as utils
from src.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        if hasattr(self.normalizer.normalizers['observations'], 'means'):
            self.diffusion_model.means = torch.from_numpy(self.normalizer.normalizers['observations'].means).to(self.device).float()
            self.diffusion_model.stds = torch.from_numpy(self.normalizer.normalizers['observations'].stds).to(self.device).float()
        else:
            self.diffusion_model.norm_mins = torch.from_numpy(self.normalizer.normalizers['observations'].mins).to(self.device).float()
            self.diffusion_model.norm_maxs = torch.from_numpy(self.normalizer.normalizers['observations'].maxs).to(self.device).float()

        # --- 计时开始 ---
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # 等待数据传输等之前的所有 GPU 操作完成
        start_time = time.time()
        # ----------------
        if self.guide is None:
            samples, b_min = self.diffusion_model(conditions, verbose=verbose, **self.sample_kwargs)  # debug
        else:
            samples, b_min = self.diffusion_model(conditions, verbose=verbose, guide=self.guide, **self.sample_kwargs)

        # --- 计时结束 ---
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # 等待最后一步 GPU 运算完成
        end_time = time.time()
        # ----------------

        # 计算统计数据
        total_time = end_time - start_time
        avg_time_per_step = total_time / (self.diffusion_model.n_timesteps)

        trajectories = utils.to_np(samples.trajectories) 
        diffusion = utils.to_np(samples.chains)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        normed_diffusion = diffusion[:, :, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        diffusion_obs = self.normalizer.unnormalize(normed_diffusion, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories, diffusion_obs, b_min, total_time, avg_time_per_step

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        
        tmp_k = next(iter(conditions.keys()))
        if len(conditions[tmp_k].shape) == 1:
            conditions = utils.apply_dict(
                einops.repeat,
                conditions,
                'd -> repeat d', repeat=batch_size,
            )
        return conditions
    

class PolyFlowPolicy(GuidedPolicy):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, 
            dataset=None):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns)

        self.dataset = dataset

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        if hasattr(self.normalizer.normalizers['observations'], 'means'):
            self.diffusion_model.means = torch.from_numpy(self.normalizer.normalizers['observations'].means).to(self.device).float()
            self.diffusion_model.stds = torch.from_numpy(self.normalizer.normalizers['observations'].stds).to(self.device).float()
        else:
            self.diffusion_model.norm_mins = torch.from_numpy(self.normalizer.normalizers['observations'].mins).to(self.device).float()
            self.diffusion_model.norm_maxs = torch.from_numpy(self.normalizer.normalizers['observations'].maxs).to(self.device).float()

        # --- 计时开始 ---
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # 等待数据传输等之前的所有 GPU 操作完成
        start_time = time.time()
        # ----------------
        x0, A, b = self.dataset.generate_prior_data(batch_size=batch_size, device=self.device)
        if self.guide is None:
            samples, b_min = self.diffusion_model(conditions, verbose=verbose, A=A, b=b, x0=x0, **self.sample_kwargs)  # debug
        else:
            samples, b_min = self.diffusion_model(conditions, verbose=verbose, A=A, b=b, x0=x0, guide=self.guide, **self.sample_kwargs)

        # --- 计时结束 ---
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # 等待最后一步 GPU 运算完成
        end_time = time.time()
        # ----------------

        # 计算统计数据
        total_time = end_time - start_time
        avg_time_per_step = total_time / (self.diffusion_model.n_timesteps)

        trajectories = utils.to_np(samples.trajectories) 
        diffusion = utils.to_np(samples.chains)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[:, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        normed_diffusion = diffusion[:, :, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        diffusion_obs = self.normalizer.unnormalize(normed_diffusion, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories, diffusion_obs, b_min, total_time, avg_time_per_step

class Go2PolyFlowPolicy(GuidedPolicy):

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, 
            dataset=None):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns)

        self.dataset = dataset

    def __call__(self, conditions, A_0, b_0, contact_0, vertex_0, batch_size=1, verbose=True):
        """
        Docstring for __call__
        
        A_0: (batch, 1, 4, num_cons, 3) Tensor cpu
        b_0: (batch, 1, 4, num_cons,) Tensor cpu
        contact_0: (batch, 1, 4,) Tensor cpu
        vertex_0: (batch, 1, 4, 3,) Tensor cpu

        """
        
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        A_0_normed, b_0_normed = self.dataset.normalize_constraints_tensor(A_0, b_0)

        ## run reverse diffusion process
        if hasattr(self.normalizer.normalizers['observations'], 'means'):
            self.diffusion_model.means = torch.from_numpy(self.normalizer.normalizers['observations'].means).to(self.device).float()
            self.diffusion_model.stds = torch.from_numpy(self.normalizer.normalizers['observations'].stds).to(self.device).float()
        else:
            self.diffusion_model.norm_mins = torch.from_numpy(self.normalizer.normalizers['observations'].mins).to(self.device).float()
            self.diffusion_model.norm_maxs = torch.from_numpy(self.normalizer.normalizers['observations'].maxs).to(self.device).float()

        # --- 计时开始 ---
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # 等待数据传输等之前的所有 GPU 操作完成
        start_time = time.time()
        # ----------------

        x0, A, b, contact = self.dataset.generate_prior_data(batch_size=batch_size, A_0=A_0_normed.to(self.device), b_0=b_0_normed.to(self.device), contact_0=contact_0.to(self.device), device=self.device, h_0=vertex_0.to(self.device))
        if self.guide is None:
            samples, b_min = self.diffusion_model(conditions, verbose=verbose, A=A, b=b, contact=contact, x0=x0, **self.sample_kwargs)  # debug
        else:
            samples, b_min = self.diffusion_model(conditions, verbose=verbose, A=A, b=b, contact=contact, x0=x0, guide=self.guide, **self.sample_kwargs)

        # --- 计时结束 ---
        if self.device.type == 'cuda':
            torch.cuda.synchronize() # 等待最后一步 GPU 运算完成
        end_time = time.time()
        # ----------------

        # 计算统计数据
        total_time = end_time - start_time
        avg_time_per_step = total_time / (self.diffusion_model.n_timesteps)

        trajectories = utils.to_np(samples.trajectories) 
        diffusion = utils.to_np(samples.chains)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[:, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        normed_diffusion = diffusion[:, :, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        diffusion_obs = self.normalizer.unnormalize(normed_diffusion, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories, diffusion_obs, b_min, total_time, avg_time_per_step
