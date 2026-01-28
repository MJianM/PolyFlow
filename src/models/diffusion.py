from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers

import src.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, env_name='hopper_cpx', safe_method='RoS', 
        height_limit=1.5, vel_scale=0.01, height_min=0.8, v_max=2.5, v_min=-2.5,
        obs_vel_idx=6,
        leg_limit=1.2, torsion_limit=0.8
    ):
        super().__init__()
        self.means = 0  # for normalization
        self.stds = 0
        self.act_means = 0 # for normalization action
        self.act_stds = 0
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        self.env_name = env_name
        self.safe_method = safe_method
        self.height_limit = height_limit
        self.vel_scale = vel_scale
        self.height_min = height_min
        self.v_max = v_max
        self.v_min = v_min
        self.obs_vel_idx = obs_vel_idx
        self.leg_limit = leg_limit
        self.torsion_limit = torsion_limit
        assert self.safe_method in ['RoS', 'none']
        assert self.env_name in ['hopper', 'hopper_cpx', 'hopper_cpx2', 'walker2d', 'walker2d_cpx', 'walker2d_cpx2', 'halfcheetah']
        print(f"Env name: {self.env_name}  Safe method: {self.safe_method}")
        if 'cpx' in self.env_name:
            print(f"Height max: {self.height_limit}  Vel scale: {self.vel_scale}  Height min: {self.height_min}  V max: {self.v_max}  V min: {self.v_min}")

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs): 
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x_t = x.clone()
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)


            if self.env_name == 'hopper_cpx' or self.env_name == 'walker2d_cpx':
                if self.safe_method == 'RoS':
                    x, b_min = self.invariance_cpx_batch(x_t, x)

            elif self.env_name == 'hopper_cpx2' or self.env_name == 'walker2d_cpx2':
                if self.safe_method == 'RoS':
                    x, b_min = self.invariance_cpx2_batch(x_t, x)

            elif self.env_name == 'halfcheetah':
                if self.safe_method == 'RoS':
                    x, b_min = self.invariance_halfcheetah_batch(x_t, x)

            
            x = apply_conditioning(x, cond, self.action_dim)


            if return_chain: chain.append(x)

        progress.stamp()
        # pdb.set_trace()  #unx = x[0,:,6:].cpu().numpy()*self.std + self.mean

        # x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        b_min = 0
        return Sample(x, values, chain), b_min

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs): 
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, return_chain = True, **sample_kwargs)    # debug

    @torch.no_grad()   # 仅用于采样
    def invariance_cpx_batch(self, x, xp1):  # RoS diffuser with complex safety specification (hopper/walker2d)
        """
        z + vel_scale * vz <= height_limit
        """
        batch_size, horizon, dim = x.shape
        
        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)
        n_total = x_flat.shape[0]

        ref = xp1_flat - x_flat


        vel_scale = self.vel_scale * self.stds[self.obs_vel_idx] / self.stds[0]
        height = (self.height_limit - self.means[0]) / self.stds[0] \
            - self.vel_scale * self.means[self.obs_vel_idx] / self.stds[0]
        z_idx = self.action_dim
        vz_idx = self.action_dim+self.obs_vel_idx

        # CBF: b = height - pos - 0.1*vel
        b = height - x_flat[:, z_idx:z_idx+1] - vel_scale * x_flat[:, vz_idx:vz_idx+1]
        
        Lfb = 0 
        Lgbu1 = -1 * torch.ones_like(x_flat[:, z_idx:z_idx+1])
        Lgbu2 = -vel_scale * torch.ones_like(x_flat[:, z_idx:z_idx+1]) 
  
        G = torch.cat([-Lgbu1, -Lgbu2], dim=1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k * b
        
        # Q, q 
        q = -torch.cat([ref[:, z_idx:z_idx+1], ref[:, vz_idx:vz_idx+1]], dim=1).to(G.device)
        
        # Q : (B*H, 2, 2)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(n_total, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        # out : (B*H, 2)
        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt_flat = xp1_flat.clone()

        rt_flat[:, z_idx:z_idx+1] = x_flat[:, z_idx:z_idx+1] + out[:, 0:1]
        rt_flat[:, vz_idx:vz_idx+1] = x_flat[:, vz_idx:vz_idx+1] + out[:, 1:2]
        
        rt = rt_flat.reshape(batch_size, horizon, dim)
        return rt, torch.min(b)

    @torch.no_grad()
    def invariance_cpx2_batch(self, x, xp1):
        """
        Math Formulation:
            Constraint: \nabla b(x)^T * u + \alpha(b(x)) >= 0
            QP Form:    -\nabla b(x)^T * u <= \alpha(b(x))
        
        Args:
            x:   
            xp1: 
        """
        batch_size, horizon, dim = x.shape
        n_total = batch_size * horizon
        
        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)
        
        ref = xp1_flat - x_flat

        gamma_coef = 1.0 
        
        std_z = self.stds[0]
        std_v = self.stds[self.obs_vel_idx]
        mean_z = self.means[0]
        mean_v = self.means[self.obs_vel_idx]


        vel_scale_norm = self.vel_scale * std_v / std_z
        
        # Norm_Limit = (Phys_Limit - Mean) / Std
        h_max_const = (self.height_limit - mean_z - self.vel_scale * mean_v) / std_z
        h_min_norm = (self.height_min - mean_z) / std_z
        v_max_norm = (self.v_max - mean_v) / std_v
        v_min_norm = (self.v_min - mean_v) / std_v

        z_idx = self.action_dim   # Position Z index
        vz_idx = self.action_dim + self.obs_vel_idx  # Velocity Z index

        z_curr = x_flat[:, z_idx:z_idx+1]
        vz_curr = x_flat[:, vz_idx:vz_idx+1]

        
        # b1: Momentum Ceiling (Limit - (z + c*v))
        b1 = h_max_const - (z_curr + vel_scale_norm * vz_curr)
        
        # b2: Height Floor (z - Min)
        b2 = z_curr - h_min_norm
        
        # b3: Velocity Max (Limit - v)
        b3 = v_max_norm - vz_curr
        
        # b4: Velocity Min (v - Limit)
        b4 = vz_curr - v_min_norm
        

        b_vals = torch.cat([b1, b2, b3, b4], dim=1) # (N, 4)

        
        ones = torch.ones_like(z_curr)
        zeros = torch.zeros_like(z_curr)

        # --- Row 1: b1 = Const - z - scale*v ---
        # \nabla b1 = [-1, -scale]
        # - \nabla b1 = [1, scale]
        g1 = torch.cat([ones, ones * vel_scale_norm], dim=1).unsqueeze(1)
        
        # --- Row 2: b2 = z - Const ---
        # \nabla b2 = [1, 0]
        # - \nabla b2 = [-1, 0]
        g2 = torch.cat([-ones, zeros], dim=1).unsqueeze(1)
        
        # --- Row 3: b3 = Const - v ---
        # \nabla b3 = [0, -1]
        # - \nabla b3 = [0, 1]
        g3 = torch.cat([zeros, ones], dim=1).unsqueeze(1)
        
        # --- Row 4: b4 = v - Const ---
        # \nabla b4 = [0, 1]
        # - \nabla b4 = [0, -1]
        g4 = torch.cat([zeros, -ones], dim=1).unsqueeze(1)

        # (N, 4, 2)
        G = torch.cat([g1, g2, g3, g4], dim=1).to(x.device)

        h_qp = gamma_coef * b_vals 

        
        ref_subset = torch.cat([ref[:, z_idx:z_idx+1], ref[:, vz_idx:vz_idx+1]], dim=1)
        
        Q = Variable(torch.eye(2)).unsqueeze(0).expand(n_total, 2, 2).to(x.device)
        q = -ref_subset # qpth minimize 0.5 xQx + qx


        e = Variable(torch.Tensor()).to(x.device)
        
        try:
            out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(Q, q, G, h_qp, e, e)
        except Exception:
            # 
            print(f"QP solver failed!")
            out = torch.zeros_like(ref_subset)

        rt_flat = xp1_flat.clone()
        
        # x_{k-1} = x_k + u
        rt_flat[:, z_idx:z_idx+1] = x_flat[:, z_idx:z_idx+1] + out[:, 0:1]
        rt_flat[:, vz_idx:vz_idx+1] = x_flat[:, vz_idx:vz_idx+1] + out[:, 1:2]
        
        rt = rt_flat.reshape(batch_size, horizon, dim)
        
        min_barrier, _ = torch.min(b_vals, dim=1)
        
        return rt, min_barrier

    def get_halfcheetah_normed_cons(self):

        back_knee_mean, back_knee_std = self.act_means[1], self.act_stds[1]
        back_thigh_mean, back_thigh_std = self.act_means[0], self.act_stds[0]
        front_knee_mean, front_knee_std = self.act_means[4], self.act_stds[4]
        front_thigh_mean, front_thigh_std = self.act_means[3], self.act_stds[3]

        # u0 + back_knee_scale * u1 <= back_leg_limit
        back_knee_scale = back_knee_std / back_thigh_std
        back_leg_limit = (self.leg_limit - back_thigh_mean - back_knee_mean) / back_thigh_std

        # u3 + front_knee_scale * u4 <= front_leg_limit
        front_knee_scale = front_knee_std / front_thigh_std
        front_leg_limit = (self.leg_limit - front_thigh_mean - front_knee_mean) / front_thigh_std

        # u0 + front_thigh_scale * u3 <= front_torsion_limit 
        front_thigh_scale = - front_thigh_std / back_thigh_std
        front_torsion_limit = (self.torsion_limit - (back_thigh_mean - front_thigh_mean)) / back_thigh_std

        # -u0 - front_thigh_scale * u3 <= front_torsion_limit2
        front_torsion_limit2 = (self.torsion_limit - (front_thigh_mean - back_thigh_mean)) / back_thigh_std

        u1_scale = back_knee_scale
        u1_limit = back_leg_limit
        u4_scale = front_knee_scale
        u4_limit = front_leg_limit
        u3_scale = front_thigh_scale
        u3_limit = front_torsion_limit
        u3_scale2 = -front_thigh_scale
        u3_limit2 = front_torsion_limit2

        return u1_scale, u1_limit, u4_scale, u4_limit, \
            u3_scale, u3_limit, u3_scale2, u3_limit2

    @torch.no_grad()
    def invariance_halfcheetah_batch(self, x, xp1):
        """
        SafeDiffuser implementation for HalfCheetah with 4 linear constraints.
        
        Math Formulation:
            Constraint: \nabla b(x)^T * u + \alpha(b(x)) >= 0
            QP Form:    -\nabla b(x)^T * u <= \alpha(b(x))
            
        Constraints (h(x) >= 0 for safety):
        1. u1_limit - (x0 + u1_scale * x1) >= 0
        2. u4_limit - (x3 + u4_scale * x4) >= 0
        3. u3_limit - (x0 + u3_scale * x3) >= 0
        4. u3_limit2 - (-x0 + u3_scale2 * x3) >= 0  
        
        Args:
            x: Current state (denoising step k)
            xp1: Proposed next state (step k-1)
        """
        batch_size, horizon, dim = x.shape
        n_total = batch_size * horizon
        

        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)
        

        ref = xp1_flat - x_flat
        

        u1_scale, u1_limit, u4_scale, u4_limit, \
            u3_scale, u3_limit, u3_scale2, u3_limit2 = self.get_halfcheetah_normed_cons()

        gamma_coef = 1.0  

        # Variable mapping: 
        # idx 0 -> x0
        # idx 1 -> x1
        # idx 3 -> x3
        # idx 4 -> x4
        x0 = x_flat[:, 0:1]
        x1 = x_flat[:, 1:2]
        x3 = x_flat[:, 3:4]
        x4 = x_flat[:, 4:5]

        
        # C1: x0 + s1 * x1 <= L1
        b1 = u1_limit - (x0 + u1_scale * x1)
        
        # C2: x3 + s4 * x4 <= L4
        b2 = u4_limit - (x3 + u4_scale * x4)
        
        # C3: x0 + s3 * x3 <= L3
        b3 = u3_limit - (x0 + u3_scale * x3)
        
        # C4: -x0 + s3b * x3 <= L3b
        # Barrier = Limit - (-x0 + s3b * x3) = Limit + x0 - s3b * x3
        b4 = u3_limit2 - (-x0 + u3_scale2 * x3) 
        
        b_vals = torch.cat([b1, b2, b3, b4], dim=1)

        
        ones = torch.ones_like(x0)
        zeros = torch.zeros_like(x0)
        
        # --- Row 1 (C1): b1 = L1 - x0 - s1*x1 ---
        # Grad = [-1, -s1, 0, 0]
        # -Grad = [1, s1, 0, 0]
        g1 = torch.cat([ones, ones * u1_scale, zeros, zeros], dim=1).unsqueeze(1)
        
        # --- Row 2 (C2): b2 = L4 - x3 - s4*x4 ---
        # Grad = [0, 0, -1, -s4]
        # -Grad = [0, 0, 1, s4]
        g2 = torch.cat([zeros, zeros, ones, ones * u4_scale], dim=1).unsqueeze(1)
        
        # --- Row 3 (C3): b3 = L3 - x0 - s3*x3 ---
        # Grad = [-1, 0, -s3, 0]
        # -Grad = [1, 0, s3, 0]
        g3 = torch.cat([ones, zeros, ones * u3_scale, zeros], dim=1).unsqueeze(1)
        
        # --- Row 4 (C4): b4 = L3b - (-x0) - s3b*x3 = L3b + x0 - s3b*x3 ---
        # Grad = [1, 0, -s3b, 0]
        # -Grad = [-1, 0, s3b, 0]
        g4 = torch.cat([-ones, zeros, ones * u3_scale2, zeros], dim=1).unsqueeze(1)
        
        # G (N, 4, 4) -> 4 Constraints, 4 Variables
        G = torch.cat([g1, g2, g3, g4], dim=1).to(x.device)


        h_qp = gamma_coef * b_vals


        ref_subset = torch.cat([
            ref[:, 0:1], 
            ref[:, 1:2], 
            ref[:, 3:4], 
            ref[:, 4:5]
        ], dim=1)
        
        # Q = I (4x4)
        reduced_dim = 4
        Q = Variable(torch.eye(reduced_dim)).unsqueeze(0).expand(n_total, reduced_dim, reduced_dim).to(x.device)
        q = -ref_subset


        e = Variable(torch.Tensor()).to(x.device)
        
        try:
            # out: [u_0, u_1, u_3, u_4]
            out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(Q, q, G, h_qp, e, e)
        except Exception:
            # Fallback: 
            print(f"QP solver failed!")
            out = torch.zeros_like(ref_subset)


        rt_flat = xp1_flat.clone()
        
        rt_flat[:, 0:1] = x_flat[:, 0:1] + out[:, 0:1] # x0
        rt_flat[:, 1:2] = x_flat[:, 1:2] + out[:, 1:2] # x1
        rt_flat[:, 3:4] = x_flat[:, 3:4] + out[:, 2:3] # x3 
        rt_flat[:, 4:5] = x_flat[:, 4:5] + out[:, 3:4] # x4
        
        rt = rt_flat.reshape(batch_size, horizon, dim)
        
        min_barrier, _ = torch.min(b_vals, dim=1)
        
        return rt, min_barrier
 
    @torch.no_grad()   #only for sampling
    def invariance_cheetah(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        radius = 0.4
        radius = (radius - self.mean[0]) / self.std[0]
        cx = 4
        cy = -0.2
        cx = (cx - self.mean[14]) / self.std[14]
        cy = (cy - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        xpos = torch.cumsum(x[:,14:15], dim=0) * 0.05

        b = (xpos - cx)**2 + (x[:,6:7] - cy)**2 - radius**2 
        Lfb = 0 
        Lgbu1 = 2*(x[:,6:7] - cy)
  
        G = torch.cat([-Lgbu1], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,6:7]], dim = 1).to(G.device) 
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]

        rt = rt.unsqueeze(0)
        return rt, torch.min(b)
    
    @torch.no_grad()   #only for sampling
    def invariance_cheetah_cpx(self, x, xp1):

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.4
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] - 0.1*x[:,15:16] 
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,6:7], ref[:,15:16]], dim = 1).to(G.device)  #
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        rt[:,15:16] = x[:,15:16] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)


    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)


class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)

