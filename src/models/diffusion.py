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

            ##########################################walker2d
            # x, b_min = self.GD(x_t, x)  # truncate method
            # x, b_min = self.Shield(x_t, x)  # classifier guidance or potential-based method
            # x, b_min = self.invariance(x_t, x)  # RoS diffuser
            # x, b_min = self.invariance_cf(x_t, x)   #RoS diffuser, closed form
            # x, b_min = self.invariance_cpx(x_t, x)  #RoS diffuser with complex safety specification
            # x, b_min = self.invariance_cpx_cf(x_t, x) #RoS diffuser with complex safety specification, closed form

            ##########################################hopper
            # x, b_min = self.GD_hopper(x_t, x) # truncate method
            # x, b_min = self.Shield_hopper(x_t, x)  # #classifier guidance or potential-based method
            # x, b_min = self.invariance_hopper(x_t, x)  # RoS diffuser
            # x, b_min = self.invariance_hopper_cf(x_t, x)   #RoS diffuser, closed form
            # x, b_min = self.invariance_hopper_cpx(x_t, x)  #RoS diffuser with complex specification
            # x, b_min = self.invariance_hopper_cpx_cf(x_t, x)  #RoS diffuser with complex specification, closed form


            ##########################################cheetah
            # x, b_min = self.invariance_cheetah(x_t, x)
            
            x = apply_conditioning(x, cond, self.action_dim)

            ############################ diffuser only, for evaluation purpose
            # height = 1.4   #1.3    walker2d
            # height = (height - self.mean[0]) / self.std[0]
            # b = height - x[:,6:7]  - 0.1*x[:,15:16]
            # b_min = torch.min(b)

            # height = 1.6   #1.5      hopper
            # height = (height - self.mean[0]) / self.std[0]
            # b = height - x[:,3:4] - 0.1*x[:,9:10]
            # b_min = torch.min(b)

            # progress.update({'t': i, 'vmin': values.min().item(), 'vmax': b_min.item()}) 
            # progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
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

###################################################################walker2d  
    @torch.no_grad()   #only for sampling
    def invariance(self, x, xp1):    # RoS diffuser

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.3
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] # - 0.1*x[:,15:16]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        #Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G = torch.cat([-Lgbu1], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,6:7]], dim = 1).to(G.device)  #, ref[:,15:16]
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        # rt[:,15:16] = x[:,15:16] + out[:,1:2]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_cf(self, x, xp1):  # RoS diffuser closed-form

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.3
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] # - 0.1*x[:,15:16]    # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        #Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G0 = torch.cat([-Lgbu1], dim = 1)
        k = 1
        h0 = Lfb + k*b

        Lgbu1 = 1*torch.ones_like(x[:,6:7])
        G1 = torch.cat([-Lgbu1], dim = 1)
        h1 = Lfb + k*(x[:,6:7] + 10)

        q = -torch.cat([ref[:,6:7]], dim = 1).to(G0.device)  #, ref[:,15:16]

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)

        return rt, torch.min(b)  # + 0.01  # for robustness
    
    
    @torch.no_grad()   #only for sampling
    def invariance_cpx(self, x, xp1):   # RoS diffuser with complex safety specification

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.4
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] - 0.1*x[:,15:16]   # - 0.01  # for robustness
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
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_cpx_cf(self, x, xp1): # RoS diffuser with complex safety specification, closed-form

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.4
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,6:7] - 0.1*x[:,15:16] # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,6:7])
        Lgbu2 = -0.1*torch.ones_like(x[:,6:7])
  
        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h0 = Lfb + k*b
        
        Lgbu1 = 1*torch.ones_like(x[:,6:7])
        Lgbu2 = 0.1*torch.ones_like(x[:,6:7])
        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        h1 = Lfb + k*(x[:,6:7] + 0.1*x[:,15:16] + 10)
   
        q = -torch.cat([ref[:,6:7], ref[:,15:16]], dim = 1).to(G0.device)  #

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar

        rt = xp1.clone()      
        rt[:,6:7] = x[:,6:7] + out[:,0:1]
        rt[:,15:16] = x[:,15:16] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b) # + 0.01  # for robustness

###################################################################hopper  

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

        # 复杂 CBF: b = height - pos - 0.1*vel
        b = height - x_flat[:, z_idx:z_idx+1] - vel_scale * x_flat[:, vz_idx:vz_idx+1]
        
        Lfb = 0 
        Lgbu1 = -1 * torch.ones_like(x_flat[:, z_idx:z_idx+1])
        Lgbu2 = -vel_scale * torch.ones_like(x_flat[:, z_idx:z_idx+1]) # 注意这里形状和 dim=0 的长度一致即可
  
        # G 形状: (B*H, 1, 2) -> 一个约束涉及两个变量
        G = torch.cat([-Lgbu1, -Lgbu2], dim=1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k * b
        
        # Q, q 针对两个变量 (位置, 速度)
        q = -torch.cat([ref[:, z_idx:z_idx+1], ref[:, vz_idx:vz_idx+1]], dim=1).to(G.device)
        
        # Q 形状: (B*H, 2, 2)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(n_total, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        # out 形状: (B*H, 2)
        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt_flat = xp1_flat.clone()
        # 更新位置和速度
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
            x:   当前去噪步的状态 (x_k^j in theorem)
            xp1: 模型预测的下一步状态 (suggestion for x_{k-1}^j)
        """
        batch_size, horizon, dim = x.shape
        n_total = batch_size * horizon
        
        # 1. 扁平化处理
        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)
        
        # ref 是模型建议的“控制量”或“步长” u_suggestion
        # 我们希望找到 u 接近 ref，且满足约束
        ref = xp1_flat - x_flat

        # ==========================================
        # 2. 参数准备与归一化
        # ==========================================
        # Class K function 的系数 gamma。SafeDiffuser 中通常取较大值以保证快速收敛，
        # 或者取 1.0 (离散时间死区控制)。这里设为 5.0 增强安全性。
        gamma_coef = 1.0 
        
        # 归一化参数
        std_z = self.stds[0]
        std_v = self.stds[self.obs_vel_idx]
        mean_z = self.means[0]
        mean_v = self.means[self.obs_vel_idx]

        # 归一化后的物理系数
        # 约束方程在物理空间是: z + vel_scale * v <= Limit
        # 在归一化空间，v 的系数变成了: vel_scale * (std_v / std_z)
        vel_scale_norm = self.vel_scale * std_v / std_z
        
        # 计算归一化后的约束边界常量
        # Norm_Limit = (Phys_Limit - Mean) / Std
        h_max_const = (self.height_limit - mean_z - self.vel_scale * mean_v) / std_z
        h_min_norm = (self.height_min - mean_z) / std_z
        v_max_norm = (self.v_max - mean_v) / std_v
        v_min_norm = (self.v_min - mean_v) / std_v

        # 关键索引
        z_idx = self.action_dim   # Position Z index
        vz_idx = self.action_dim + self.obs_vel_idx  # Velocity Z index

        # 提取当前状态
        z_curr = x_flat[:, z_idx:z_idx+1]
        vz_curr = x_flat[:, vz_idx:vz_idx+1]

        # ==========================================
        # 3. 计算 Barrier Function b(x)
        # Definition: b(x) >= 0 implies Safe
        # ==========================================
        
        # b1: Momentum Ceiling (Limit - (z + c*v))
        b1 = h_max_const - (z_curr + vel_scale_norm * vz_curr)
        
        # b2: Height Floor (z - Min)
        b2 = z_curr - h_min_norm
        
        # b3: Velocity Max (Limit - v)
        b3 = v_max_norm - vz_curr
        
        # b4: Velocity Min (v - Limit)
        b4 = vz_curr - v_min_norm
        
        # 这里的 h_vals 对应定理图片中的 b(x_k^j)
        b_vals = torch.cat([b1, b2, b3, b4], dim=1) # (N, 4)

        # ==========================================
        # 4. 构建梯度矩阵 G (Negative Gradients)
        # QP: G * u <= h_qp
        # Formula: - \nabla b(x) * u <= \alpha(b(x))
        # G 的行向量 = - \nabla b(x)
        # ==========================================
        
        # 我们需要计算 b(x) 对优化变量 u=[u_z, u_vz] 的梯度
        # 注意：u 是 x 的增量，所以 \nabla_x b 和 \nabla_u b 方向一致
        
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

        # 组合 G 矩阵 (N, 4, 2)
        G = torch.cat([g1, g2, g3, g4], dim=1).to(x.device)

        # ==========================================
        # 5. 构建 QP 向量 h_qp (\alpha(b(x)))
        # ==========================================
        
        # 定义 Extended Class K function \alpha(x) = \gamma * x
        # 这里的 h_qp 对应定理中的 \alpha(b(x_k^j))
        # 这决定了当 b(x) 接近 0 时，我们允许 u 违反边界的程度（余量）
        h_qp = gamma_coef * b_vals 

        # ==========================================
        # 6. QP 目标函数
        # min || u - ref ||^2  => min 0.5 * u^T I u - ref^T u
        # ==========================================
        
        # 取出仅与 Z, VZ 相关的建议步长
        ref_subset = torch.cat([ref[:, z_idx:z_idx+1], ref[:, vz_idx:vz_idx+1]], dim=1)
        
        Q = Variable(torch.eye(2)).unsqueeze(0).expand(n_total, 2, 2).to(x.device)
        q = -ref_subset # qpth minimize 0.5 xQx + qx

        # ==========================================
        # 7. 求解
        # ==========================================
        e = Variable(torch.Tensor()).to(x.device)
        
        try:
            # 求解出的 out 即为最优修正量 u
            out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(Q, q, G, h_qp, e, e)
        except Exception:
            # 求解失败时的 Fallback：不做修正或直接停止
            print(f"QP solver failed!")
            out = torch.zeros_like(ref_subset)

        # ==========================================
        # 8. 更新状态
        # ==========================================
        rt_flat = xp1_flat.clone()
        
        # x_{k-1} = x_k + u
        rt_flat[:, z_idx:z_idx+1] = x_flat[:, z_idx:z_idx+1] + out[:, 0:1]
        rt_flat[:, vz_idx:vz_idx+1] = x_flat[:, vz_idx:vz_idx+1] + out[:, 1:2]
        
        rt = rt_flat.reshape(batch_size, horizon, dim)
        
        # 返回修正轨迹和最小 Barrier 值（用于监控是否违反安全）
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
        4. u3_limit2 - (-x0 + u3_scale2 * x3) >= 0  <-- 注意这里 x0 前面的负号
        
        Args:
            x: Current state (denoising step k)
            xp1: Proposed next state (step k-1)
        """
        batch_size, horizon, dim = x.shape
        n_total = batch_size * horizon
        
        # 1. 扁平化处理
        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)
        
        # ref: 模型建议的步长 (u_suggestion)
        ref = xp1_flat - x_flat
        
        # ==========================================
        # 2. 参数获取与准备
        # ==========================================
        # 获取归一化后的约束参数
        u1_scale, u1_limit, u4_scale, u4_limit, \
            u3_scale, u3_limit, u3_scale2, u3_limit2 = self.get_halfcheetah_normed_cons()

        gamma_coef = 1.0  # SafeDiffuser 推荐系数 (Discrete time invariance)

        # 提取相关维度的当前状态
        # Variable mapping: 
        # idx 0 -> x0
        # idx 1 -> x1
        # idx 3 -> x3
        # idx 4 -> x4
        x0 = x_flat[:, 0:1]
        x1 = x_flat[:, 1:2]
        x3 = x_flat[:, 3:4]
        x4 = x_flat[:, 4:5]

        # ==========================================
        # 3. 计算 Barrier Function b(x)
        # Definition: b(x) >= 0 implies Safe
        # ==========================================
        
        # C1: x0 + s1 * x1 <= L1
        b1 = u1_limit - (x0 + u1_scale * x1)
        
        # C2: x3 + s4 * x4 <= L4
        b2 = u4_limit - (x3 + u4_scale * x4)
        
        # C3: x0 + s3 * x3 <= L3
        b3 = u3_limit - (x0 + u3_scale * x3)
        
        # C4: -x0 + s3b * x3 <= L3b
        # Barrier = Limit - (-x0 + s3b * x3) = Limit + x0 - s3b * x3
        b4 = u3_limit2 - (-x0 + u3_scale2 * x3) 
        
        # 拼接 Barrier 值 (N, 4)
        b_vals = torch.cat([b1, b2, b3, b4], dim=1)

        # ==========================================
        # 4. 构建梯度矩阵 G (Negative Gradients)
        # QP: G * u_reduced <= h_qp
        # u_reduced vector order: [u_0, u_1, u_3, u_4]
        # G row = -Gradient of b(x) w.r.t [x0, x1, x3, x4]
        # ==========================================
        
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
        
        # 组合 G 矩阵 (N, 4, 4) -> 4 Constraints, 4 Variables
        G = torch.cat([g1, g2, g3, g4], dim=1).to(x.device)

        # ==========================================
        # 5. 构建 QP 向量 h_qp (\alpha(b(x)))
        # ==========================================
        h_qp = gamma_coef * b_vals

        # ==========================================
        # 6. QP 目标函数
        # min || u_reduced - ref_reduced ||^2
        # ==========================================
        
        # 提取 4 个相关维度的建议步长
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

        # ==========================================
        # 7. 求解
        # ==========================================
        e = Variable(torch.Tensor()).to(x.device)
        
        try:
            # out: [u_0, u_1, u_3, u_4]
            out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(Q, q, G, h_qp, e, e)
        except Exception:
            # Fallback: 不做修正
            print(f"QP solver failed!")
            out = torch.zeros_like(ref_subset)

        # ==========================================
        # 8. 更新状态
        # ==========================================
        rt_flat = xp1_flat.clone()
        
        # 将计算出的修正量填回对应的维度
        rt_flat[:, 0:1] = x_flat[:, 0:1] + out[:, 0:1] # x0
        rt_flat[:, 1:2] = x_flat[:, 1:2] + out[:, 1:2] # x1
        rt_flat[:, 3:4] = x_flat[:, 3:4] + out[:, 2:3] # x3 (注意 out 索引 2)
        rt_flat[:, 4:5] = x_flat[:, 4:5] + out[:, 3:4] # x4 (注意 out 索引 3)
        
        rt = rt_flat.reshape(batch_size, horizon, dim)
        
        # 返回修正轨迹和最小 Barrier 值
        min_barrier, _ = torch.min(b_vals, dim=1)
        
        return rt, min_barrier

###################################################################cheetah     
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

####################################################################shield    
    @torch.no_grad()   #Walker2d
    def Shield(self, x0, xp10):  # Truncate method (Walker2d)

        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)

        nBatch = xp1.shape[0]

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.3  
        height = (height - self.mean[0]) / self.std[0]

        ############################################ceiling
        b = height - xp1[:,6:7] # - 0.1*x[:,15:16] 

        for k in range(nBatch):
            if b[k, 0] < 0: 
                xp1[k,6] = height

        b = height - xp1[:,6:7]

        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:,0])
    
    @torch.no_grad()   #Hopper
    def Shield_hopper(self, x0, xp10): # Truncate method (hopper)

        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)

        nBatch = xp1.shape[0]

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.5 
        height = (height - self.means[0]) / self.stds[0]

        ############################################ceiling
        b = height - xp1[:,3:4] 

        for k in range(nBatch):
            if b[k, 0] < 0: 
                xp1[k,3] = height

        b = height - xp1[:,3:4]

        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:,0])

    @torch.no_grad()   # Hopper
    def Shield_hopper_batch(self, x0, xp10): # Truncate method (hopper)
        """
        支持任意 Batch Size 的截断法 (Truncate) 实现。
        输入形状: x0, xp10 -> (Batch, Horizon, Dim)
        """
        # 1. 克隆输入，避免修改原始数据
        # x0 在此方法中未使用，保留它是为了与其他安全函数的接口签名保持一致
        xp1 = xp10.clone()

        # 2. 获取维度信息 (仅用于调试或断言，实际计算不需要显式使用)
        # batch_size, horizon, dim = xp1.shape

        # 3. 归一化障碍物高度
        # 这里沿用之前的归一化逻辑：物理高度 1.5m -> 归一化后的潜变量数值
        height_limit = self.height_limit
        height_val = (height_limit - self.means[0]) / self.stds[0]

        ############################################
        # 4. 执行向量化截断 (Vectorized Truncation)
        # 原始逻辑是: if z > height_limit, then z = height_limit
        # 这等价于对 z 维度取上界 (Upper Bound)
        
        # xp1[:, :, 3] 选中了所有 Batch 和所有 Horizon 的高度数据 (索引 3)
        # torch.clamp(..., max=height_val) 会并行地检查并修正所有大于 height_val 的值
        xp1[:, :, 3] = torch.clamp(xp1[:, :, 3], max=height_val)

        # 5. 计算安全余量 (用于返回监控)
        # b >= 0 表示安全。因为我们刚刚强制截断了，所以理论上返回的 b 应该是非负的 (>=0)
        b = height_val - xp1[:, :, 3]

        # 返回修正后的轨迹 xp1 和当前 Batch 中最小的安全余量
        return xp1, torch.min(b)

###################################################################GD     
    @torch.no_grad()   #walker2d
    def GD(self, x0, xp10):  #classifier guidance or potential-based method (walker2d)

        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.4  #1.3
        height = (height - self.mean[0]) / self.std[0]

        ############################################ceiling
        b = height - xp1[:,6:7]  - 0.1*x[:,15:16] 


        for k in range(nBatch):
            if b[k, 0] < 0:  # 0
                # u = -0.1
                u = -0.05
                xp1[k,6] = x[k,6] + u

                u2 = -0.05*10
                xp1[k,15] = x[k, 15] + u2

        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:,0])
    
    @torch.no_grad()   #Hopper
    def GD_hopper(self, x0, xp10):  #classifier guidance or potential-based method (hopper)

        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.6  # 1.5
        height = (height - self.mean[0]) / self.std[0]

        ############################################ceiling
        b = height - x[:,3:4] - 0.1*x[:,9:10]  


        for k in range(nBatch):
            if b[k, 0] < 0:  # 0
                # u = -0.1
                u = -0.05
                xp1[k,3] = x[k,3] + u

                u2 = -0.05*10
                xp1[k,9] = x[k, 9] + u2

        xp1 = xp1.unsqueeze(0)
        return xp1, torch.min(b[:,0])

    @torch.no_grad()   # Hopper
    def GD_hopper_cpx_batch(self, x0, xp10):  # Classifier guidance or potential-based method (hopper)
        """
        支持任意 Batch Size 的引导法 (Guidance) 实现。
        输入形状: x0, xp10 -> (Batch, Horizon, Dim)
        """
        x = x0.clone()
        xp1 = xp10.clone()

        # 1. 移除 squeeze，保持 (Batch, Horizon, Dim) 形状
        # x = x.squeeze(0) 
        # xp1 = xp1.squeeze(0)

        # 2. 归一化障碍物高度
        height_target = self.height_limit
        vel_scale = self.vel_scale * self.stds[6] / self.stds[0]
        height_norm = (height_target - self.means[0]) / self.stds[0] \
            - self.vel_scale * self.means[6] / self.stds[0]


        ############################################
        # 3. 向量化计算 Barrier Value (b)
        # 原始公式: b = height - z - 0.1 * v_z
        # x[:, :, 3] 是高度 (z), x[:, :, 9] 是垂直速度 (v_z)
        # 结果 b 的形状为 (Batch, Horizon)
        b = height_norm - x[:, :, 3] - vel_scale * x[:, :, 9]

        # 4. 生成违规掩码 (Mask)
        # mask 形状为 (Batch, Horizon)，True 表示该位置不安全
        mask = b < 0 

        # 5. 定义引导更新量 (Nudge)
        u = -0.05
        u2 = -0.5  # -0.05 * 10

        # 6. 向量化应用更新 (Vectorized Update)
        # 逻辑：
        # 如果 mask 为 True (不安全): 使用 x (当前步) + u (人工修正)
        # 如果 mask 为 False (安全): 保持 xp1 (扩散模型预测的下一步)
        
        # 更新高度 (Index 3)
        xp1[:, :, 3] = torch.where(mask, x[:, :, 3] + u, xp1[:, :, 3])
        
        # 更新垂直速度 (Index 9)
        xp1[:, :, 9] = torch.where(mask, x[:, :, 9] + u2, xp1[:, :, 9])

        # 7. 移除 unsqueeze
        # xp1 = xp1.unsqueeze(0)
        
        return xp1, torch.min(b)


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

