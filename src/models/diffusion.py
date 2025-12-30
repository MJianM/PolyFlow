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
        action_weight=1.0, loss_discount=1.0, loss_weights=None, env_name='hopper_cpx', safe_method='RoS'
    ):
        super().__init__()
        self.means = 0  # for normalization
        self.stds = 0
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        self.env_name = env_name
        self.safe_method = safe_method
        assert self.safe_method in ['RoS', 'RoS_cf', 'GD', 'Shield', 'none']
        assert self.env_name in ['hopper', 'hopper_cpx']
        print(f"Env name: {self.env_name}  Safe method: {self.safe_method}")

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


            if self.env_name == 'hopper_cpx':
                if self.safe_method == 'RoS_cf':
                    x, b_min = self.invariance_hopper_cpx_cf_batch(x_t, x)
                elif self.safe_method == 'RoS':
                    x, b_min = self.invariance_hopper_cpx_batch(x_t, x)
                elif self.safe_method == 'GD':
                    x, b_min = self.GD_hopper_cpx_batch(x_t, x)
            elif self.env_name == 'hopper':
                if self.safe_method == 'RoS_cf':
                    x, b_min = self.invariance_hopper_cf_batch(x_t, x)
                elif self.safe_method == 'RoS':
                    x, b_min = self.invariance_hopper_batch(x_t, x)
                elif self.safe_method == 'Shield':
                    x, b_min = self.Shield_hopper_batch(x_t, x)

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
    @torch.no_grad()   #only for sampling
    def invariance_hopper(self, x, xp1):   # RoS diffuser (hopper)

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.5
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] # - 0.1*x[:,9:10]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        #Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G = torch.cat([-Lgbu1], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,3:4]], dim = 1).to(G.device)  #, ref[:,15:16]
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(nBatch, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        # rt[:,15:16] = x[:,15:16] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_hopper_cf(self, x, xp1):  # RoS diffuser closed form (hopper)

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.5
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] # - 0.1*x[:,9:10]  # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        #Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G0 = torch.cat([-Lgbu1], dim = 1)
        k = 1
        h0 = Lfb + k*b

        Lgbu1 = 1*torch.ones_like(x[:,3:4])
        G1 = torch.cat([-Lgbu1], dim = 1)
        h1 = Lfb + k*(x[:,3:4] + 10)
        
   
        q = -torch.cat([ref[:,3:4]], dim = 1).to(G0.device)  #, ref[:,15:16]

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
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        # print(out[0:4,0:1])
        rt = rt.unsqueeze(0)

        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_hopper_cpx(self, x, xp1):  # RoS diffuser with complex safety specification (hopper)

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.6
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] - 0.1*x[:,9:10]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b
        
   
        q = -torch.cat([ref[:,3:4], ref[:,9:10]], dim = 1).to(G.device)  #
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        rt[:,9:10] = x[:,9:10] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)  # + 0.01  # for robustness
    
    @torch.no_grad()   #only for sampling
    def invariance_hopper_cpx_cf(self, x, xp1):   # RoS diffuser with complex safety specification, closed-form (hopper)

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle: Gaussian, x:0-6 control, 6-23 state
        height = 1.6
        height = (height - self.mean[0]) / self.std[0]

        #CBF
        ############################################ceiling
        b = height - x[:,3:4] - 0.1*x[:,9:10]   # - 0.01  # for robustness
        Lfb = 0 
        Lgbu1 = -1*torch.ones_like(x[:,3:4])
        Lgbu2 = -0.1*torch.ones_like(x[:,3:4])
  
        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h0 = Lfb + k*b

        Lgbu1 = 1*torch.ones_like(x[:,3:4])
        Lgbu2 = 0.1*torch.ones_like(x[:,3:4])
        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        h1 = Lfb + k*(x[:,3:4] + 0.1*x[:,9:10] + 10)    
   
        q = -torch.cat([ref[:,3:4], ref[:,9:10]], dim = 1).to(G0.device)  #

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
        rt[:,3:4] = x[:,3:4] + out[:,0:1]
        rt[:,9:10] = x[:,9:10] + out[:,1:2]
        rt = rt.unsqueeze(0)
        return rt, torch.min(b)   # + 0.01  # for robustness


    @torch.no_grad()   # 仅用于采样
    def invariance_hopper_batch(self, x, xp1):   # RoS diffuser (hopper)
        """
        支持任意 Batch Size 的版本。
        输入形状: x, xp1 -> (Batch, Horizon, Dim)
        """
        # 1. 获取维度信息
        batch_size, horizon, dim = x.shape
        
        # 2. 展平 (Flatten) 操作
        # 将 (Batch, Horizon, Dim) -> (Batch * Horizon, Dim)
        # 这样我们可以把每一个时间步都看作一个独立的优化问题，利用 QP 求解器的并行能力
        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)

        # 这里 nBatch 实际上是总的优化问题数量 (Total QP Problems = B * H)
        n_total = x_flat.shape[0]  
        
        # 计算参考控制量 (u_ref)
        ref = xp1_flat - x_flat

        # 归一化障碍物高度
        # TODO: 检查归一化索引是否对应
        height = 1.5
        height = (height - self.means[0]) / self.stds[0]

        # 3. 构建 CBF (基于展平后的数据)
        # b 的形状: (B*H, 1)
        b = height - x_flat[:, 3:4] 
        
        Lfb = 0 
        Lgbu1 = -1 * torch.ones_like(x_flat[:, 3:4])
  
        # CBF: \dot{b} + alpha(b) >= 0
        # Lfb + Lgbu1^T * u + k * b >= 0
        # -Lgbu1^T * u <= Lfb + k * b

        # G 的形状: (B*H, 1) -> (B*H, 1, 1) 适配 QP 求解器
        G = torch.cat([-Lgbu1], dim=1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k * b
        
        # 4. 构建 QP 矩阵
        # q 的形状: (B*H, 1)
        q = -torch.cat([ref[:, 3:4]], dim=1).to(G.device)
        
        # Q 的形状: (B*H, 1, 1)
        # 为每一个优化问题构建一个单位矩阵
        Q = Variable(torch.eye(1))
        Q = Q.unsqueeze(0).expand(n_total, 1, 1).to(G.device)
        
        e = Variable(torch.Tensor())
        
        # 求解 QP
        # out 的形状: (B*H, 1)
        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        # 5. 应用修正
        rt_flat = xp1_flat.clone()
        rt_flat[:, 3:4] = x_flat[:, 3:4] + out[:, 0:1]
        
        # 6. 还原形状 (Reshape back)
        # (B*H, Dim) -> (Batch, Horizon, Dim)
        rt = rt_flat.reshape(batch_size, horizon, dim)
        
        # 返回修正后的轨迹和全局最小安全值（用于监控）
        return rt, torch.min(b)

    @torch.no_grad()   # 仅用于采样
    def invariance_hopper_cf_batch(self, x, xp1):  # RoS diffuser closed form (hopper)
        """
        闭式解 (Closed-Form) 的任意 Batch Size 版本
        """
        batch_size, horizon, dim = x.shape
        
        # 展平: (Batch, Horizon, Dim) -> (Batch * Horizon, Dim)
        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)
        
        ref = xp1_flat - x_flat

        # 归一化
        height = 1.5
        height = (height - self.means[0]) / self.stds[0]

        # CBF 定义
        b = height - x_flat[:, 3:4]
        Lfb = 0 
        Lgbu1 = -1 * torch.ones_like(x_flat[:, 3:4])
  
        # 约束 0
        G0 = torch.cat([-Lgbu1], dim=1)
        k = 1
        h0 = Lfb + k * b

        # 约束 1
        Lgbu1_pos = 1 * torch.ones_like(x_flat[:, 3:4])
        G1 = torch.cat([-Lgbu1_pos], dim=1)
        h1 = Lfb + k * (x_flat[:, 3:4] + 10)
        
        q = -torch.cat([ref[:, 3:4]], dim=1).to(G0.device)

        # KKT 解析解求解
        # 所有变量形状第一维均为 (B*H)
        y1_bar = 1 * G0
        y2_bar = 1 * G1
        u_bar = -1 * q
        
        p1_bar = h0 - torch.sum(G0 * u_bar, dim=1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1 * u_bar, dim=1).unsqueeze(1)

        # 计算 Gram 矩阵元素
        G = torch.cat([
            torch.sum(y1_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0), 
            torch.sum(y1_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0), 
            torch.sum(y2_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0), 
            torch.sum(y2_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0)
        ], dim=0)
        
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # 计算 Lagrange 乘子 lambda
        # 这里的 torch.where 是逐元素操作，完美支持 (B*H, 1) 的形状
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), 
                  torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], 
                  torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], 
                  torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), 
                  torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1 * y1_bar + lambda2 * y2_bar + u_bar
        
        # 应用修正
        rt_flat = xp1_flat.clone()
        rt_flat[:, 3:4] = x_flat[:, 3:4] + out[:, 0:1]
        
        # 还原形状
        rt = rt_flat.reshape(batch_size, horizon, dim)

        return rt, torch.min(b)

    @torch.no_grad()   # 仅用于采样
    def invariance_hopper_cpx_batch(self, x, xp1):  # RoS diffuser with complex safety specification (hopper)
        """
        复杂约束 (位置+速度) 的任意 Batch Size 版本
        """
        batch_size, horizon, dim = x.shape
        
        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)
        n_total = x_flat.shape[0]

        ref = xp1_flat - x_flat

        height = 1.6
        height = (height - self.means[0]) / self.stds[0]
        vel_scale = 0.09 # 这是经过norm之后的

        # 复杂 CBF: b = height - pos - 0.1*vel
        b = height - x_flat[:, 3:4] - vel_scale * x_flat[:, 9:10]
        
        Lfb = 0 
        Lgbu1 = -1 * torch.ones_like(x_flat[:, 3:4])
        Lgbu2 = -vel_scale * torch.ones_like(x_flat[:, 3:4]) # 注意这里形状和 dim=0 的长度一致即可
  
        # G 形状: (B*H, 1, 2) -> 一个约束涉及两个变量
        G = torch.cat([-Lgbu1, -Lgbu2], dim=1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k * b
        
        # Q, q 针对两个变量 (位置, 速度)
        q = -torch.cat([ref[:, 3:4], ref[:, 9:10]], dim=1).to(G.device)
        
        # Q 形状: (B*H, 2, 2)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(n_total, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        # out 形状: (B*H, 2)
        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt_flat = xp1_flat.clone()
        # 更新位置和速度
        rt_flat[:, 3:4] = x_flat[:, 3:4] + out[:, 0:1]
        rt_flat[:, 9:10] = x_flat[:, 9:10] + out[:, 1:2]
        
        rt = rt_flat.reshape(batch_size, horizon, dim)
        return rt, torch.min(b)

    @torch.no_grad()   # 仅用于采样
    def invariance_hopper_cpx_cf_batch(self, x, xp1):   # RoS diffuser with complex safety specification, closed-form (hopper)
        """
        复杂约束 + 闭式解 的任意 Batch Size 版本
        """
        batch_size, horizon, dim = x.shape
        
        x_flat = x.reshape(-1, dim)
        xp1_flat = xp1.reshape(-1, dim)

        ref = xp1_flat - x_flat

        height = 1.6
        vel_scale = 0.09
        height = (height - self.means[0]) / self.stds[0]

        b = height - x_flat[:, 3:4] - vel_scale * x_flat[:, 9:10]
        Lfb = 0 
        Lgbu1 = -1 * torch.ones_like(x_flat[:, 3:4])
        Lgbu2 = -vel_scale * torch.ones_like(x_flat[:, 3:4]) # 修正：这里应该是 x_flat
  
        # 约束 0
        G0 = torch.cat([-Lgbu1, -Lgbu2], dim=1)
        k = 1
        h0 = Lfb + k * b

        # 约束 1
        Lgbu1_pos = 1 * torch.ones_like(x_flat[:, 3:4])
        Lgbu2_vel = 0.1 * torch.ones_like(x_flat[:, 3:4])
        G1 = torch.cat([-Lgbu1_pos, -Lgbu2_vel], dim=1)
        h1 = Lfb + k * (x_flat[:, 3:4] + 0.1 * x_flat[:, 9:10] + 10)    
   
        q = -torch.cat([ref[:, 3:4], ref[:, 9:10]], dim=1).to(G0.device)

        y1_bar = 1 * G0
        y2_bar = 1 * G1
        u_bar = -1 * q
        
        p1_bar = h0 - torch.sum(G0 * u_bar, dim=1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1 * u_bar, dim=1).unsqueeze(1)

        # 构造 Gram 矩阵
        G = torch.cat([
            torch.sum(y1_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0), 
            torch.sum(y1_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0), 
            torch.sum(y2_bar * y1_bar, dim=1).unsqueeze(1).unsqueeze(0), 
            torch.sum(y2_bar * y2_bar, dim=1).unsqueeze(1).unsqueeze(0)
        ], dim=0)
        
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # 闭式解逻辑 (逐元素)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), 
                  torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], 
                  torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], 
                  torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), 
                  torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1 * y1_bar + lambda2 * y2_bar + u_bar

        rt_flat = xp1_flat.clone()
        rt_flat[:, 3:4] = x_flat[:, 3:4] + out[:, 0:1]
        rt_flat[:, 9:10] = x_flat[:, 9:10] + out[:, 1:2]
        
        rt = rt_flat.reshape(batch_size, horizon, dim)
        return rt, torch.min(b)



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
        height_limit = 1.5 
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
        height_target = 1.6  
        vel_scale = 0.09
        height_norm = (height_target - self.means[0]) / self.stds[0]

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

