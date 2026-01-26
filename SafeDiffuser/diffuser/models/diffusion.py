import numpy as np
import torch
from torch import nn
import pdb
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
import einops

import SafeDiffuser.diffuser.utils as utils
from SafeDiffuser.diffuser.models.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GaussianDiffusion(nn.Module):
    def __init__(self, model_none, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model_none
        self.norm_mins = 0
        self.norm_maxs = 0

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

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

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
            input:
                x_t: [batch_size, horizon, transition_dim]
                t: [batch_size]
                noise: [batch_size, horizon, transition_dim]
            output:
                x_0: [batch_size, horizon, transition_dim]
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """
        input:
            x_start: [batch_size, horizon, transition_dim] 
            x_t: [batch_size, horizon, transition_dim]
            t: [batch_size]
        """

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        """
        input:
            x: [batch_size, horizon, transition_dim]
            cond: dict 
            t: [batch_size]
        output:
            model_mean, posterior_variance, posterior_log_variance
        """

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    


    @torch.no_grad()   #only for sampling
    def Shield(self, x0, xp10):  #Truncate method
        """
        input:
            x0: [1, horizon, transition_dim] 
            xp10: [1, horizon, transition_dim] 
        """

        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)

        nBatch = xp1.shape[0]

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        for k in range(nBatch):
            if b[k, 0] < 0: 
                theta = torch.atan2((xp1[k,2:3] - off_y)/yr, (xp1[k,3:4] - off_x)/xr)
                xp1[k,2] = yr*torch.sin(theta) + off_y
                xp1[k,3] = xr*torch.cos(theta) + off_x

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

         #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        self.safe1 = torch.min(b[:,0])
        self.safe2 = torch.min(b2[:,0])

        xp1 = xp1.unsqueeze(0)
        return xp1
    
    @torch.no_grad()   #only for sampling
    def GD(self, x0, xp10):    #classifier guidance or potential-based method
        """
        input:
            x0: [1, horizon, transition_dim]
            xp10: [1, horizon, transition_dim]
        """

        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        for k in range(nBatch):
            if b[k, 0] < 0.1:  # 0, 0.2
                u1 = 0.2/(2*((xp1[k,2:3] - off_y)/yr)/yr)
                u2 = 0.2/(2*((xp1[k,3:4] - off_x)/xr)/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001  #note no 0.1/0.01 for GD, but has for potential
                xp1[k,3] = xp1[k,3] + u2*0.001
            elif b2[k, 0] < 0.1:  # 0, 0.2
                u1 = 0.2/(4*((xp1[k,2:3] - off_y)/yr)**3/yr)
                u2 = 0.2/(4*((xp1[k,3:4] - off_x)/xr)**3/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001
                xp1[k,3] = xp1[k,3] + u2*0.001
            # else:
            #     x[k,2] = xp1[k,2]
            #     x[k,3] = xp1[k,3]

        self.safe1 = torch.min(b[:,0])
        self.safe2 = torch.min(b2[:,0])

        xp1 = xp1.unsqueeze(0)
        return xp1

    # @torch.no_grad()   #only for sampling
    # def invariance_umaze(self, x, xp1):   #  RoS-diffuser for umaze
    #     """
    #     input:
    #         x: [1, horizon, transition_dim]
    #         xp1: [1, horizon, transition_dim]
    #     """

    #     x = x.squeeze(0)
    #     xp1 = xp1.squeeze(0)

    #     nBatch = x.shape[0]
    #     ref = xp1 - x

    #     #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
    #     xr = 2*1.52/(self.norm_maxs[1] - self.norm_mins[1])
    #     yr = 2*1.52/(self.norm_maxs[0] - self.norm_mins[0])
    #     off_x = 2*(2.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
    #     off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

    #    #CBF
    #     b = 1 - ((x[:,2:3] - off_y)/yr)**4 - ((x[:,3:4] - off_x)/xr)**4
    #     Lfb = 0
    #     Lgbu1 = -4*((x[:,2:3] - off_y)/yr)**3/yr
    #     Lgbu2 = -4*((x[:,3:4] - off_x)/xr)**3/xr

    #     G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
    #     G = G.unsqueeze(1)
    #     k = 1
    #     h = Lfb + k*b

    #     self.safe1 = torch.min(b[:,0])

    #     #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
    #     xr = 2*1.2/(self.norm_maxs[1] - self.norm_mins[1])
    #     yr = 2*0.6/(self.norm_maxs[0] - self.norm_mins[0])
    #     off_x = 2*(2-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
    #     off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

    #     #CBF
    #     b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
    #     Lfb = 0
    #     Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
    #     Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

    #     self.safe2 = torch.min(b[:,0])

    #     G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
    #     G1 = G1.unsqueeze(1)
    #     k = 1
    #     h1 = Lfb + k*b

    #     G = torch.cat([G, G1], dim = 1)
    #     h = torch.cat([h, h1], dim = 1)
        
   
    #     q = -ref[:,2:4].to(G.device)
    #     Q = Variable(torch.eye(2))
    #     Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
    #     e = Variable(torch.Tensor())
    #     out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

    #     rt = xp1.clone()      
    #     rt[:,2:4] = x[:,2:4] + out
    #     rt = rt.unsqueeze(0)
    #     return rt
    
    # @torch.no_grad()   #only for sampling
    # def invariance_umaze_relax(self, x, xp1, t):  #  ReS-diffuser for umaze

    #     x = x.squeeze(0)
    #     xp1 = xp1.squeeze(0)

    #     nBatch = x.shape[0]
    #     ref = xp1 - x

    #     #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
    #     xr = 2*1.52/(self.norm_maxs[1] - self.norm_mins[1])
    #     yr = 2*1.52/(self.norm_maxs[0] - self.norm_mins[0])
    #     off_x = 2*(2.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
    #     off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

    #     #CBF
    #     b = 1 - ((x[:,2:3] - off_y)/yr)**4 - ((x[:,3:4] - off_x)/xr)**4
    #     Lfb = 0
    #     Lgbu1 = -4*((x[:,2:3] - off_y)/yr)**3/yr
    #     Lgbu2 = -4*((x[:,3:4] - off_x)/xr)**3/xr

    #     self.safe1 = torch.min(b[:,0])

    #     if t >= 10:
    #         sign = 100   #relax
    #     else:
    #         sign = 0   #non-relax

    #     rx0 = torch.zeros_like(Lgbu1).to(b.device)
    #     rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

    #     G = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
    #     G = G.unsqueeze(1)
    #     k = 1
    #     h = Lfb + k*b

    #     #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
    #     xr = 2*1.2/(self.norm_maxs[1] - self.norm_mins[1])
    #     yr = 2*0.6/(self.norm_maxs[0] - self.norm_mins[0])
    #     off_x = 2*(2-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
    #     off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

    #     #CBF
    #     b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
    #     Lfb = 0
    #     Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
    #     Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

    #     self.safe2 = torch.min(b[:,0])

    #     G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
    #     G1 = G1.unsqueeze(1)
    #     k = 1
    #     h1 = Lfb + k*b

    #     G = torch.cat([G, G1], dim = 1)
    #     h = torch.cat([h, h1], dim = 1)
        
   
    #     q = -ref[:,2:4].to(G.device)
    #     q0 = torch.zeros_like(q).to(G.device)
    #     q = torch.cat([q, q0], dim = 1)
    #     Q = Variable(torch.eye(4))
    #     Q = Q.unsqueeze(0).expand(nBatch, 4, 4).to(G.device)
        
    #     e = Variable(torch.Tensor())
    #     out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

    #     rt = xp1.clone()      
    #     rt[:,2:4] = x[:,2:4] + out[:,0:2]
    #     rt = rt.unsqueeze(0)
    #     return rt

    @torch.no_grad()   #only for sampling
    def invariance(self, x, xp1):    #  RoS-diffuser for maze2d-large-v1
        """
        input:
            x: [1, horizon, transition_dim]
            xp1: [1, horizon, transition_dim]
        """

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0] 
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01  # robust term 09/25
        Lfb = 0

        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1 
        h = Lfb + k*b

        self.safe1 = torch.min(b[:,0] + 0.01)  # robust term 09/25

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 # robust term 09/25
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0]+ 0.01) # robust term 09/25

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
        q = -ref[:,2:4].to(G.device)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
        # print(out)
        rt = rt.unsqueeze(0)
        return rt

    @torch.no_grad()   #only for sampling
    def invariance_cf(self, x, xp1):  # closed form solution,  RoS-diffuser for maze2d-large-v1
        """
        input:
            x: [1, horizon, transition_dim]
            xp1: [1, horizon, transition_dim]
        """

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b0 = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01  # robust term 09/25
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h0 = Lfb + k*b0

        self.safe1 = torch.min(b0[:,0] + 0.01)  # robust term 09/25

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 # robust term 09/25
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0]+ 0.01) # robust term 09/25

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h1 = Lfb + k*b
        
        q = -ref[:,2:4].to(b.device)
        
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
        rt[:,2:4] = x[:,2:4] + out
        # print(out)
        rt = rt.unsqueeze(0)
        return rt
        



    @torch.no_grad()   #only for sampling
    def invariance_relax(self, x, xp1, t):  #  ReS-diffuser for maze2d-large-v1
        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0] 
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        if t >= 10:   # debug  10
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        G = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0], dim = 1)
        Q = Variable(torch.eye(4))
        Q = Q.unsqueeze(0).expand(nBatch, 4, 4).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt
    
    @torch.no_grad()   #only for sampling
    def invariance_relax_cf(self, x, xp1, t):  # closed-form solution, ReS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        if t >= 10:   # debug  10
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        G0 = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        k = 1
        h0 = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
        k = 1
        h1 = Lfb + k*b
        
   
        q = -ref[:,2:4].to(G0.device)
        q0 = torch.zeros_like(q).to(G0.device)
        q = torch.cat([q, q0], dim = 1)

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
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        # print(out)
        rt = rt.unsqueeze(0)
        return rt
    

    @torch.no_grad()   #only for sampling
    def invariance_relax_narrow(self, x, xp1, t):  #  ReS-diffuser for maze2d-large-v1,  narrow passage case

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        if t >= 10:   # debug  10
            sign = 1   #relax
        else:
            sign = 0   #non-relax

        #normalize obstacle 1,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])

        off_x = 2*(5.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.4  # 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        self.safe1 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0, rx0, rx0, rx0, rx0], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        ########################################### obs 2
        off_x = 2*(5.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.6 #0.01
        Lfb = 0
        Lgbu12 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu22 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b2[:,0] + 0.01)

        G2 = torch.cat([-Lgbu12, -Lgbu22, rx0, rx1, rx0, rx0, rx0, rx0], dim = 1)
        G2 = G2.unsqueeze(1)
        k = 1
        h2 = Lfb + k*b2

        ########################################### obs 3
        off_x = 2*(3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b3 = ((x[:,2:3] - off_y)/yr/0.5)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu13 = 4*((x[:,2:3] - off_y)/yr/0.5)**3/yr
        Lgbu23 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G3 = torch.cat([-Lgbu13, -Lgbu23, rx0, rx0, rx1, rx0, rx0, rx0], dim = 1)
        G3 = G3.unsqueeze(1)
        k = 1
        h3 = Lfb + k*b3

        ########################################### obs 4
        off_x = 2*(8.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(3.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b4 = ((x[:,2:3] - off_y)/yr/1.8)**4 + ((x[:,3:4] - off_x)/xr/1.8)**4 - 1 - 0.01
        Lfb = 0
        Lgbu14 = 4*((x[:,2:3] - off_y)/yr/1.8)**3/yr
        Lgbu24 = 4*((x[:,3:4] - off_x)/xr/1.8)**3/xr

        G4 = torch.cat([-Lgbu14, -Lgbu24, rx0, rx0, rx0, rx1, rx0, rx0], dim = 1)
        G4 = G4.unsqueeze(1)
        k = 1
        h4 = Lfb + k*b4

        ########################################### obs 5
        off_x = 2*(7.6-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(7-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b5 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.4 #0.01
        Lfb = 0
        Lgbu15 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu25 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G5 = torch.cat([-Lgbu15, -Lgbu25, rx0, rx0, rx0, rx0, rx1, rx0], dim = 1)
        G5 = G5.unsqueeze(1)
        k = 1
        h5 = Lfb + k*b5

        ########################################### obs 6
        off_x = 2*(10-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(6.3-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b6 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu16 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu26 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G6 = torch.cat([-Lgbu16, -Lgbu26, rx0, rx0, rx0, rx0, rx0, rx1], dim = 1)
        G6 = G6.unsqueeze(1)
        k = 1
        h6 = Lfb + k*b6

        b0 = torch.cat([b, b2, b3, b4, b5, b6], dim = 1)
        idx = torch.argmin(b0, dim = 1).cpu().numpy()
        G0 = torch.cat([G1, G2, G3, G4, G5, G6], dim = 1)
        h0 = torch.cat([h1, h2, h3, h4, h5, h6], dim = 1)
        rows = len(G0[:,0,0])
        G = []
        h = []
        for i in range(rows):
            G.append(G0[i:i+1,idx[i]:idx[i]+1])
            h.append(h0[i:i+1,idx[i]:idx[i]+1])
        G = torch.cat(G, dim = 0)
        h = torch.cat(h, dim = 0)


        

        # G = torch.cat([G1, G2, G3, G4, G5, G6], dim = 1)
        # h = torch.cat([h1, h2, h3, h4, h5, h6], dim = 1)

        # G = torch.cat([G1, G2, G3, G5], dim = 1)
        # h = torch.cat([h1, h2, h3, h5], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0, q0, q0], dim = 1)
        Q = Variable(torch.eye(8))
        Q = Q.unsqueeze(0).expand(nBatch, 8, 8).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt


    @torch.no_grad()   #only for sampling
    def invariance_time(self, x, xp1, t):  #  TVS-diffuser for maze2d-large-v1
        t_bias = 5  #50

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - nn.Sigmoid()(t_bias - t) -0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1  #0.3
        h = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - nn.Sigmoid()(t_bias - t) - 0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1  #0.4
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
        rt = rt.unsqueeze(0)
        return rt

    @torch.no_grad()   #only for sampling
    def invariance_time_cf(self, x, xp1, t):  # closed-form solution, TVS-diffuser for maze2d-large-v1
        t_bias = 5  #50 

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - nn.Sigmoid()(t_bias - t) -0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1  #0.3
        h0 = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - nn.Sigmoid()(t_bias - t) - 0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1  #0.4
        h1 = Lfb + k*b
        
   
        q = -ref[:,2:4].to(G0.device)

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
        rt[:,2:4] = x[:,2:4] + out
        # print(out)
        rt = rt.unsqueeze(0)
        return rt        

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        """
        input:
            x: [batch_size, horizon, transition_dim]
            cond: dict
            t: [batch_size]
        output:
            x: [batch_size, horizon, transition_dim] 
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Note:  choose any one of the below
        #---------------------------------------start--------------------------------------------------#
        ####################### original diffuser only
        # x = xp1      
        # xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        # yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        # off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        # b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1
        # self.safe1 = torch.min(b[:,0])
        # xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        # yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        # off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        # b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        # self.safe2 = torch.min(b[:,0])

        ####################### truncate (shield) and GD (classifier-guidance/potential-based)
        # x = self.Shield(x, xp1)
        # x = self.GD(x, xp1)

        ####################### SafeDiffusers 
        x = xp1 # for training only
        # x = self.invariance(x, xp1)    # RoS
        # x = self.invariance_cf(x, xp1)  # RoS closed form
        # x = self.invariance_relax(x, xp1, t) # ReS
        # x = self.invariance_relax_cf(x, xp1, t)   #ReS closed form    
        # x = self.invariance_time(x, xp1, t)   # TVS
        # x = self.invariance_time_cf(x, xp1, t)  # TVS closed form
        # x = self.invariance_relax_narrow(x, xp1, t)  # narrow passage case

        ####################### Applying SafeDiffusers to only the last 10 steps
        # if t <= 10:  #10
        #     # x = self.invariance_relax(x, xp1, t)  #done
        #     # x = self.invariance_relax_narrow(x, xp1, t)

        #     x = self.GD(x, xp1)
        # else:
        #     x = xp1
        #     xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        #     yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        #     off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        #     off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        #     b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1
        #     self.safe1 = torch.min(b[:,0])
        #     xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        #     yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        #     off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        #     off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        #     b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        #     self.safe2 = torch.min(b[:,0])

        
        ###################### umaze case
        # x = self.invariance_umaze(x, xp1)   #umaze
        # x = self.invariance_umaze_relax(x, xp1, t)   #umaze
        #-----------------------------------------end--------------------------------------------------#
        return x

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        """
        input:
            shape: (batch_size, horizon, transition_dim)
            cond: dict
        output:
            x: [batch_size, horizon, transition_dim]
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        safe1, safe2 = [], []
        for i in reversed(range(0, self.n_timesteps)):  #-50 change here for the number of diffusion steps,
            if i < 0:
                i = 0
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)
            safe1.append(self.safe1.unsqueeze(0))
            safe2.append(self.safe2.unsqueeze(0))
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
        
        self.safe1 = torch.cat(safe1, dim=0)
        self.safe2 = torch.cat(safe2, dim=0)

        progress.close()
        # pdb.set_trace()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, n_samples, horizon=None, return_diffusion = True, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = n_samples
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, return_diffusion= return_diffusion, **kwargs)   ## debug

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        """
        forward diffusion process: (x_t | x_0)。
        input:
            x_start: [batch_size, horizon, transition_dim]
            t: [batch_size]
            noise: [batch_size, horizon, transition_dim]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        """
        input:
            x_start: [batch_size, horizon, transition_dim]
            cond: dict
            t: [batch_size]
        """
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

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)


    def _vb_terms_bpd(
        self, x_start, conditions, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # batch_size = x_start.shape(0)
        # conditions = self._format_conditions(conditions, batch_size)

        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        
        mean, _, log_variance = self.p_mean_variance(
             x_t, conditions, t)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, mean, log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)

        # import pdb; pdb.set_trace()
        # decoder_nll = -discretized_gaussian_log_likelihood(
        #     x_start, means=mean, log_scales=0.5 * log_variance
        # )
        
        # assert decoder_nll.shape == x_start.shape
        # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        # output = torch.where((t == 0), decoder_nll, kl)

        return kl


    def forward(self, cond, n_samples, **kwargs):
        return self.conditional_sample(cond=cond, n_samples=n_samples, **kwargs)



class SafeGaussianDiffusion(GaussianDiffusion):
    def __init__(self, 
                 model_bone, 
                 horizon, 
                 observation_dim, 
                 action_dim, 
                 n_timesteps=1000, 
                 loss_type='l1', 
                 clip_denoised=False, 
                 predict_epsilon=True, 
                 action_weight=1, loss_discount=1, loss_weights=None,
                 ellips_list=None,
                 safe_method='truncate', # truncate, classifier_guidance, invariance, invariance_relax
                 sample_end_timestep=0,
                 sample_cbf_timestep=None,
                 ):
        super().__init__(model_bone, 
                         horizon, 
                         observation_dim, 
                         action_dim, 
                         n_timesteps, 
                         loss_type, 
                         clip_denoised, 
                         predict_epsilon, 
                         action_weight, loss_discount, loss_weights)
        
        assert safe_method in ['diffuser', 'diffusertrunc', 'diffuserguide', 'RoS', 'RoS_cf', 'ReS', 'TVS']

        assert sample_end_timestep <= 0
        assert sample_cbf_timestep is None or sample_cbf_timestep > 0

        self.ellips_list = ellips_list
        self.safe_method = safe_method

        self.cbfs = [0 for _ in range(len(ellips_list))] 

        self.sample_end_timestep = sample_end_timestep
        self.sample_cbf_timestep = sample_cbf_timestep
        if self.sample_cbf_timestep is None:
            self.sample_cbf_timestep = self.n_timesteps + 1

    def get_normalize_obstacles(self, ellips_list):
        """
        ellips_list: [(xc, yc, a, b, n), ...]
        xc -> norm_mins[0], yc -> norm_mins[1]
        """
        obstacles = []
        for (xc, yc, a, b, n) in ellips_list:
            xc_norm = 2 * (xc - self.norm_mins[0]) / (self.norm_maxs[0] - self.norm_mins[0]) - 1
            yc_norm = 2 * (yc - self.norm_mins[1]) / (self.norm_maxs[1] - self.norm_mins[1]) - 1
            
            a_norm = 2 * a / (self.norm_maxs[0] - self.norm_mins[0])
            b_norm = 2 * b / (self.norm_maxs[1] - self.norm_mins[1])
            
            obstacles.append({
                'xc': xc_norm, 'yc': yc_norm,
                'a': a_norm, 'b': b_norm,
                'n': n
            })
        return obstacles
    

    @torch.no_grad()
    def Shield(self, x0, xp10):
        """
        input:
            x0: [batch, horizon, transition_dim] 
            xp10: [batch, horizon, transition_dim] 
        """
        xp1 = xp10.clone()
        xp1 = xp1.reshape(-1, xp1.shape[-1])  # [batch*horizon, transition_dim]
        idx_x = self.action_dim + 0
        idx_y = self.action_dim + 1
        
        obstacles = self.get_normalize_obstacles(self.ellips_list)

        for i, obs in enumerate(obstacles):
            dx = (xp1[:, idx_x] - obs['xc']) / obs['a']
            dy = (xp1[:, idx_y] - obs['yc']) / obs['b']
            
            dist_val = torch.abs(dx)**obs['n'] + torch.abs(dy)**obs['n']
            inside = dist_val < 1.0
            
            self.cbfs[i] = torch.min(dist_val - 1.0)

            if inside.any():
                scale = (1.0 / dist_val[inside])**(1.0 / obs['n'])
                xp1[inside, idx_x] = obs['xc'] + dx[inside] * obs['a'] * scale
                xp1[inside, idx_y] = obs['yc'] + dy[inside] * obs['b'] * scale
        
        xp1 = xp1.reshape(xp10.shape)
        return xp1      

    @torch.no_grad()
    def GD(self, x0, xp10):
        """
        Classifier Guidance / Potential-based
        input:
            x0: [1, horizon, transition_dim]
            xp10: [1, horizon, transition_dim]
        """
        xp1 = xp10.clone()
        xp1 = xp1.reshape(-1, xp1.shape[-1])  # [batch*horizon, transition_dim]
        idx_x = self.action_dim + 0
        idx_y = self.action_dim + 1
        
        obstacles = self.get_normalize_obstacles(self.ellips_list)

        for i, obs in enumerate(obstacles):
            dx_n = (xp1[:, idx_x] - obs['xc']) / obs['a']
            dy_n = (xp1[:, idx_y] - obs['yc']) / obs['b']
            b = torch.abs(dx_n)**obs['n'] + torch.abs(dy_n)**obs['n'] - 1

            self.cbfs[i] = torch.min(b)
            
            mask = b < 0.1 
            if mask.any():
                grad_x = obs['n'] * torch.abs(dx_n[mask])**(obs['n']-1) * torch.sign(dx_n[mask]) / obs['a']
                grad_y = obs['n'] * torch.abs(dy_n[mask])**(obs['n']-1) * torch.sign(dy_n[mask]) / obs['b']
                
                xp1[mask, idx_x] += grad_x * 0.001
                xp1[mask, idx_y] += grad_y * 0.001
                
        xp1 = xp1.reshape(xp10.shape)
        return xp1     
    
    @torch.no_grad()
    def invariance(self, x, xp1, k=1.0):
        """
        input:
            x: [1, horizon, transition_dim]
            xp1: [1, horizon, transition_dim]
            k: CBF gain
        """
        x_curr = x.reshape(-1, x.shape[-1])  # [batch*horizon, transition_dim]
        x_pred = xp1.reshape(-1, xp1.shape[-1])  # [batch*horizon, transition_dim]
        n_steps = x_curr.shape[0]
        
        idx_x = self.action_dim + 0
        idx_y = self.action_dim + 1
        
        u_ref_x = x_pred[:, idx_x] - x_curr[:, idx_x]
        u_ref_y = x_pred[:, idx_y] - x_curr[:, idx_y]
        
        all_G = []
        all_h = []
        
        obstacles = self.get_normalize_obstacles(self.ellips_list)

        for i, obs in enumerate(obstacles):
            dx_n = (x_curr[:, idx_x] - obs['xc']) / obs['a']
            dy_n = (x_curr[:, idx_y] - obs['yc']) / obs['b']
            
            b = torch.abs(dx_n)**obs['n'] + torch.abs(dy_n)**obs['n'] - 1 - 0.01
            
            self.cbfs[i] = torch.min(b)

            # -grad_x * ux - grad_y * uy <= k * b
            grad_x = obs['n'] * torch.abs(dx_n)**(obs['n']-1) * torch.sign(dx_n) / obs['a']
            grad_y = obs['n'] * torch.abs(dy_n)**(obs['n']-1) * torch.sign(dy_n) / obs['b']
            
            G_i = torch.stack([-grad_x, -grad_y], dim=1).unsqueeze(1) # [H, 1, 2]
            h_i = (k * b).unsqueeze(1) # [H, 1]
            
            all_G.append(G_i)
            all_h.append(h_i)
            
        G = torch.cat(all_G, dim=1) # [H, num_obs, 2]
        h = torch.cat(all_h, dim=1) # [H, num_obs]
        
        # objective function: 1/2 ||u - u_ref||^2 = 1/2 (u^T I u - 2 u^T u_ref + const)
        # 1/2 u^T Q u + q^T u
        Q = torch.eye(2).unsqueeze(0).expand(n_steps, 2, 2).to(x.device) + \
            torch.eye(2, device=x.device).unsqueeze(0) * 1e-6
        q = -torch.stack([u_ref_x, u_ref_y], dim=1).to(x.device)
        
        e = Variable(torch.Tensor())

        dtype_ori = x.dtype
        try:
            
            Q = Q.double()
            q = q.double()
            G = G.double()
            h = h.double()
            e = e.double()
            u_optimal = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(Q, q, G, h, e, e)
        
            if torch.isnan(u_optimal).any():
                raise ValueError("QP Solver returned NaN")
        except:
            print(f"QP failed, falling back to Shielding. Error: {e}")
            return self.Shield(x, xp1)

        rt = x_pred.clone()
        rt[:, idx_x] = x_curr[:, idx_x] + u_optimal[:, 0].to(dtype=dtype_ori)
        rt[:, idx_y] = x_curr[:, idx_y] + u_optimal[:, 1].to(dtype=dtype_ori)

        rt = rt.reshape(xp1.shape)
        return rt


    @torch.no_grad()
    def invariance_cf_multi(self, x, xp1):
        x_curr = x.reshape(-1, x.shape[-1])  # [H, transition_dim]
        x_pred = xp1.reshape(-1, xp1.shape[-1])  # [H, transition_dim]
        n_steps = x_curr.shape[0]
        
        idx_x = self.action_dim + 0
        idx_y = self.action_dim + 1
        
        u_bar_x = x_pred[:, idx_x] - x_curr[:, idx_x]
        u_bar_y = x_pred[:, idx_y] - x_curr[:, idx_y]
        
        all_b = []
        all_G = []
        all_h = []

        obstacles = self.get_normalize_obstacles(self.ellips_list)

        for i, obs in enumerate(obstacles):
            dx_n = (x_curr[:, idx_x] - obs['xc']) / obs['a']
            dy_n = (x_curr[:, idx_y] - obs['yc']) / obs['b']
            
            b = torch.abs(dx_n)**obs['n'] + torch.abs(dy_n)**obs['n'] - 1 - 0.01
            all_b.append(b) # [H]

            self.cbfs[i] = torch.min(b)
            
            Lgbu_x = obs['n'] * torch.abs(dx_n)**(obs['n']-1) * torch.sign(dx_n) / obs['a']
            Lgbu_y = obs['n'] * torch.abs(dy_n)**(obs['n']-1) * torch.sign(dy_n) / obs['b']
            
            # G * u <= h 
            # G = [-Lgbu_x, -Lgbu_y], h = k * b (k=1)
            G_i = torch.stack([-Lgbu_x, -Lgbu_y], dim=1) # [H, 2]
            h_i = 1.0 * b # [H]
            
            all_G.append(G_i)
            all_h.append(h_i)

        all_b_tensor = torch.stack(all_b, dim=1) # [H, Num_Obs]
        min_b_val, min_idx = torch.min(all_b_tensor, dim=1) # [H]
        
        G_final = torch.stack(all_G, dim=1) # [H, Num_Obs, 2]
        h_final = torch.stack(all_h, dim=1) # [H, Num_Obs]
        
        G_star = G_final[torch.arange(n_steps), min_idx] # [H, 2]
        h_star = h_final[torch.arange(n_steps), min_idx] # [H]
        
        # goal: min ||out - u_bar||^2 s.t. G_star * out <= h_star
        u_bar = torch.stack([u_bar_x, u_bar_y], dim=1) # [H, 2]
        p_bar = h_star - torch.sum(G_star * u_bar, dim=1) # [H]
        
        # lambda = clamp(p_bar / ||G||^2, max=0)
        norm_sq = torch.sum(G_star * G_star, dim=1) + 1e-8
        lambda_val = torch.clamp(p_bar, max=0) / norm_sq # [H]
        
        out = u_bar + lambda_val.unsqueeze(1) * G_star # [H, 2]

        rt = x_pred.clone()
        rt[:, idx_x] = x_curr[:, idx_x] + out[:, 0]
        rt[:, idx_y] = x_curr[:, idx_y] + out[:, 1]
        
        rt = rt.reshape(xp1.shape)
        return rt
    
    @torch.no_grad()
    def invariance_relax(self, x, xp1, t):
        """
        :params x: [batch, horizon, transition_dim]
        :params xp1: [batch, horizon, transition_dim]
        :params t: [batch,]
        """
        x_curr = x.reshape(-1, x.shape[-1])  # [H, transition_dim]
        x_pred = xp1.reshape(-1, xp1.shape[-1])  # [H, transition_dim]
        t = t.unsqueeze(1).repeat(1, x.shape[1]).reshape(-1)  # [H,]
        n_steps = x_curr.shape[0]  # Horizon
        n_obs = len(self.ellips_list) 
        
        idx_x = self.action_dim + 0
        idx_y = self.action_dim + 1
        
        ref = x_pred - x_curr
        
        sign = torch.where(t >= 0, torch.full_like(t, 1.0), torch.full_like(t, 0.0)) 

        n_vars = 2 + n_obs
        all_G = []
        all_h = []

        obstacles = self.get_normalize_obstacles(self.ellips_list)

        for i, obs in enumerate(obstacles):
            dx_n = (x_curr[:, idx_x] - obs['xc']) / obs['a']
            dy_n = (x_curr[:, idx_y] - obs['yc']) / obs['b']
            
            b = torch.abs(dx_n)**obs['n'] + torch.abs(dy_n)**obs['n'] - 1 - 0.01

            Lgbu_x = obs['n'] * torch.abs(dx_n)**(obs['n']-1) * torch.sign(dx_n) / obs['a']
            Lgbu_y = obs['n'] * torch.abs(dy_n)**(obs['n']-1) * torch.sign(dy_n) / obs['b']
            grad = torch.concatenate([Lgbu_x.unsqueeze(1), Lgbu_y.unsqueeze(1)], dim=1)
            grad_norm = torch.norm(grad, dim=1, keepdim=True)
            # grad = grad / (grad_norm + 1e-6) * torch.clamp(grad_norm, max=1.0)
            Lgbu_x = grad[:, 0]
            Lgbu_y = grad[:, 1]

            k = 1.0

            self.cbfs[i] = torch.min(b)

            G_row = torch.zeros(n_steps, n_vars).to(x.device)
            G_row[:, 0] = -Lgbu_x
            # G_row[:, 0] = -grad[:, 0]
            G_row[:, 1] = -Lgbu_y
            # G_row[:, 1] = -grad[:, 1]
            G_row[:, 2 + i] = sign 
            
            all_G.append(G_row.unsqueeze(1))
            all_h.append(k * b.unsqueeze(1))

        # [H, n_obs, 2 + n_obs]
        G = torch.cat(all_G, dim=1)
        h = torch.cat(all_h, dim=1) # [H, n_obs]

        # goal：min 1/2 * (ux^2 + uy^2 + xi_1^2 + ... + xi_n^2) + q^T * z
        Q = torch.eye(n_vars).unsqueeze(0).expand(n_steps, n_vars, n_vars).to(x.device) + \
            1e-4 * torch.eye(n_vars, device=x.device).unsqueeze(0)
        
        # [-u_ref_x, -u_ref_y, 0, ..., 0]
        q_vec = torch.zeros(n_steps, n_vars).to(x.device)
        q_vec[:, 0] = -ref[:, idx_x]
        q_vec[:, 1] = -ref[:, idx_y]
        
        dtype_orig = x.dtype
        Q = Q.double()
        q_vec = q_vec.double()
        G = G.double()
        h = h.double()
        e = Variable(torch.Tensor().double().to(x.device))
        out = QPFunction(verbose=True, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(Q, q_vec, G, h, e, e)

        rt = x_pred.clone()      
        rt[:, idx_x] = x_curr[:, idx_x] + out[:, 0].to(dtype=dtype_orig)
        rt[:, idx_y] = x_curr[:, idx_y] + out[:, 1].to(dtype=dtype_orig)

        rt = rt.reshape(xp1.shape)
        return rt
    
    @torch.no_grad()
    def invariance_time(self, x, xp1, t):
        """
        :params x: [batch, horizon, transition_dim]
        :params xp1: [batch, horizon, transition_dim]
        :params t: [batch,]
        """
        t_bias = 5
        B, H, D = x.shape  
        
        x_curr = x.view(B * H, D)
        x_pred = xp1.view(B * H, D)
        n_total = B * H 
        
        idx_x = self.action_dim + 0
        idx_y = self.action_dim + 1
        
        t_expand = t.view(B, 1).expand(B, H).reshape(-1)

        T_temp = 10.0 
        t_bias = 5.0
        
        sig_input = (t_expand - t_bias) / T_temp 
        sig_val = torch.sigmoid(sig_input)
        
        gamma_scale = 5.0
        gamma = - gamma_scale * sig_val
        Lfb = (gamma_scale / T_temp) * sig_val * (1 - sig_val)
        
        all_G = []
        all_h = []
        
        obstacles = self.get_normalize_obstacles(self.ellips_list)

        for i, obs in enumerate(obstacles):
            dx_n = (x_curr[:, idx_x] - obs['xc']) / obs['a']
            dy_n = (x_curr[:, idx_y] - obs['yc']) / obs['b']
            
            eps = 1e-6
            dx_n_safe = torch.where(torch.abs(dx_n) < eps, torch.sign(dx_n + 1e-9)*eps, dx_n)
            dy_n_safe = torch.where(torch.abs(dy_n) < eps, torch.sign(dy_n + 1e-9)*eps, dy_n)

            # b(x, t) = h_geom(x) - sigma(t_bias - t) - 0.01
            h_geom = torch.abs(dx_n)**obs['n'] + torch.abs(dy_n)**obs['n'] - 1
            b = h_geom - gamma - 0.01

            self.cbfs[i] = torch.min(b)
            
            Lgbu_x = obs['n'] * torch.abs(dx_n)**(obs['n']-1) * torch.sign(dx_n_safe) / obs['a']
            Lgbu_y = obs['n'] * torch.abs(dy_n)**(obs['n']-1) * torch.sign(dy_n_safe) / obs['b']
            
            # G_i [n_total, 1, 2] 和 h_i [n_total, 1]
            G_i = torch.stack([-Lgbu_x, -Lgbu_y], dim=1).unsqueeze(1)
            k = 1.0
            h_i = ( - Lfb + k * b).unsqueeze(1)
            
            all_G.append(G_i)
            all_h.append(h_i)
            
        G = torch.cat(all_G, dim=1) # [B*H, num_obs, 2]
        h = torch.cat(all_h, dim=1) # [B*H, num_obs]
        
        ref_x = x_pred[:, idx_x] - x_curr[:, idx_x]
        ref_y = x_pred[:, idx_y] - x_curr[:, idx_y]
        q = -torch.stack([ref_x, ref_y], dim=1).to(G.device)
        Q = torch.eye(2).unsqueeze(0).expand(n_total, 2, 2).to(G.device) + \
            torch.eye(2, device=G.device).unsqueeze(0) * 1e-4
        
        e = Variable(torch.Tensor())
        dtype_ori = x.dtype
        Q = Q.double()
        q = q.double()
        G = G.double()
        h = h.double()
        e = e.double()
        out = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED, eps=1e-3)(Q, q, G, h, e, e)
        
        rt = xp1.clone().view(B * H, D)
        rt[:, idx_x] = x_curr[:, idx_x] + out[:, 0].to(dtype=dtype_ori)
        rt[:, idx_y] = x_curr[:, idx_y] + out[:, 1].to(dtype=dtype_ori)
        
        return rt.view(B, H, D)
    

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        """
        input:
            x: [batch_size, horizon, transition_dim]
            cond: dict
            t: [batch_size]
        output:
            x: [batch_size, horizon, transition_dim] (经过安全修正后的 x_{t-1})
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        if self.safe_method == 'diffuser':
            x = xp1
        elif self.safe_method == 'diffusertrunc':
            x = self.Shield(x, xp1)
        elif self.safe_method == 'diffuserguide':
            x = self.GD(x, xp1)
        elif self.safe_method == 'RoS':
            if t[0] <= self.sample_cbf_timestep:
                x = self.invariance(x, xp1)
            else:
                x = xp1
        elif self.safe_method == 'ReS':
            if t[0] <= self.sample_cbf_timestep:
                x = self.invariance_relax(x, xp1, t)
            else:
                x = xp1
        elif self.safe_method == 'TVS':
            if t[0] <= self.sample_cbf_timestep:
                x = self.invariance_time(x, xp1, t)
            else:
                x = xp1
        else:
            raise ValueError(f"Unknown safe_method: {self.safe_method}")

        return x

        # Note:  choose any one of the below
        #---------------------------------------start--------------------------------------------------#
        ####################### original diffuser only
        # x = xp1      
        # xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        # yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        # off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        # b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1
        # self.safe1 = torch.min(b[:,0])
        # xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        # yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        # off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        # off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        # b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        # self.safe2 = torch.min(b[:,0])

        ####################### truncate (shield) and GD (classifier-guidance/potential-based)
        # x = self.Shield(x, xp1)
        # x = self.GD(x, xp1)

        ####################### SafeDiffusers 
        # 3. 应用安全修正 (Invariance / CBF)
        # x = xp1 # for training only
        # x = self.invariance(x, xp1)    # RoS
        # x = self.invariance_cf(x, xp1)  # RoS closed form
        # x = self.invariance_relax(x, xp1, t) # ReS
        # x = self.invariance_relax_cf(x, xp1, t)   #ReS closed form    
        # x = self.invariance_time(x, xp1, t)   # TVS
        # x = self.invariance_time_cf(x, xp1, t)  # TVS closed form
        # x = self.invariance_relax_narrow(x, xp1, t)  # narrow passage case

        ####################### Applying SafeDiffusers to only the last 10 steps
        # if t <= 10:  #10
        #     # x = self.invariance_relax(x, xp1, t)  #done
        #     # x = self.invariance_relax_narrow(x, xp1, t)

        #     x = self.GD(x, xp1)
        # else:
        #     x = xp1
        #     xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        #     yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        #     off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        #     off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        #     b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1
        #     self.safe1 = torch.min(b[:,0])
        #     xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        #     yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        #     off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        #     off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1
        #     b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        #     self.safe2 = torch.min(b[:,0])

        
        ###################### umaze case
        # x = self.invariance_umaze(x, xp1)   #umaze
        # x = self.invariance_umaze_relax(x, xp1, t)   #umaze
        #-----------------------------------------end--------------------------------------------------#
        # return x

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        """
        input:
            shape: (batch_size, horizon, transition_dim)
            cond: dict
        output:
            x: [batch_size, horizon, transition_dim] 
        """
        print(f"Sampling using method {self.safe_method}....")

        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(self.sample_end_timestep, self.n_timesteps)):  #-50 change here for the number of diffusion steps,
            if i < 0:
                i = 0
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)
        

        progress.close()
        # pdb.set_trace()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x