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

from .diffusion import Sample



def euler_step(model, x, cond, t, dt, **kwargs):
    """
    Standard Euler integration step: x_{new} = x + v * dt
    """
    # Predict velocity field v_t
    v_pred = model.model(x, cond, t)
    
    # Integration
    x_next = x + v_pred * dt
    
    return x_next, 0.0


def guided_euler_step(model, x, cond, t, dt, guide=None, scale=1.0, n_guide_steps=1, **kwargs):
    """
    Euler step with explicit guidance optimization.
    Matches the logic of your 'n_step_guided_p_sample'.
    """
    grad_value = 0.0
    
    # 1. Guidance Optimization (Optional Inner Loop)
    # 在计算这一步的流动方向之前，先根据 guide 修正当前位置 x
    if guide is not None and n_guide_steps > 0:
        # Clone current x to avoid modifying the integration path variable directly in case of errors
        x_guided = x.clone()

        for _ in range(n_guide_steps):
            # Critical Fix: 
            # ValueGuide.gradients calls x.requires_grad_().
            # This requires x to be a leaf variable (detached from graph).
            x_guided = x_guided.detach()

            # Enable grad specifically for the guide calculation
            with torch.enable_grad():
                # returns value (y) and gradient (grad)
                y, grad = guide.gradients(x_guided, cond, t)

            # Gradient Ascent (assuming we want to maximize the value y)
            # If y is a cost to minimize, change '+' to '-'
            x_guided = x_guided + scale * grad
            
            # Re-apply conditioning (Ensure constraints are met during optimization)
            x_guided = apply_conditioning(x_guided, cond, model.action_dim)
        
        # Update x to the optimized position
        x = x_guided
        grad_value = y.mean().item()

    # 2. ODE Flow Step
    v_pred = model.model(x, cond, t)
    x_next = x + v_pred * dt
    
    return x_next, grad_value

class FlowMatching(nn.Module):

    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps,
            loss_type='l2', action_weight=1.0, loss_discount=1.0, loss_weights=None):
        super().__init__()

        # for normalization
        self.mean = 0.0
        self.std = 0.0

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.n_timesteps = n_timesteps

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
            return:
                (horizon, transition_dim)
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
    
    def loss(self, x, cond):
        """
        compute batch flow matching loss
        
        :param self: Description
        :param x: (batch, horizon, act_dim+obs_dim)
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        return: 
            loss: 
            info: dict
        
          model API: self.model(x, cond, t)
        """
        batch_size = len(x)
        device = x.device

        # 1. Sample continuous time t uniform in [0, 1]
        # shape: (batch,)
        t = torch.rand((batch_size,), device=device)

        # 2. Sample noise x_0 from standard normal
        x_0 = torch.randn_like(x)
        x_1 = x # The target data

        # 3. Compute Interpolation (Optimal Transport Path)
        # Formula: x_t = (1 - t) * x_0 + t * x_1
        # Reshape t for broadcasting: (batch,) -> (batch, 1, 1)
        t_b = t.view(batch_size, 1, 1)
        x_t = (1 - t_b) * x_0 + t_b * x_1

        # 4. Apply conditioning to model input
        # 这一点借鉴了你的 Diffusion 实现：强制模型输入的特定维度（如起点）符合条件
        # 这有助于模型在推理时感知到正确的上下文
        x_t = apply_conditioning(x_t, cond, self.action_dim)

        # 5. Model Forward
        # Predict the vector field v_t
        # model signature: (x, cond, t)
        v_pred = self.model(x_t, cond, t)

        # 6. Compute Target Velocity
        # For OT path x_t = (1-t)x_0 + t*x_1, the derivative dx/dt is (x_1 - x_0)
        v_target = x_1 - x_0

        # Note: 
        # 在 Diffusion 实现中，你对 output (x_recon) 再次使用了 apply_conditioning。
        # 在 Flow Matching 中，模型预测的是速度 v 而不是状态 x。
        # 对速度应用状态的 conditioning 数值上通常是不对的，且 v_target 已经包含了由 x_1 确定的方向信息，
        # 所以这里通常不需要对 v_pred 进行 apply_conditioning。

        # 7. Compute Loss
        loss, info = self.loss_fn(v_pred, v_target)

        return loss, info

    def forward(self, cond, verbose, **kwargs):
        return self.conditional_sample(cond, verbose, **kwargs)

    @torch.no_grad()
    def conditional_sample(self, cond, verbose=True, return_chain=True, **sample_kwargs): 
        """
        conditional_sample
        
        :param self: Description
        :param cond: dict {horizon_idx: (batch, obs_dim)}
        :param verbose: Description
        :param return_chain: Description
        :param sample_kwargs: Description
        """
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, verbose=verbose, return_chain=return_chain, **sample_kwargs)    # debug

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=None, guide=None, **sample_kwargs):
        """
        Docstring for p_sample_loop
        
        :param self: Description
        :param shape: Description
        :param cond: Description
        :param verbose: Description
        :param return_chain: Description
        :param sample_fn: Description
        :param guide: Description
        :param sample_kwargs: Description
        """
        device = next(iter(self.model.parameters())).device # assuming model is on correct device
        batch_size = shape[0]

        # 1. Start from Noise (t=0)
        x = torch.randn(shape, device=device)
        
        # 2. Force start condition
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        
        # Setup progress bar
        iterator = range(self.n_timesteps)
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        
        # Define time step size dt = 1 / N
        dt = 1.0 / self.n_timesteps

        # Select sampling function
        if sample_fn is None:
            sample_fn = guided_euler_step if guide is not None else euler_step

        # 3. Integration Loop: t goes from 0 to 1
        for i in iterator:
            # Current time t (scalar)
            t_value = i / self.n_timesteps
            
            # Create batch time tensor
            t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

            # 4. Step: x_{t+1} <- x_t + v(x_t)*dt
            # sample_fn returns next_x and any info (like guidance value)
            x, values = sample_fn(self, x, cond, t, dt, guide=guide, **sample_kwargs)

            # 5. Re-apply conditioning
            # 这一步非常重要，确保轨迹始终“锚定”在起始点/终点约束上
            x = apply_conditioning(x, cond, self.action_dim)

            if return_chain: chain.append(x)

        progress.stamp()

        if return_chain: chain = torch.stack(chain, dim=1)
        
        # Return Sample namedtuple to match your interface
        return Sample(x, values, chain), 0


