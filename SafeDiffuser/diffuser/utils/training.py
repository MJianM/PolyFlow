import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

# 一个辅助函数，可以无限地从数据加载器中循环取数据
def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        指数移动平均（Exponential Moving Average）
        用于在训练过程中维护一个模型的影子副本（EMA模型），
        这个模型的参数是过去多步训练模型参数的平滑平均。
        在推理时使用EMA模型通常能获得更稳定和更好的性能。
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            # 获取旧的EMA模型权重和当前模型权重
            old_weight, up_weight = ma_params.data, current_params.data
            # 更新EMA模型权重
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        # 如果旧权重不存在（第一次更新），直接返回新权重
        if old is None:
            return new
        # EMA更新公式
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    # Trainer类，封装了整个训练流程
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        use_condition=False,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=1,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=-1,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.use_condition = use_condition

        # diffusion_model: 核心的扩散模型
        self.model = diffusion_model

        self.device = next(self.model.parameters()).device
        # ema: EMA更新器
        self.ema = EMA(ema_decay)
        # ema_model: EMA模型，是主模型的深拷贝
        self.ema_model = copy.deepcopy(self.model)
        # update_ema_every: 每隔多少步更新一次EMA模型
        self.update_ema_every = update_ema_every

        # step_start_ema: 在多少步之后开始进行EMA更新
        self.step_start_ema = step_start_ema
        # log_freq: 日志打印频率
        self.log_freq = log_freq
        # sample_freq: 生成并渲染样本的频率
        self.sample_freq = sample_freq
        # save_freq: 保存模型的频率
        self.save_freq = save_freq
        # label_freq: 用于给保存的模型文件添加标签的频率
        self.label_freq = label_freq
        # save_parallel: 是否并行保存（例如，同步到云存储时）
        self.save_parallel = save_parallel

        # batch_size: 训练时的批次大小
        self.batch_size = train_batch_size
        # gradient_accumulate_every: 梯度累积步数，用于模拟更大的批次大小
        self.gradient_accumulate_every = gradient_accumulate_every

        # dataset: 训练数据集
        self.dataset = dataset
        # dataloader: 训练数据加载器，使用cycle函数实现无限循环
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        # dataloader_vis: 用于可视化的数据加载器，批次大小为1
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        # renderer: 用于渲染轨迹和生成图像/视频的工具
        self.renderer = renderer
        # optimizer: 优化器，这里使用Adam
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        # logdir: 日志和模型保存的根目录
        self.logdir = results_folder
        # bucket: 云存储桶，用于同步日志
        self.bucket = bucket
        # n_reference: 渲染参考样本的数量
        self.n_reference = n_reference
        # n_samples: 每次渲染时生成的样本数量
        self.n_samples = n_samples

        # 重置EMA模型参数并初始化训练步数
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        # 将主模型的参数复制到EMA模型中
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        # 如果训练步数未达到开始EMA的阈值，则重置EMA模型参数
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        # 使用EMA更新器更新EMA模型
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        '''
        执行n_train_steps个训练步骤。

        输入:
            n_train_steps (int): 要执行的训练步数。
        '''

        timer = Timer()
        for step in range(n_train_steps):
            # ----- 梯度累积循环 -----
            for i in range(self.gradient_accumulate_every):
                # 从数据加载器中获取一个批次的数据
                batch = next(self.dataloader)
                # 将数据移动到指定的设备（如GPU）
                batch = batch_to_device(batch, device=self.device)

                # 计算损失
                # batch 是一个 namedtuple，包含 trajectories 和 conditions
                # loss(*batch) 等价于 loss(batch.trajectories, batch.conditions)
                if self.use_condition:
                    loss, infos = self.model.loss(batch.trajectories, cond=batch.conditions)
                else:
                    loss, infos = self.model.loss(batch.trajectories, cond=None)
                # 对损失进行缩放，以适应梯度累积
                loss = loss / self.gradient_accumulate_every
                # 反向传播计算梯度
                loss.backward()

            # ----- 更新模型参数 -----
            # 使用累积的梯度更新模型参数
            self.optimizer.step()
            # 清空梯度
            self.optimizer.zero_grad()

            # ----- EMA 更新 -----
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # ----- 保存模型 -----
            if self.step % self.save_freq == 0:
                # 计算保存标签，用于区分不同的checkpoint
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            # ----- 打印日志 -----
            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            # # ----- 渲染和采样 -----
            # # 在训练开始时渲染参考样本
            # if self.step == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference)

            # # 定期渲染生成的样本
            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     self.render_samples(n_samples=self.n_samples)

            self.step += 1

    def save(self, epoch):
        '''
            将模型和EMA模型的状态保存到磁盘；
            如果指定了云存储桶，则同步到桶中。

            输入:
                epoch (int or str): 用于命名保存文件的时期或标签。
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            从磁盘加载模型和EMA模型的状态。

            输入:
                epoch (int or str): 要加载的模型的时期或标签。
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            渲染来自训练数据集的参考轨迹。

            输入:
                batch_size (int): 要渲染的参考轨迹数量。
        '''

        ## get a temporary dataloader to load a single batch
        # 获取一个临时的dataloader来加载一个批次的参考数据
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        # 从批次中获取轨迹和条件，并转换为numpy数组
        # trajectories shape: [batch_size, horizon, transition_dim]
        trajectories = to_np(batch.trajectories)
        # conditions shape: [batch_size, 1, observation_dim]
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        # 从轨迹中分离出观测值
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        # 反归一化观测值以进行渲染
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        # 使用渲染器将多条轨迹合成为一张图像并保存
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            从EMA扩散模型中采样并渲染生成的轨迹。

            输入:
                batch_size (int): 从数据集中获取的条件数量。
                n_samples (int): 每个条件要生成的样本数量。
        '''
        for i in range(batch_size):

            ## get a single datapoint
            # 从可视化数据加载器中获取一个数据点作为条件
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            # 使用EMA模型进行条件采样，生成轨迹
            samples = self.ema_model.conditional_sample(conditions, return_diffusion = False)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            # 从生成的样本中分离出归一化后的观测值
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            # 获取批次中的原始条件
            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            # 将初始条件与生成的观测序列拼接起来，形成完整的轨迹
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            # 反归一化完整的观测序列
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            # 渲染生成的轨迹并保存
            self.renderer.composite(savepath, observations)
