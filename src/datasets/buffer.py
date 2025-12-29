import numpy as np

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:
    """
    ReplayBuffer 用于存储离线强化学习的轨迹数据。
    它预先分配固定大小的内存块，并在加载数据后裁剪多余部分。
    支持通过属性访问数据 (例如 buffer.observations)。
    """

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        """
        初始化 ReplayBuffer。
        
        参数:
            max_n_episodes (int): 预分配的最大 episode 数量。
            max_path_length (int): 每个 episode 的最大时间步长。
            termination_penalty (float): 如果 episode 提前终止 (terminal=True 且非超时)，在最后一步奖励上施加的惩罚。
        """
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int_),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            将字典中的键注册为对象属性，
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        """
        为指定的键 (如 'observations', 'actions') 分配内存。
        形状为 [max_n_episodes, max_path_length, dim]。
        """
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        """
        向 buffer 中添加一条轨迹。
        
        参数:
            path (dict): 包含 'observations', 'actions', 'rewards', 'terminals' 等键的字典。
        """
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        ## if first path added, set keys based on contents
        # 如果是第一次添加，根据 path 中的键初始化 self.keys
        self._add_keys(path)

        ## add tracked keys in path
        # 遍历所有键，将数据存入预分配的数组中
        for key in self.keys:
            array = atleast_2d(path[key])
            # 如果该键尚未分配内存，则进行分配
            if key not in self._dict: self._allocate(key, array)
            # 将数据存入当前 episode 对应的位置
            self._dict[key][self._count, :path_length] = array

        ## penalize early termination
        # 如果轨迹因失败而终止 (terminal=True) 且不是因为超时 (timeouts=False)，则施加惩罚
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        ## record path length
        self._dict['path_lengths'][self._count] = path_length

        ## increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        """
        加载完成后，裁剪掉未使用的预分配内存，并注册属性以便通过 . 访问。
        """
        ## remove extra slots
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

if __name__ == '__main__':
    # 1. 初始化 ReplayBuffer
    # 预分配空间：最多存储 10 个 episodes，每个 episode 最长 100 步。
    # 如果一个 episode 因为失败而提前终止，其最后一步的奖励将被减去 10。
    buffer = ReplayBuffer(max_n_episodes=10, max_path_length=100, termination_penalty=-10)
    print("----------- Initialized Buffer -----------")
    print(buffer)

    # 2. 创建并添加一些模拟的轨迹数据
    print("\n----------- Adding Paths -----------")
    # 轨迹 1: 长度为 20，正常结束
    path1_len = 20
    path1 = {
        'observations': np.random.rand(path1_len, 4), # 观测维度为 4
        'actions': np.random.rand(path1_len, 2),      # 动作维度为 2
        'rewards': np.ones((path1_len, 1)),
        'terminals': np.zeros((path1_len, 1), dtype=bool),
        'timeouts': np.zeros((path1_len, 1), dtype=bool),
    }
    path1['timeouts'][-1] = True # 正常超时结束
    buffer.add_path(path1)
    print("Added path 1 with length 20.")

    # 轨迹 2: 长度为 15，提前失败终止
    path2_len = 15
    path2 = {
        'observations': np.random.rand(path2_len, 4),
        'actions': np.random.rand(path2_len, 2),
        'rewards': np.ones((path2_len, 1)),
        'terminals': np.zeros((path2_len, 1), dtype=bool),
        'timeouts': np.zeros((path2_len, 1), dtype=bool),
    }
    path2['terminals'][-1] = True # 提前失败终止
    buffer.add_path(path2)
    print("Added path 2 with length 15 (early termination).")

    # 3. Finalize 缓冲区，裁剪掉未使用的空间
    print("\n----------- Finalizing Buffer -----------")
    buffer.finalize()
    print("\n----------- Finalized Buffer State -----------")
    print(buffer)
    print(f"\nTotal episodes: {buffer.n_episodes}")
    print(f"Total steps: {buffer.n_steps}")
    print(f"Path lengths: {buffer.path_lengths}")
    print(f"Shape of observations: {buffer.observations.shape}")
    # 检查提前终止的惩罚是否已施加
    print(f"Reward of last step in terminated path (path 2): {buffer.rewards[1, path2_len-1]}")
