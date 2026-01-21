import numpy as np
import gymnasium
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from tqdm import trange


class Walker2dEnv:
    def __init__(
            self,
            height_limit: float = 1.5,
            use_cpx: int = 0,
            vel_scale: float = 0.01,
            height_min: float = 0.9,
            v_max: float = 2.5,
            v_min: float = -2.5
    ):

        self.env = gymnasium.make("Walker2d-v5", render_mode="rgb_array")
        self.height_limit = height_limit
        self.use_cpx = use_cpx # 是否使用复杂约束 0 不使用 1 使用simple 2 使用hard
        self.vel_scale = vel_scale
        self.height_min = height_min
        self.v_max = v_max
        self.v_min = v_min
        print(f"Env Walker use complex constraints: {self.use_cpx}")


    def safety_check(self, traj_obs: np.array, ignore_first_horizon=True, eps=1e-3) -> np.ndarray:
        """
        检查轨迹是否满足高度限制约束。
        
        :param traj_obs: 观测轨迹数据，形状为 [B, H, D] 
                         B: Batch size, H: Horizon, D: Dimension (Walker2d obs dim=17)
        :return: 布尔数组 [B]，表示每条轨迹是否安全 (True 表示安全/满足约束)
        """
        # 1. 提取高度信息
        # obs[0] 是 z-coordinate (高度)
        # heights 形状变为: [B, H]
        heights = traj_obs[:, :, 0]

        # 提取速度信息
        vels = traj_obs[:, :, 9]

        # 2. 检查约束
        # 我们需要整条轨迹的所有点都满足 height < height_limit
        # axis=1 表示沿着时间步(Horizon)方向进行逻辑与(AND)操作
        if ignore_first_horizon:
            # horizon=0 时刻不检查
            heights = heights[:, 1:]
            vels = vels[:, 1:]

        if self.use_cpx == 1.0:
            is_safe = np.all(heights + self.vel_scale * vels <= self.height_limit + eps, axis=1)
        elif self.use_cpx == 2.0:
            flag = (heights + self.vel_scale * vels <= self.height_limit + eps) & \
                (heights >= self.height_min - eps) & \
                (vels >= self.v_min - eps) & (vels <= self.v_max + eps)
            is_safe = np.all(flag, axis=1)
        else:
            is_safe = (np.ones(heights.shape[0]) > 0)
        
        return is_safe

    def _get_x_pos(self):
        """辅助函数：直接从 MuJoCo 模拟器数据中获取当前 X 轴绝对位置"""
        return self.env.unwrapped.data.qpos[0]

    def rollout(self, policy, n_episodes, seed=42, 
                is_video: bool = False,         # 是否录制
                video_episodes: int = 1,        # 录制前几集
                video_path: str = "videos"      # 保存路径
                ):

        obs_dim = self.env.observation_space.shape[0]
        
        # 1. 【关键】准备环境实例
        # 默认使用原始环境
        env_to_use = self.env

        # 如果需要视频，创建一个临时的 Wrapper 环境
        if is_video:
            # 定义触发器：episode_id 是从 wrapper 初始化开始计数的
            # 所以这里 lambda x: x < video_episodes 表示录制本次 rollout 的前 N 个
            trigger = lambda ep_id: ep_id < video_episodes
            
            # 使用 RecordVideo 包裹原始环境
            # disable_logger=True 防止终端刷屏 "MoviePy - Writing video..."
            env_to_use = RecordVideo(
                self.env, 
                video_folder=video_path, 
                episode_trigger=trigger,
                name_prefix="rollout_eval", # 文件名前缀
                disable_logger=False 
            )

        obs_traj_list = []
        obs_expand_traj_list = []
        rew_traj_list = []

        try:
            for i in range(n_episodes):
                print(f"Rollout {i}...")
                obs_list = []
                obs_expand_list = []
                rew_list = []
                
                done = False
                current_seed = seed + i
                
                # 2. 【关键】使用 env_to_use 而不是 self.env
                # RecordVideo 会在 reset 时自动判断是否开启录制
                obs, info = env_to_use.reset(seed=current_seed)
                
                x_pos = self._get_x_pos()
                obs_list.append(obs)
                obs_expand_list.append(np.concatenate([obs, [x_pos]]))

                while not done:
                    cond = {0: obs.reshape(1, obs_dim)}
                    action, _, _, _, _, _ = policy(cond, batch_size=1)
                    action = action.flatten()
                    
                    # 使用 env_to_use.step
                    # 如果当前 episode 被触发录制，wrapper 会自动保存 frame
                    obs, reward, truncation, termination, info = env_to_use.step(action)
                    
                    x_pos = self._get_x_pos()
                    
                    obs_list.append(obs)
                    obs_expand_list.append(np.concatenate([obs, [x_pos]]))
                    rew_list.append(reward)

                    if truncation or termination:
                        done = True
                        traj = np.stack(obs_list)[:-1] 
                        traj_expand = np.stack(obs_expand_list)[:-1]
                        rew = np.array(rew_list)
                        
                        obs_traj_list.append(traj)
                        obs_expand_traj_list.append(traj_expand)
                        rew_traj_list.append(rew)
                        break
        finally:
            # 3. 【关键】清理工作
            # 如果使用了 Wrapper，必须 close 才能确保视频文件写入磁盘完成
            if is_video:
                env_to_use.close()

        # ... (后续统计代码完全保持不变) ...
        ret_list = [np.sum(r) for r in rew_traj_list]
        
        if len(ret_list) > 0:
            ret_mean = np.mean(ret_list)
            ret_std = np.std(ret_list)
        else:
            ret_mean, ret_std = 0.0, 0.0

        unsafe_cnt = 0
        total_traj = len(obs_traj_list)
        for obs_traj in obs_traj_list:
            flag = self.safety_check(obs_traj.reshape(1, -1, obs_dim), ignore_first_horizon=False)[0]
            if not flag:
                unsafe_cnt += 1

        safety_ratio = float(total_traj - unsafe_cnt) / total_traj if total_traj > 0 else 0.0

        metrics = {
            'ret_mean': ret_mean,
            'ret_std': ret_std,
            'safety_ratio': safety_ratio,
        }

        return obs_traj_list, obs_expand_traj_list, ret_list, metrics


    def _get_walker_skeleton(self, obs, x_pos):
        """
        根据观测值和当前的绝对 X 坐标，计算 Walker2d 的骨架线段。
        用于 matplotlib 绘制火柴人。
        """
        # --- 1. 解析观测数据 ---
        # obs shape: (17,)
        # 0: root_z (height)
        # 1: root_angle (pitch)
        # 2-4: right leg (thigh, leg, foot)
        # 5-7: left leg (thigh, leg, foot)
        
        root_z = obs[0]
        root_ang = obs[1]
        
        # 右腿关节角度
        theta_thigh_r = obs[2]
        theta_leg_r = obs[3]
        theta_foot_r = obs[4]
        
        # 左腿关节角度
        theta_thigh_l = obs[5]
        theta_leg_l = obs[6]
        theta_foot_l = obs[7]

        # --- 2. 定义几何尺寸 (基于 XML 解析) ---
        L_TORSO_UP = 0.2    # 躯干中心向上长度 (头部)
        L_TORSO_DOWN = 0.2  # 躯干中心到髋关节长度
        L_THIGH = 0.45      # 大腿长度
        L_LEG = 0.5         # 小腿长度
        L_FOOT = 0.2        # 脚掌长度

        # --- 3. 辅助函数：2D 旋转 ---
        def rotate(x, z, theta):
            """将向量 (x, z) 旋转 theta 弧度"""
            c, s = np.cos(theta), np.sin(theta)
            return x * c - z * s, x * s + z * c

        # --- 4. 计算全局角度 (累加) ---
        # 注意：MuJoCo Walker2d 的初始姿态是直立的，关节角度为 0 表示伸直。
        # 角度正方向根据 XML axis="0 -1 0" 定义
        
        # 躯干角度
        ang_root = root_ang
        
        # 右腿运动学链
        ang_thigh_r_global = ang_root + theta_thigh_r
        ang_leg_r_global = ang_thigh_r_global + theta_leg_r
        ang_foot_r_global = ang_leg_r_global + theta_foot_r
        
        # 左腿运动学链
        ang_thigh_l_global = ang_root + theta_thigh_l
        ang_leg_l_global = ang_thigh_l_global + theta_leg_l
        ang_foot_l_global = ang_leg_l_global + theta_foot_l

        # --- 5. 计算关节点坐标 (Forward Kinematics) ---
        
        # 根节点 (Torso Center)
        p_root = np.array([x_pos, root_z])
        
        # 躯干顶部 (Head)
        dx, dz = rotate(0, L_TORSO_UP, ang_root)
        p_head = p_root + np.array([dx, dz])
        
        # 髋关节 (Hip) - 躯干底部
        dx, dz = rotate(0, -L_TORSO_DOWN, ang_root)
        p_hip = p_root + np.array([dx, dz])
        
        # --- 右腿链 ---
        # 膝盖 (Knee R)
        dx, dz = rotate(0, -L_THIGH, ang_thigh_r_global)
        p_knee_r = p_hip + np.array([dx, dz])
        
        # 脚踝 (Ankle R)
        dx, dz = rotate(0, -L_LEG, ang_leg_r_global)
        p_ankle_r = p_knee_r + np.array([dx, dz])
        
        # 脚尖 (Toe R) - 假设脚向前方伸展
        dx, dz = rotate(L_FOOT, 0, ang_foot_r_global)
        p_toe_r = p_ankle_r + np.array([dx, dz])
        
        # --- 左腿链 ---
        # 膝盖 (Knee L)
        dx, dz = rotate(0, -L_THIGH, ang_thigh_l_global)
        p_knee_l = p_hip + np.array([dx, dz])
        
        # 脚踝 (Ankle L)
        dx, dz = rotate(0, -L_LEG, ang_leg_l_global)
        p_ankle_l = p_knee_l + np.array([dx, dz])
        
        # 脚尖 (Toe L)
        dx, dz = rotate(L_FOOT, 0, ang_foot_l_global)
        p_toe_l = p_ankle_l + np.array([dx, dz])

        # --- 6. 构建线段列表 ---
        # 格式: [(x_start, z_start), (x_end, z_end)]
        lines = [
            [p_head, p_hip],            # 躯干
            [p_hip, p_knee_r],          # 右大腿
            [p_knee_r, p_ankle_r],      # 右小腿
            [p_ankle_r, p_toe_r],       # 右脚
            [p_hip, p_knee_l],          # 左大腿
            [p_knee_l, p_ankle_l],      # 左小腿
            [p_ankle_l, p_toe_l],       # 左脚
        ]
        
        return lines

    def plot_expand_trajectory(
            self, traj_expand_list, plot_height_limit=True, max_plot=5, save_path=None
    ):
        """
        完全抛弃 MuJoCo Render，使用 Matplotlib 绘制 Walker 骨架侧视图。
        """
        n_plot = min(len(traj_expand_list), max_plot)
        if n_plot == 0: return

        # 设置图表大小，保持一定的宽高比
        fig, axes = plt.subplots(n_plot, 1, figsize=(12, 4 * n_plot))
        if n_plot == 1: axes = [axes]

        stride = 5  # 抽样频率，每隔5帧画一个火柴人，避免过于密集

        for i in range(n_plot):
            ax = axes[i]
            traj = traj_expand_list[i]  # [T, 12]
            
            obs_seq = traj[:, :11]
            x_seq = traj[:, 11]
            
            # --- 1. 绘制地板 ---
            min_x, max_x = np.min(x_seq), np.max(x_seq)
            ax.plot([min_x - 1, max_x + 1], [0, 0], color='black', linewidth=2) # 地面

            # --- 2. 遍历轨迹绘制骨架 ---
            # 使用颜色映射表示时间进度 (例如：浅蓝 -> 深蓝)
            T = len(traj)
            colors = cm.viridis(np.linspace(0, 1, T))

            for t in range(0, T, stride):
                # 获取当前时刻的骨架线段
                lines = self._get_walker_skeleton(obs_seq[t], x_seq[t])
                
                # 绘制身体部件
                # alpha 设置透明度，让旧的帧稍微淡一点，或者保持清晰
                lc = LineCollection(lines, colors=colors[t], linewidths=2, alpha=0.8)
                ax.add_collection(lc)
                
                # 可选：画出躯干中心的一个点，方便看质心轨迹
                ax.plot(x_seq[t], obs_seq[t, 0], marker='o', markersize=6, color=colors[t], alpha=1.0)

            # --- 3. 绘制高度限制线 ---
            if plot_height_limit:
                ax.axhline(y=self.height_limit, color='red', linestyle='--', linewidth=2, label='Height Limit')
                # 在限制线下方填充一点红色背景，表示警告区域（可选）
                # ax.axhspan(self.height_limit, self.height_limit + 0.5, color='red', alpha=0.05)
                
                if i == 0: ax.legend(loc='upper right')

            # --- 4. 设置坐标轴 ---
            ax.set_aspect('equal') # 关键！保证物理比例 1:1，否则高度看起来会变形
            ax.set_xlabel('Position X (m)')
            ax.set_ylabel('Height Z (m)')
            ax.set_title(f"Trajectory {i+1} (Length: {max_x - min_x:.2f}m)")
            
            # 限制 Y 轴范围，防止画出天空
            ax.set_ylim(-0.1, max(2.0, self.height_limit + 0.5))
            # 限制 X 轴范围
            ax.set_xlim(min_x - 0.5, max_x + 0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
