import numpy as np
import gymnasium
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from tqdm import trange


class HopperEnv:
    def __init__(
            self,
            height_limit: float = 1.5
    ):

        self.env = gymnasium.make("Hopper-v5", render_mode="rgb_array")
        self.height_limit = height_limit


        # [新增] 保存初始状态，用于在绘图函数中通过观测值还原环境状态
        self.env.reset()
        self.init_qpos = self.env.unwrapped.data.qpos.ravel().copy()
        self.init_qvel = self.env.unwrapped.data.qvel.ravel().copy()

        # # 4. 计算 像素-物理 缩放比例 (Calibration)
        # frame = self.env.render()
        # self.frame_h, self.frame_w, _ = frame.shape
        
        # # Hopper 默认垂直 FOV 为 45 度
        # fovy_deg = 45.0 
        # # 视野物理高度 = 2 * d * tan(fov/2)
        # view_height_m = 2 * self.cam_distance * np.tan(np.deg2rad(fovy_deg) / 2)
        
        # # pixels_per_meter: 每米对应多少个像素
        # self.pixels_per_meter = self.frame_h / view_height_m

    def safety_check(self, traj_obs: np.array) -> np.ndarray:
        """
        检查轨迹是否满足高度限制约束。
        
        :param traj_obs: 观测轨迹数据，形状为 [B, H, D] 
                         B: Batch size, H: Horizon, D: Dimension (Hopper obs dim=11)
        :return: 布尔数组 [B]，表示每条轨迹是否安全 (True 表示安全/满足约束)
        """
        # 1. 提取高度信息
        # 在 Hopper 环境中，obs[0] 是 z-coordinate (高度)
        # heights 形状变为: [B, H]
        heights = traj_obs[:, :, 0]

        # 2. 检查约束
        # 我们需要整条轨迹的所有点都满足 height < height_limit
        # axis=1 表示沿着时间步(Horizon)方向进行逻辑与(AND)操作
        is_safe = np.all(heights < self.height_limit, axis=1)
        
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
            flag = self.safety_check(obs_traj.reshape(1, -1, obs_dim))[0]
            if not flag:
                unsafe_cnt += 1
        
        safety_ratio = float(total_traj - unsafe_cnt) / total_traj if total_traj > 0 else 0.0

        metrics = {
            'ret_mean': ret_mean,
            'ret_std': ret_std,
            'safety_ratio': safety_ratio
        }

        return obs_traj_list, obs_expand_traj_list, ret_list, metrics


    def _get_hopper_skeleton(self, obs, x_pos):
        """
        修正版：解决腿部倒置问题，并校准关节弯曲方向。
        """
        z = obs[0]
        q_torso = obs[1]
        q_thigh = obs[2]
        q_leg   = obs[3]
        q_foot  = obs[4]

        # 1. 几何参数 (保持不变)
        L_torso_half = 0.20
        L_thigh      = 0.45
        L_leg        = 0.50
        L_foot_front = 0.26
        L_foot_back  = 0.13

        # 2. 角度计算 (核心修正)
        # -----------------------------------------------------------
        # Torso: 0度=垂直向上(pi/2)。q>0 为前倾(顺时针)，故减去 q
        theta_torso = np.pi/2 - q_torso
        
        # Thigh (大腿): 
        # (1) 基础方向：相对于躯干，大腿默认是向下的，所以加 pi (180度)。
        # (2) 关节方向：XML中 thigh range="-150 0"，负值代表抬腿(向前)。
        #     在我们的坐标系中，从向下(-90)变到向前(0)，角度需要增加。
        #     因此负的 q 应该贡献正的增量。所以这里是 - q_thigh。
        theta_thigh = theta_torso + np.pi - q_thigh
        
        # Leg (小腿):
        # XML中 leg range="-150 0"，负值代表向后弯曲(收腿)。
        # 这一级是相对于大腿的。如果大腿向前(0度)，小腿收缩应变为负角度(向下)。
        # 所以负的 q 应该贡献负的增量。这里保持 + q_leg。
        theta_leg   = theta_thigh + q_leg
        
        # Foot (脚):
        # 简单累加
        theta_foot  = theta_leg + q_foot
        
        # Foot Visual:
        # 脚掌几何体默认是水平的。当骨骼垂直向下(-90)时，脚掌指向前方(0)。
        # 所以视觉角度 = 骨骼角度 + 90度
        theta_foot_visual = theta_foot + np.pi/2 
        # -----------------------------------------------------------

        # 3. 坐标推导 (FK)
        root_x, root_z = x_pos, z

        # Head & Hip
        # Hip 在 Root 下方
        head_x = root_x + L_torso_half * np.cos(theta_torso)
        head_z = root_z + L_torso_half * np.sin(theta_torso)
        
        hip_x = root_x - L_torso_half * np.cos(theta_torso)
        hip_z = root_z - L_torso_half * np.sin(theta_torso)

        # Knee
        knee_x = hip_x + L_thigh * np.cos(theta_thigh)
        knee_z = hip_z + L_thigh * np.sin(theta_thigh)

        # Ankle
        ankle_x = knee_x + L_leg * np.cos(theta_leg)
        ankle_z = knee_z + L_leg * np.sin(theta_leg)

        # Foot
        toe_x = ankle_x + L_foot_front * np.cos(theta_foot_visual)
        toe_z = ankle_z + L_foot_front * np.sin(theta_foot_visual)
        
        heel_x = ankle_x - L_foot_back * np.cos(theta_foot_visual)
        heel_z = ankle_z - L_foot_back * np.sin(theta_foot_visual)

        lines = [
            [(hip_x, hip_z), (head_x, head_z)],
            [(hip_x, hip_z), (knee_x, knee_z)],
            [(knee_x, knee_z), (ankle_x, ankle_z)],
            [(heel_x, heel_z), (toe_x, toe_z)]
        ]
        return lines

    def plot_expand_trajectory(
            self, traj_expand_list, plot_height_limit=True, max_plot=5, save_path=None
    ):
        """
        完全抛弃 MuJoCo Render，使用 Matplotlib 绘制 Hopper 骨架侧视图。
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
                lines = self._get_hopper_skeleton(obs_seq[t], x_seq[t])
                
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
