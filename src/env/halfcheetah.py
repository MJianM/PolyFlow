import numpy as np
import gymnasium
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from tqdm import trange


class HalfCheetahEnv:
    def __init__(
        self,
        leg_limit: float = 1.2,
        torsion_limit: float = 0.8,
        bound_limit: float = 1.0
    ):

        self.env = gymnasium.make("HalfCheetah-v5", render_mode="rgb_array")
        self.leg_limit = leg_limit # 腿部驱动限制
        self.torsion_limit = torsion_limit # 躯干扭转限制
        self.bound_limit = bound_limit # 关节力矩限制
        print(f"Leg limit: {self.leg_limit}  Torsion limit: {self.torsion_limit}  Bound limit: {self.bound_limit}")


    def safety_check(self, traj_act: np.array, eps=1e-3) -> np.ndarray:
        """
        检查动作轨迹是否满足约束。
        
        :param traj_act: 动作轨迹数据，形状为 [B, H, D] 
                         B: Batch size, H: Horizon, D: Dimension (d = 6)
        :return: 布尔数组 [B]，表示每条轨迹是否安全 (True 表示安全/满足约束)

        halfcheetah 中动作维度排列为：
            后大腿，后小腿，后脚，前大腿，前小腿，前脚

        """
        u0, u1 = traj_act[:, :, 0], traj_act[:, :, 1]
        u3, u4 = traj_act[:, :, 3], traj_act[:, :, 4]

        # 腿部液压驱动限制
        leg_flag = (u0 + u1 < self.leg_limit + eps) & \
            (u3 + u4 < self.leg_limit + eps)
        
        # 躯干扭转限制
        torsion_flag = (u0 - u3 < self.torsion_limit + eps) & \
            (u3 - u0 < self.torsion_limit + eps)

        flag = leg_flag & torsion_flag

        is_safe = np.all(flag, axis=1)

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
        act_dim = self.env.action_space.shape[0]
        
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
        act_traj_list = []

        try:
            for i in range(n_episodes):
                print(f"Rollout {i}...")
                obs_list = []
                obs_expand_list = []
                rew_list = []
                act_list = []
                
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
                    act_list.append(action)
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
                        act_traj = np.stack(act_list)
                        
                        obs_traj_list.append(traj)
                        obs_expand_traj_list.append(traj_expand)
                        rew_traj_list.append(rew)
                        act_traj_list.append(act_traj)
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
        total_traj = len(act_traj_list)
        for act_traj in act_traj_list:
            flag = self.safety_check(act_traj.reshape(1, -1, act_dim))[0]
            if not flag:
                unsafe_cnt += 1

        safety_ratio = float(total_traj - unsafe_cnt) / total_traj if total_traj > 0 else 0.0

        metrics = {
            'ret_mean': ret_mean,
            'ret_std': ret_std,
            'safety_ratio': safety_ratio,
            'act_traj_list': act_traj_list, # 新增动作rollout轨迹列表
        }

        return obs_traj_list, obs_expand_traj_list, ret_list, metrics


    def _get_halfcheetah_skeleton(self, obs, x_pos):
        """
        根据观测值和当前的绝对 X 坐标，计算 halfcheetah 的骨架线段。
        用于 matplotlib 绘制火柴人。
        
        Args:
            obs: 单步观测值，shape (17,) 或 (11,)。
                 我们假设前8维度包含位置信息: 
                 [root_z, root_y, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
            x_pos: 躯干中心的绝对 X 坐标 (来自 info 或 env.data.qpos[0])
            
        Returns:
            lines: list of [(x_start, y_start), (x_end, y_end)]
                   用于 matplotlib.collections.LineCollection
        """
        # --- 1. 提取状态变量 ---
        # 注意：obs[0] 是 z 坐标 (root_z)，obs[1] 是躯干角度 (root_y)
        # 之后的索引依次是后腿关节和前腿关节
        root_z = obs[0]
        root_angle = obs[1]
        
        # 后腿角度 (Back Leg)
        bthigh_ang = obs[2]
        bshin_ang = obs[3]
        bfoot_ang = obs[4]
        
        # 前腿角度 (Front Leg)
        fthigh_ang = obs[5]
        fshin_ang = obs[6]
        ffoot_ang = obs[7]

        # --- 2. 定义几何参数 (参考 HalfCheetah XML 文件估算) ---
        # 躯干长度约为 1.0m，后腿挂载点在后方，前腿在中心偏前
        torso_len = 1.0  
        back_hip_offset = -0.5 # 后腿髋关节相对于躯干中心的偏移
        front_hip_offset = 0.5 # 前腿髋关节相对于躯干中心的偏移 (头部方向)
        
        # 腿部连杆长度
        thigh_len = 0.45
        shin_len = 0.5
        foot_len = 0.45
        
        # --- 3. 辅助函数：根据长度和绝对角度计算偏移向量 ---
        # 假设 0 度垂直向下 (这符合 MuJoCo 的视觉直觉，但也取决于坐标系定义)
        # 这里使用标准的三角函数：x += len * sin(ang), y -= len * cos(ang)
        # 意味着角度增加是逆时针旋转
        def get_delta(length, angle):
            return length * np.sin(angle), -length * np.cos(angle)

        # --- 4. 计算关键点坐标 (Forward Kinematics) ---
        
        # 4.1 躯干 (Torso)
        # 躯干是一根杆，围绕中心 (x_pos, root_z) 旋转
        # 计算躯干的首尾 (Head & Butt)
        # 躯干角度 0 度表示水平向右。
        # dx_torso = (torso_len/2) * cos(root_angle)
        # dy_torso = (torso_len/2) * sin(root_angle)
        dx_head = (torso_len / 2.0) * np.cos(root_angle)
        dy_head = (torso_len / 2.0) * np.sin(root_angle)
        
        torso_center = np.array([x_pos, root_z])
        head_pos = torso_center + np.array([dx_head, dy_head])
        butt_pos = torso_center - np.array([dx_head, dy_head])

        # 4.2 后腿 (Back Leg)
        # 后髋关节位置 (附着在躯干后侧)
        # 为了视觉清晰，我们将后髋关节定在 Butt 附近
        b_hip_pos = butt_pos 
        
        # 角度累加：绝对角度 = 躯干角度 + 相对关节角度
        # 注意：这里需要根据具体的 XML 定义调整初始相位。
        # HalfCheetah 的腿在 0 度时通常是向下的，因此直接累加即可。
        b_thigh_global = root_angle + bthigh_ang
        b_shin_global  = b_thigh_global + bshin_ang
        b_foot_global  = b_shin_global + bfoot_ang
        
        dx, dy = get_delta(thigh_len, b_thigh_global)
        b_knee_pos = b_hip_pos + np.array([dx, dy])
        
        dx, dy = get_delta(shin_len, b_shin_global)
        b_ankle_pos = b_knee_pos + np.array([dx, dy])
        
        dx, dy = get_delta(foot_len, b_foot_global)
        b_toe_pos = b_ankle_pos + np.array([dx, dy])

        # 4.3 前腿 (Front Leg)
        # 前髋关节位置 (附着在躯干前侧，接近 Head 但不要完全重合以便区分)
        # 简单起见，我们假设它挂在躯干中心向头方向 0.2m 处，或者直接挂在 Head 处
        # 这里为了视觉像 HalfCheetah，通常挂在头部稍微靠后的位置
        f_hip_offset_vec = np.array([back_hip_offset + torso_len, 0]) # 简化处理
        # 让我们直接用 Head 作为前腿挂载点，或者稍微靠后一点
        f_hip_pos = head_pos # 简单处理，或者计算特定的 offset
        
        f_thigh_global = root_angle + fthigh_ang
        f_shin_global  = f_thigh_global + fshin_ang
        f_foot_global  = f_shin_global + ffoot_ang

        dx, dy = get_delta(thigh_len, f_thigh_global)
        f_knee_pos = f_hip_pos + np.array([dx, dy])
        
        dx, dy = get_delta(shin_len, f_shin_global)
        f_ankle_pos = f_knee_pos + np.array([dx, dy])
        
        dx, dy = get_delta(foot_len, f_foot_global)
        f_toe_pos = f_ankle_pos + np.array([dx, dy])

        # --- 5. 构建线段列表 ---
        lines = [
            (butt_pos, head_pos),      # 躯干
            (b_hip_pos, b_knee_pos),   # 后大腿
            (b_knee_pos, b_ankle_pos), # 后小腿
            (b_ankle_pos, b_toe_pos),  # 后脚
            (f_hip_pos, f_knee_pos),   # 前大腿
            (f_knee_pos, f_ankle_pos), # 前小腿
            (f_ankle_pos, f_toe_pos),  # 前脚
        ]
        
        return lines

    def plot_expand_trajectory(
            self, traj_expand_list, max_plot=5, save_path=None
    ):
        """
        完全抛弃 MuJoCo Render，使用 Matplotlib 绘制骨架侧视图。
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
                lines = self._get_halfcheetah_skeleton(obs_seq[t], x_seq[t])
                
                # 绘制身体部件
                # alpha 设置透明度，让旧的帧稍微淡一点，或者保持清晰
                lc = LineCollection(lines, colors=colors[t], linewidths=2, alpha=0.8)
                ax.add_collection(lc)
                
                # 可选：画出躯干中心的一个点，方便看质心轨迹
                ax.plot(x_seq[t], obs_seq[t, 0], marker='o', markersize=6, color=colors[t], alpha=1.0)


            # --- 4. 设置坐标轴 ---
            ax.set_aspect('equal') # 关键！保证物理比例 1:1，否则高度看起来会变形
            ax.set_xlabel('Position X (m)')
            ax.set_ylabel('Height Z (m)')
            ax.set_title(f"Trajectory {i+1} (Length: {max_x - min_x:.2f}m)")
            
            # 限制 Y 轴范围，防止画出天空
            ax.set_ylim(-0.1, max(2.0, 1.25 + 0.5))
            # 限制 X 轴范围
            ax.set_xlim(min_x - 0.5, max_x + 0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
