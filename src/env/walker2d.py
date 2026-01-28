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
        self.use_cpx = use_cpx # 0: unconstrained, 1: simple, 2: complex
        self.vel_scale = vel_scale
        self.height_min = height_min
        self.v_max = v_max
        self.v_min = v_min
        print(f"Env Walker use complex constraints: {self.use_cpx}")


    def safety_check(self, traj_obs: np.array, ignore_first_horizon=True, eps=1e-3) -> np.ndarray:


        heights = traj_obs[:, :, 0]

        vels = traj_obs[:, :, 9]

        if ignore_first_horizon:
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
        return self.env.unwrapped.data.qpos[0]

    def rollout(self, policy, n_episodes, seed=42, 
                is_video: bool = False,         # 是否录制
                video_episodes: int = 1,        # 录制前几集
                video_path: str = "videos"      # 保存路径
                ):

        obs_dim = self.env.observation_space.shape[0]
        

        env_to_use = self.env

        if is_video:
            trigger = lambda ep_id: ep_id < video_episodes
            
            env_to_use = RecordVideo(
                self.env, 
                video_folder=video_path, 
                episode_trigger=trigger,
                name_prefix="rollout_eval", 
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
                
                obs, info = env_to_use.reset(seed=current_seed)
                
                x_pos = self._get_x_pos()
                obs_list.append(obs)
                obs_expand_list.append(np.concatenate([obs, [x_pos]]))

                while not done:
                    cond = {0: obs.reshape(1, obs_dim)}
                    action, _, _, _, _, _ = policy(cond, batch_size=1)
                    action = action.flatten()
                    
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
            if is_video:
                env_to_use.close()

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

        # obs shape: (17,)
        # 0: root_z (height)
        # 1: root_angle (pitch)
        # 2-4: right leg (thigh, leg, foot)
        # 5-7: left leg (thigh, leg, foot)
        
        root_z = obs[0]
        root_ang = obs[1]
        
        theta_thigh_r = obs[2]
        theta_leg_r = obs[3]
        theta_foot_r = obs[4]
        
        theta_thigh_l = obs[5]
        theta_leg_l = obs[6]
        theta_foot_l = obs[7]

        L_TORSO_UP = 0.2    # 躯干中心向上长度 (头部)
        L_TORSO_DOWN = 0.2  # 躯干中心到髋关节长度
        L_THIGH = 0.45      # 大腿长度
        L_LEG = 0.5         # 小腿长度
        L_FOOT = 0.2        # 脚掌长度


        def rotate(x, z, theta):
            """将向量 (x, z) 旋转 theta 弧度"""
            c, s = np.cos(theta), np.sin(theta)
            return x * c - z * s, x * s + z * c

        
        ang_root = root_ang
        
        ang_thigh_r_global = ang_root + theta_thigh_r
        ang_leg_r_global = ang_thigh_r_global + theta_leg_r
        ang_foot_r_global = ang_leg_r_global + theta_foot_r
        
        ang_thigh_l_global = ang_root + theta_thigh_l
        ang_leg_l_global = ang_thigh_l_global + theta_leg_l
        ang_foot_l_global = ang_leg_l_global + theta_foot_l

        
        p_root = np.array([x_pos, root_z])
        
        dx, dz = rotate(0, L_TORSO_UP, ang_root)
        p_head = p_root + np.array([dx, dz])
        
        dx, dz = rotate(0, -L_TORSO_DOWN, ang_root)
        p_hip = p_root + np.array([dx, dz])
        
        dx, dz = rotate(0, -L_THIGH, ang_thigh_r_global)
        p_knee_r = p_hip + np.array([dx, dz])
        
        dx, dz = rotate(0, -L_LEG, ang_leg_r_global)
        p_ankle_r = p_knee_r + np.array([dx, dz])
        
        dx, dz = rotate(L_FOOT, 0, ang_foot_r_global)
        p_toe_r = p_ankle_r + np.array([dx, dz])
        
        dx, dz = rotate(0, -L_THIGH, ang_thigh_l_global)
        p_knee_l = p_hip + np.array([dx, dz])
        
        dx, dz = rotate(0, -L_LEG, ang_leg_l_global)
        p_ankle_l = p_knee_l + np.array([dx, dz])
        
        dx, dz = rotate(L_FOOT, 0, ang_foot_l_global)
        p_toe_l = p_ankle_l + np.array([dx, dz])

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

        n_plot = min(len(traj_expand_list), max_plot)
        if n_plot == 0: return


        fig, axes = plt.subplots(n_plot, 1, figsize=(12, 4 * n_plot))
        if n_plot == 1: axes = [axes]

        stride = 5  

        for i in range(n_plot):
            ax = axes[i]
            traj = traj_expand_list[i]  # [T, 12]
            
            obs_seq = traj[:, :11]
            x_seq = traj[:, 11]
            
            min_x, max_x = np.min(x_seq), np.max(x_seq)
            ax.plot([min_x - 1, max_x + 1], [0, 0], color='black', linewidth=2) 

            T = len(traj)
            colors = cm.viridis(np.linspace(0, 1, T))

            for t in range(0, T, stride):
                lines = self._get_walker_skeleton(obs_seq[t], x_seq[t])
                
                lc = LineCollection(lines, colors=colors[t], linewidths=2, alpha=0.8)
                ax.add_collection(lc)
                
                ax.plot(x_seq[t], obs_seq[t, 0], marker='o', markersize=6, color=colors[t], alpha=1.0)

            if plot_height_limit:
                ax.axhline(y=self.height_limit, color='red', linestyle='--', linewidth=2, label='Height Limit')
                
                if i == 0: ax.legend(loc='upper right')


            ax.set_aspect('equal') 
            ax.set_xlabel('Position X (m)')
            ax.set_ylabel('Height Z (m)')
            ax.set_title(f"Trajectory {i+1} (Length: {max_x - min_x:.2f}m)")
            
            ax.set_ylim(-0.1, max(2.0, self.height_limit + 0.5))

            ax.set_xlim(min_x - 0.5, max_x + 0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
