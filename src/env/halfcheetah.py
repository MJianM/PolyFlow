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

        u0, u1 = traj_act[:, :, 0], traj_act[:, :, 1]
        u3, u4 = traj_act[:, :, 3], traj_act[:, :, 4]


        leg_flag = (u0 + u1 < self.leg_limit + eps) & \
            (u3 + u4 < self.leg_limit + eps)
        

        torsion_flag = (u0 - u3 < self.torsion_limit + eps) & \
            (u3 - u0 < self.torsion_limit + eps)

        flag = leg_flag & torsion_flag

        is_safe = np.all(flag, axis=1)

        return is_safe

    def _get_x_pos(self):
        return self.env.unwrapped.data.qpos[0]

    def rollout(self, policy, n_episodes, seed=42, 
                is_video: bool = False,         # 是否录制
                video_episodes: int = 1,        # 录制前几集
                video_path: str = "videos"      # 保存路径
                ):

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        

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
                
                obs, info = env_to_use.reset(seed=current_seed)
                
                x_pos = self._get_x_pos()
                obs_list.append(obs)
                obs_expand_list.append(np.concatenate([obs, [x_pos]]))

                while not done:
                    cond = {0: obs.reshape(1, obs_dim)}
                    action, _, _, _, _, _ = policy(cond, batch_size=1)
                    action = action.flatten()
                    
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
            if is_video:
                env_to_use.close()

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
            'act_traj_list': act_traj_list, 
        }

        return obs_traj_list, obs_expand_traj_list, ret_list, metrics


    def _get_halfcheetah_skeleton(self, obs, x_pos):


        root_z = obs[0]
        root_angle = obs[1]
        
        # Back Leg
        bthigh_ang = obs[2]
        bshin_ang = obs[3]
        bfoot_ang = obs[4]
        
        # Front Leg
        fthigh_ang = obs[5]
        fshin_ang = obs[6]
        ffoot_ang = obs[7]


        torso_len = 1.0  
        back_hip_offset = -0.5 
        front_hip_offset = 0.5 
        
        thigh_len = 0.45
        shin_len = 0.5
        foot_len = 0.45
        

        def get_delta(length, angle):
            return length * np.sin(angle), -length * np.cos(angle)

        # dx_torso = (torso_len/2) * cos(root_angle)
        # dy_torso = (torso_len/2) * sin(root_angle)
        dx_head = (torso_len / 2.0) * np.cos(root_angle)
        dy_head = (torso_len / 2.0) * np.sin(root_angle)
        
        torso_center = np.array([x_pos, root_z])
        head_pos = torso_center + np.array([dx_head, dy_head])
        butt_pos = torso_center - np.array([dx_head, dy_head])

        b_hip_pos = butt_pos 
        
        b_thigh_global = root_angle + bthigh_ang
        b_shin_global  = b_thigh_global + bshin_ang
        b_foot_global  = b_shin_global + bfoot_ang
        
        dx, dy = get_delta(thigh_len, b_thigh_global)
        b_knee_pos = b_hip_pos + np.array([dx, dy])
        
        dx, dy = get_delta(shin_len, b_shin_global)
        b_ankle_pos = b_knee_pos + np.array([dx, dy])
        
        dx, dy = get_delta(foot_len, b_foot_global)
        b_toe_pos = b_ankle_pos + np.array([dx, dy])


        f_hip_offset_vec = np.array([back_hip_offset + torso_len, 0]) 
        f_hip_pos = head_pos 
        
        f_thigh_global = root_angle + fthigh_ang
        f_shin_global  = f_thigh_global + fshin_ang
        f_foot_global  = f_shin_global + ffoot_ang

        dx, dy = get_delta(thigh_len, f_thigh_global)
        f_knee_pos = f_hip_pos + np.array([dx, dy])
        
        dx, dy = get_delta(shin_len, f_shin_global)
        f_ankle_pos = f_knee_pos + np.array([dx, dy])
        
        dx, dy = get_delta(foot_len, f_foot_global)
        f_toe_pos = f_ankle_pos + np.array([dx, dy])


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
                lines = self._get_halfcheetah_skeleton(obs_seq[t], x_seq[t])
                
                lc = LineCollection(lines, colors=colors[t], linewidths=2, alpha=0.8)
                ax.add_collection(lc)
                
                ax.plot(x_seq[t], obs_seq[t, 0], marker='o', markersize=6, color=colors[t], alpha=1.0)


            ax.set_aspect('equal') 
            ax.set_xlabel('Position X (m)')
            ax.set_ylabel('Height Z (m)')
            ax.set_title(f"Trajectory {i+1} (Length: {max_x - min_x:.2f}m)")
            
            ax.set_ylim(-0.1, max(2.0, 1.25 + 0.5))

            ax.set_xlim(min_x - 0.5, max_x + 0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
