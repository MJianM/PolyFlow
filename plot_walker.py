from plot import plot_hopper_height_vel, calculate_traj_metric
import numpy as np

def calculate_halfcheetah_traj_metric(
    traj_path_list, label_list, leg_limit=1.2, torsion_limit=0.8
):
    
    for idx, (filepath, label) in enumerate(zip(traj_path_list, label_list)):
    
        # A. Load Data
        npz_data = np.load(filepath, allow_pickle=True)
        
        trajectories = [] # list of (T, obs_dim)
        
        # Load from obs_traj_list / obs_expand_traj_list
        # Raw data might be object arrays
        raw_obs = npz_data['act_traj_list'] # 
        
        # Convert all valid trajectories to a standard list format
        for i in range(len(raw_obs)):
            try:
                t_obs = np.array(raw_obs[i]).astype(float)
                if len(t_obs) > 1:
                    trajectories.append(t_obs)
            except: continue

        max_violation_ratio = []
        violation_cnt_ratio = []
        for traj in trajectories:
            u0 = traj[:, 0]
            u1 = traj[:, 1]
            u3 = traj[:, 3]
            u4 = traj[:, 4]

            vio_mag_1 = max(np.max(u0 + u1 - leg_limit) / leg_limit, 0.0)
            vio_mag_2 = max(np.max(u3 + u4 - leg_limit) / leg_limit, 0.0)
            vio_mag_3 = max(np.max(np.abs(u0 - u3) - torsion_limit) / torsion_limit, 0.0)
            vio_mag = max(vio_mag_1, vio_mag_2, vio_mag_3)

            flag = (u0 + u1 > leg_limit) | (u3 + u4 > leg_limit) | (np.abs(u0 - u3) > torsion_limit)
            vio_ratio = np.sum(flag) / u0.shape[0]

            max_violation_ratio.append(vio_mag)
            violation_cnt_ratio.append(vio_ratio)
        
        max_violation_ratio = np.array(max_violation_ratio)
        violation_cnt_ratio = np.array(violation_cnt_ratio)

        print(f"Label {label}")
        print(f"max_violation_ratio: {np.max(max_violation_ratio)} \n {max_violation_ratio}")
        print(f"violation cnt ratio: {np.max(violation_cnt_ratio)} \n {violation_cnt_ratio}")


if __name__=='__main__':

    #%% walker2d-hard
    # plot_hopper_height_vel(
    #     traj_path_list=[
    #         "outputs/walker2dcpx2/flow_sample_step100/42_2026-01-08_14-32-32/final_traj.npz",
    #         "outputs/walker2dcpx2/polyflow_sample/42_2026-01-08_18-27-38/final_traj.npz",
    #         "outputs/walker2dcpx2/safeflow_sample_step100/42_2026-01-08_18-30-27/final_traj.npz",
    #         "outputs/walker2dcpx2/RoS_sample_horizon160/42_2026-01-08_14-52-18/final_traj.npz",
    #         "outputs/walker2dcpx2/gaugeflow_train_step100/42_2026-01-09_15-58-19/final_traj.npz"
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", 'RoSD', 'GaugeFlow'], 
    #     height_limit=1.35, 
    #     vel_scale=0.01,
    #     height_min=0.9,
    #     v_max=1.4,
    #     v_min=-1.4,
    #     plot_horizon_length=100, 
    #     save_path="walkerhard_height_vel.png",
    #     select='all',
    #     env='walker2d'
    # )

    # calculate_traj_metric(
    #     traj_path_list=[
    #         "outputs/walker2dcpx2/flow_sample_step100/42_2026-01-08_14-32-32/final_traj.npz",
    #         "outputs/walker2dcpx2/polyflow_sample/42_2026-01-08_18-27-38/final_traj.npz",
    #         "outputs/walker2dcpx2/safeflow_sample_step100/42_2026-01-08_18-30-27/final_traj.npz",
    #         "outputs/walker2dcpx2/RoS_sample_horizon160/42_2026-01-08_14-52-18/final_traj.npz",
    #         "outputs/walker2dcpx2/gaugeflow_train_step100/42_2026-01-09_15-58-19/final_traj.npz"
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", 'RoSD', 'GaugeFlow'], 
    #     height_limit=1.35,
    #     vel_scale=0.01,
    #     obs_v_idx=9,
    #     v_limit=1.4,
    #     height_min=0.9  
    # )

    #%% walker2d-simple
    # plot_hopper_height_vel(
    #     traj_path_list=[
    #         "outputs/walker2dcpx/flow_sample_step100/42_2026-01-08_23-53-54/final_traj.npz",
    #         "outputs/walker2dcpx/polyflow_sample/42_2026-01-09_00-35-45/final_traj.npz",
    #         "outputs/walker2dcpx/safeflow_sample_step100/42_2026-01-09_13-20-46/final_traj.npz",
    #         "outputs/walker2dcpx/RoS_sample_horizon160/42_2026-01-09_13-19-30/final_traj.npz",
    #         "outputs/walker2dcpx/gaugeflow_train_step100/42_2026-01-09_15-57-56/final_traj.npz"
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", 'RoSD', 'GaugeFlow'], 
    #     height_limit=1.35, 
    #     vel_scale=0.01,
    #     height_min=None,
    #     v_max=None,
    #     v_min=None,
    #     plot_horizon_length=100, 
    #     save_path="walkersimple_height_vel.png",
    #     select='all',
    #     env='walker2d'
    # )

    # calculate_traj_metric(
    #     traj_path_list=[
    #         "outputs/walker2dcpx/flow_sample_step100/42_2026-01-08_23-53-54/final_traj.npz",
    #         "outputs/walker2dcpx/polyflow_sample/42_2026-01-09_00-35-45/final_traj.npz",
    #         "outputs/walker2dcpx/safeflow_sample_step100/42_2026-01-09_13-20-46/final_traj.npz",
    #         "outputs/walker2dcpx/RoS_sample_horizon160/42_2026-01-09_13-19-30/final_traj.npz",
    #         "outputs/walker2dcpx/gaugeflow_train_step100/42_2026-01-09_15-57-56/final_traj.npz"
    #     ], 
    #     label_list=["Flow", "PolyFlow(Ours)", "SafeFlow", 'RoSD', 'GaugeFlow'], 
    #     height_limit=1.35,
    #     vel_scale=0.01,
    #     obs_v_idx=9,
    #     v_limit=None,
    #     height_min=None  
    # )

    #%% hopper-simple
    # calculate_traj_metric(
    #     traj_path_list=[
    #         'outputs/hoppercpx/gaugeflow_sample_step200/42_2026-01-10_16-36-43/final_traj.npz',
    #     ],
    #     label_list=['GaugeFlow'],
    #     height_limit=1.6,
    #     vel_scale=0.01,
    #     obs_v_idx=6,
    #     v_limit=None,
    #     height_min=None
    # )

    #%%
    # calculate_traj_metric(
    #     traj_path_list=[
    #         'outputs/hoppercpx2/gaugeflow_sample_step200/42_2026-01-10_16-37-00/final_traj.npz',
    #     ],
    #     label_list=['GaugeFlow'],
    #     height_limit=1.5,
    #     vel_scale=0.01,
    #     obs_v_idx=6,
    #     v_limit=2.5,
    #     height_min=0.8,
    # )

    #%%
    # calculate_halfcheetah_traj_metric(
    #     traj_path_list=[
    #         'outputs/halfcheetah/flow_sample_step100/42_2026-01-08_18-03-03/final_traj.npz',
    #         'outputs/halfcheetah/polyflow_sample/42_2026-01-08_18-05-28/final_traj.npz',
    #         'outputs/halfcheetah/safeflow_sample_step100/42_2026-01-10_15-38-47/final_traj.npz',
    #         'outputs/halfcheetah/RoS_sample_horizon160/42_2026-01-09_13-14-40/final_traj.npz',
    #         'outputs/halfcheetah/gaugeflow_train_step100/42_2026-01-09_15-56-16/final_traj.npz'
    #     ],
    #     label_list=[
    #         'Flow', 'PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow'
    #     ],
    #     leg_limit=1.2,
    #     torsion_limit=0.8
    # )

    calculate_halfcheetah_traj_metric(
        traj_path_list=[
            'outputs/halfcheetah/polyflow_sample/42_2026-01-08_18-05-28/final_traj.npz',
            'outputs/halfcheetah/polyflow_train_softmin80/42_2026-01-12_19-50-17/final_traj.npz',
            'outputs/halfcheetah/polyflow_train_boltzmann80/42_2026-01-12_19-56-07/final_traj.npz',
        ],
        label_list=[
            'Hard', 'Softmin', 'Boltzmann'
        ],
        leg_limit=1.2,
        torsion_limit=0.8
    )