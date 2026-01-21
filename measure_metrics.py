from plot import calculate_traj_metric
from plot_walker import calculate_halfcheetah_traj_metric


# # hoppercpx
# calculate_traj_metric(
#     traj_path_list=[
#         'outputs/hoppercpx/flow_time_step200/42_2026-01-17_18-28-10/final_traj.npz',
#         'outputs/hoppercpx/polyflow_time/42_2026-01-17_20-23-58/final_traj.npz',
#         'outputs/hoppercpx/safeflow_time_step200/42_2026-01-17_19-16-43/final_traj.npz',
#         'outputs/hoppercpx/RoS_time_horizon600/42_2026-01-17_17-33-39/final_traj.npz',
#         'outputs/hoppercpx/gaugeflow_time_step200/42_2026-01-19_01-18-11/final_traj.npz',
#         'outputs/hoppercpx/polyflow_time_fixcons/42_2026-01-20_12-56-05/final_traj.npz',
#         'outputs/hoppercpx/polyflow_time_fixcons_ot/42_2026-01-20_12-22-05/final_traj.npz'
        
#     ],
#     label_list=[
#         'Flow', 'PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow', 'PolyFlowFix', 'PolyFlowFixOT'
#     ],
#     height_limit=1.6,
#     vel_scale=0.01,
#     obs_v_idx=6,
#     v_limit=None,
#     height_min=0.7,
# )

# # hoppercpx2
# calculate_traj_metric(
#     traj_path_list=[
#         'outputs/hoppercpx2/flow_time_step200/42_2026-01-17_22-07-24/final_traj.npz',
#         'outputs/hoppercpx2/polyflow_time/42_2026-01-18_11-21-32/final_traj.npz',
#         'outputs/hoppercpx2/RoS_time/42_2026-01-17_21-37-15/final_traj.npz',
#         'outputs/hoppercpx2/gaugeflow_time_step200/42_2026-01-18_11-28-18/final_traj.npz',
#         'outputs/hoppercpx2/polyflow_time_fixcons/42_2026-01-20_13-24-48/final_traj.npz'
        
#     ],
#     label_list=[
#         'Flow', 'PolyFlow', 'RoSD', 'GaugeFlow', 'PolyFlowFix'
#     ],
#     height_limit=1.5,
#     vel_scale=0.01,
#     obs_v_idx=6,
#     v_limit=2.5,
#     height_min=0.8,
# )


# # walkercpx
# calculate_traj_metric(
#     traj_path_list=[
#         'outputs/walker2dcpx/flow_time_step100/42_2026-01-18_15-47-07/final_traj.npz',
#         'outputs/walker2dcpx/polyflow_time/42_2026-01-18_17-15-30/final_traj.npz',
#         'outputs/walker2dcpx/safeflow_time_step100/42_2026-01-18_16-26-18/final_traj.npz',
#         'outputs/walker2dcpx/RoS_time_horizon160/42_2026-01-18_12-45-09/final_traj.npz',
#         'outputs/walker2dcpx/gaugeflow_time_step100/42_2026-01-18_17-25-37/final_traj.npz',
#         'outputs/walker2dcpx/polyflow_time_fixcons/42_2026-01-20_13-56-24/final_traj.npz'
        
#     ],
#     label_list=[
#         'Flow', 'PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow', 'PolyFlowFix'
#     ],
#     height_limit=1.35,
#     vel_scale=0.01,
#     obs_v_idx=9,
#     v_limit=None,
#     height_min=0.8,
# )



# # walkercpx
# calculate_traj_metric(
#     traj_path_list=[
#         'outputs/walker2dcpx2/flow_time_step100/42_2026-01-18_19-39-41/final_traj.npz',
#         'outputs/walker2dcpx2/polyflow_time/42_2026-01-19_05-09-13/final_traj.npz',
#         'outputs/walker2dcpx2/safeflow_time_step100/42_2026-01-18_20-20-04/final_traj.npz',
#         'outputs/walker2dcpx2/RoS_time_horizon160/42_2026-01-18_18-15-23/final_traj.npz',
#         'outputs/walker2dcpx2/gaugeflow_time_step100/42_2026-01-19_05-19-08/final_traj.npz',
#         'outputs/walker2dcpx2/polyflow_time_fixcons/42_2026-01-18_21-14-30/final_traj.npz',
        
#     ],
#     label_list=[
#         'Flow', 'PolyFlow', 'SafeFlow', 'RoSD', 'GaugeFlow', 'PolyFlowFix'
#     ],
#     height_limit=1.35,
#     vel_scale=0.01,
#     obs_v_idx=9,
#     v_limit=1.4,
#     height_min=0.9,
# )


calculate_halfcheetah_traj_metric(
    traj_path_list=[
        # 'outputs/halfcheetah/flow_time_step100/42_2026-01-20_16-49-45/final_traj.npz',
        'outputs/halfcheetah/polyflow_time_boltzmann80/42_2026-01-21_12-35-03/final_traj.npz'
    ],
    label_list=[
        'Boltzmann',
    ],
    leg_limit=1.2,
    torsion_limit=0.8
)
