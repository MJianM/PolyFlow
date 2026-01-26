import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cvxpy as cp
import math
import random

NUM_TRAJ = 100         
W_SMOOTH = 1.0        
W_RANDOM = 0.2       
W_OVERLAP = 10.0     
np.random.seed(42)
SAVE_PATH = 'L_maze_traj_data.npz'

def build_constraint_matrices(T, mean_begin, mean_end, c1, c2, overlap):
    """
    param:
        T: int, 
        mean_begin: int, 
        mean_end: int, 
        c1: list, corridor_1 [xmin, xmax, ymin, ymax]
        c2: list, corridor_2 [xmin, xmax, ymin, ymax]
        overlap: list, overlap_rect [xmin, xmax, ymin, ymax]
        
    return:
        A: np.ndarray, (T, 4, 2)
        b: np.ndarray, (T, 4)
    """
    
    # bounds shape: (T, 4) -> [xmin, xmax, ymin, ymax]
    bounds = np.zeros((T, 4))
    
    # corridor_1
    bounds[0:mean_begin] = c1
    # overlap
    bounds[mean_begin : mean_end + 1] = overlap
    # corridor_2
    bounds[mean_end + 1:] = c2

    # -x <= -xmin, x <= xmax, -y <= -ymin, y <= ymax
    # b_blocks (T, 4)
    b = np.stack([
        -bounds[:, 0],  # -x_min
        bounds[:, 1],   # x_max
        -bounds[:, 2],  # -y_min
        bounds[:, 3]    # y_max
    ], axis=1)
    
    # b = b_blocks.flatten() 

    # [-1,  0] -> -x
    # [ 1,  0] ->  x
    # [ 0, -1] -> -y
    # [ 0,  1] ->  y
    block = np.array([
        [-1,  0], 
        [ 1,  0], 
        [ 0, -1], 
        [ 0,  1]
    ]) # (4, 2)
    
    A = np.tile(block[np.newaxis, :, :], (T, 1, 1))
    
    return A, b

def bezier_curve(control_points, t):
    n = len(control_points) - 1
    result = np.zeros(2)
    for i in range(n + 1):
        comb = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
        term = comb * ((1 - t) ** (n - i)) * (t ** i)
        result += term * control_points[i]
    return result

def get_overlap_rect(c1, c2):
    x_min = max(c1[0], c2[0])
    x_max = min(c1[1], c2[1])
    y_min = max(c1[2], c2[2])
    y_max = min(c1[3], c2[3])
    if x_min >= x_max or y_min >= y_max:
        return None
    return [x_min, x_max, y_min, y_max]

corridor_1 = [0.0, 3.2, 0.8, 1.2]
corridor_2 = [2.8, 3.2, 0.8, 4.5]
corridors = [corridor_1, corridor_2]
n_segments = len(corridors)

n_order = 5
n_points = n_order + 1

all_trajectories_P = []

print(f"Generating {NUM_TRAJ} trajectories...")

for traj_idx in range(NUM_TRAJ):
    P = cp.Variable((n_segments, n_points, 2))
    constraints = []
    
    P_random_targets = np.zeros((n_segments, n_points, 2))
    for i in range(n_segments):
        c = corridors[i]
        P_random_targets[i, :, 0] = np.random.uniform(c[0], c[1], n_points)
        P_random_targets[i, :, 1] = np.random.uniform(c[2], c[3], n_points)
        
    overlap_targets = [] 
    for i in range(n_segments - 1):
        overlap_rect = get_overlap_rect(corridors[i], corridors[i+1])
        if overlap_rect:
            rand_x = np.random.uniform(overlap_rect[0], overlap_rect[1])
            rand_y = np.random.uniform(overlap_rect[2], overlap_rect[3])
            overlap_targets.append(np.array([rand_x, rand_y]))
        else:
            overlap_targets.append(None)

    for i in range(n_segments):
        c = corridors[i]
        constraints += [
            P[i, :, 0] >= c[0], P[i, :, 0] <= c[1],
            P[i, :, 1] >= c[2], P[i, :, 1] <= c[3]
        ]
    constraints += [P[0, 0, :] == np.array([0.5, 1.0])] 
    constraints += [P[-1, -1, :] == np.array([3.0, 4.0])]
    for i in range(n_segments - 1):
        constraints += [P[i, -1, :] == P[i+1, 0, :]]
        constraints += [P[i, -1, :] - P[i, -2, :] == P[i+1, 1, :] - P[i+1, 0, :]]

    smoothness_cost = 0
    random_attraction_cost = 0
    overlap_attraction_cost = 0

    for i in range(n_segments):
        for k in range(n_points - 1):
            smoothness_cost += cp.sum_squares(P[i, k+1, :] - P[i, k, :])
        for k in range(n_points):
            random_attraction_cost += cp.sum_squares(P[i, k, :] - P_random_targets[i, k, :])
            
    for i in range(n_segments - 1):
        if overlap_targets[i] is not None:
            overlap_attraction_cost += cp.sum_squares(P[i, -1, :] - overlap_targets[i])

    cost = (W_SMOOTH * smoothness_cost + 
            W_RANDOM * random_attraction_cost + 
            W_OVERLAP * overlap_attraction_cost)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    if prob.status == 'optimal':
        all_trajectories_P.append(P.value)
    else:
        print(f"Trajectory {traj_idx} failed.")

traj_list = []
for idx, P_val in enumerate(all_trajectories_P):
    whole_traj = []
    for i in range(n_segments):
        for t in np.linspace(0, 1, 25):
            whole_traj.append(bezier_curve(P_val[i], t))
    traj_list.append(np.array(whole_traj).reshape(1, 50, 2))

traj_dataset = np.concatenate(traj_list, axis=0) # (num_traj, 100, 2)

overlap_rect = get_overlap_rect(corridors[0], corridors[1])
mean_begin, mean_end = -1, -1
if overlap_rect is not None:
    x_coords = traj_dataset[:, :, 0]
    y_coords = traj_dataset[:, :, 1]
    overlap_flag = (x_coords >= overlap_rect[0]) & (x_coords <= overlap_rect[1]) & \
                   (y_coords >= overlap_rect[2]) & (y_coords <= overlap_rect[3])

    min_list = []
    max_list = []
    for k in range(traj_dataset.shape[0]):
        cur = overlap_flag[k]
        idx_list = np.argwhere(cur)
        if len(idx_list) > 0:
            min_list.append(idx_list[0][0])
            max_list.append(idx_list[-1][0])

    if len(min_list) > 0:
        max_begin = int(np.max(np.array(min_list)))
        min_end = int(np.min(np.array(max_list)))
        assert max_begin <= min_end, f"max begin {max_begin} > min end {min_end}"
        mean_begin = int(np.mean(np.array(min_list)))
        mean_end = int(np.mean(np.array(max_list)))
        print(f"Overlap Analysis -> Begin Step: {max_begin}, End Step: {min_end}")

single_A, single_b = build_constraint_matrices(
    traj_dataset.shape[1], 
    max_begin, 
    min_end, 
    c1=corridor_1, 
    c2=corridor_2, 
    overlap=overlap_rect
)

single_A = np.expand_dims(single_A, axis=0).repeat(NUM_TRAJ, axis=0)
single_b = np.expand_dims(single_b, axis=0).repeat(NUM_TRAJ, axis=0)

print(f"Constraint A shape: {single_A.shape}")
print(f"Constraint b shape: {single_b.shape}") 

np.savez(
    SAVE_PATH, 
    traj_dataset=traj_dataset,  # (num_traj, seq_length*2)
    mean_begin=max_begin, 
    mean_end=min_end,
    overlap_rect=overlap_rect,
    corridor_1=corridor_1,
    corridor_2=corridor_2,
    single_A=single_A, 
    single_b=single_b 
)
print(f"Data saved in: {SAVE_PATH}")

plt.figure(figsize=(8, 8))
for c in corridors:
    rect = patches.Rectangle((c[0], c[2]), c[1]-c[0], c[3]-c[2], 
                             linewidth=3, edgecolor='r', facecolor='none', linestyle='--', alpha=0.5)
    plt.gca().add_patch(rect)

for traj in traj_dataset[:10]:
    plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)

plt.title("Maze Trajectories (Verification)")
plt.axis('equal')
plt.savefig("L_maze_dataset_check.png")