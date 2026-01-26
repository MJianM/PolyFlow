import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# x_min, x_max, y_min, y_max
corridor_1 = [0.0, 3.2, 0.8, 1.2]
corridor_2 = [2.8, 3.2, 0.8, 4.5]
corridors = [corridor_1, corridor_2]

class LMazeEnv:
    def __init__(self):
        
        
        # [x_min, x_max, y_min, y_max]
        self.corridors_list = [
            [0.0, 3.2, 0.8, 1.2], # Corridor 1
            [2.8, 3.2, 0.8, 4.5]  # Corridor 2
        ]
        self.corridors_tensor = torch.tensor(self.corridors_list)

    def safety_check(self, trajectories):
        safe_list = []
        for i in range(trajectories.shape[0]):
            is_safe = True
            points_safe_list = []
            for j in range(trajectories.shape[1]):
                x, y = trajectories[i, j]
                point_safe = False
                for c in corridors:
                    if c[0] <= x <= c[1] and c[2] <= y <= c[3]:
                        point_safe = True
                        break
                points_safe_list.append(point_safe)
                if not point_safe:
                    is_safe = False
                    break
            safe_list.append(is_safe)
        return safe_list

    def Shield(self, x, x_new, t):
        device = x_new.device
        corridors = self.corridors_tensor.to(device) # (N_corr, 4)

        # x_new: (B, S, 2) -> (B, S, 1, 2) 
        point = x_new.unsqueeze(2) 
        
        #  (1, 1, N_corr, 1)
        x_min = corridors[:, 0].view(1, 1, -1, 1)
        x_max = corridors[:, 1].view(1, 1, -1, 1)
        y_min = corridors[:, 2].view(1, 1, -1, 1)
        y_max = corridors[:, 3].view(1, 1, -1, 1)
        
        # proj_x shape: (B, S, N_corr, 1)
        proj_x = torch.clamp(point[..., 0:1], min=x_min, max=x_max)
        proj_y = torch.clamp(point[..., 1:2], min=y_min, max=y_max)
        
        # (B, S, N_corr, 2)
        candidates = torch.cat([proj_x, proj_y], dim=-1)
        
        # dists: (B, S, N_corr)
        dists_sq = torch.sum((candidates - point) ** 2, dim=-1)
        
        # min_indices: (B, S)
        min_indices = torch.argmin(dists_sq, dim=-1)
        
        # candidates: (B, S, 1, 2)
        min_indices_expanded = min_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        
        x_proj = torch.gather(candidates, 2, min_indices_expanded).squeeze(2)
        
        return x_proj
    
    def GD(self, x, x_new, t, step_size=0.1, num_steps=1):
        device = x_new.device
        corridors = self.corridors_tensor.to(device)
        
        x_opt = x_new.detach().clone().requires_grad_(True)
        
        with torch.enable_grad():
            for _ in range(num_steps):
                point = x_opt.unsqueeze(2) # (B, S, 1, 2)
                
                x_min = corridors[:, 0].view(1, 1, -1, 1)
                x_max = corridors[:, 1].view(1, 1, -1, 1)
                y_min = corridors[:, 2].view(1, 1, -1, 1)
                y_max = corridors[:, 3].view(1, 1, -1, 1)
                
                d_x = torch.maximum(x_min - point[..., 0:1], torch.tensor(0., device=device)) + \
                      torch.maximum(point[..., 0:1] - x_max, torch.tensor(0., device=device))
                d_y = torch.maximum(y_min - point[..., 1:2], torch.tensor(0., device=device)) + \
                      torch.maximum(point[..., 1:2] - y_max, torch.tensor(0., device=device))
                
                dist_sq_per_corridor = d_x**2 + d_y**2
                
                min_dist_sq, _ = torch.min(dist_sq_per_corridor, dim=-1)
                
                loss = min_dist_sq.sum()
                
                if loss.item() < 1e-6:
                    break 
                
                grad = torch.autograd.grad(loss, x_opt)[0]
                
                with torch.no_grad():
                    x_opt = x_opt - step_size * grad
                    x_opt.requires_grad_(True)
                
        return x_opt.detach()

    def plot_trajectory_comparison(self, true_trajs, gene_trajs, plot_ellips=False, max_plot=100, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, corridor in enumerate(corridors):
            x_min, x_max, y_min, y_max = corridor
            width = x_max - x_min
            height = y_max - y_min
            
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='black', facecolor='none',
                linestyle='--', alpha=0.7, label=f'Corridor {i+1}' if i == 0 else ""
            )
            ax.add_patch(rect)

        true_traj = np.array(true_trajs).reshape(true_trajs.shape[0], -1)
        gene_traj = np.array(gene_trajs).reshape(gene_trajs.shape[0], -1)
        n_true = min(len(true_trajs), max_plot)
        n_gene = min(len(gene_trajs), max_plot)
        true_traj = true_traj[:n_true]
        gene_traj = gene_traj[:n_gene]

        if len(true_traj.shape) == 1:
            true_traj = true_traj.reshape(1, -1)
        if len(gene_traj.shape) == 1:
            gene_traj = gene_traj.reshape(1, -1)
        
        n_samples_true = true_traj.shape[0]
        n_samples_gene = gene_traj.shape[0]
        
        for i in range(n_samples_true):
            traj = true_traj[i]
            T = len(traj) // 2
            points = traj.reshape(T, 2)
            
            x = points[:, 0]
            y = points[:, 1]
            
            ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7, 
                    label='True Trajectory' if i == 0 else "")
            ax.scatter(x, y, c='blue', s=20, alpha=0.6)
            
            ax.scatter(x[0], y[0], c='green', s=100, marker='o', 
                    edgecolors='black', linewidth=2, label='Start' if i == 0 else "")
            ax.scatter(x[-1], y[-1], c='red', s=100, marker='s', 
                    edgecolors='black', linewidth=2, label='End' if i == 0 else "")
        
        for i in range(n_samples_gene):
            traj = gene_traj[i]
            T = len(traj) // 2
            points = traj.reshape(T, 2)
            
            x = points[:, 0]
            y = points[:, 1]
            
            ax.plot(x, y, 'r-', linewidth=1.5, alpha=0.7,
                    label='Generated Trajectory' if i == 0 else "")
            ax.scatter(x, y, c='red', s=20, alpha=0.6)
            
            ax.scatter(x[0], y[0], c='green', s=100, marker='o', 
                    edgecolors='black', linewidth=2)
            ax.scatter(x[-1], y[-1], c='red', s=100, marker='s', 
                    edgecolors='black', linewidth=2)
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title('Trajectory Comparison in L-shaped Corridor', fontsize=14, fontweight='bold')
        
        ax.set_xlim(-0.5, 4.0)
        ax.set_ylim(-0.5, 5.0)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
        
        ax.set_aspect('equal', adjustable='box')
        
        plt.figtext(0.02, 0.98, f'True samples: {n_samples_true}', 
                    fontsize=10, verticalalignment='top')
        plt.figtext(0.02, 0.95, f'Generated samples: {n_samples_gene}', 
                    fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
       
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()