import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from physicsnemo.models.fno import FNO

# ==========================================
# 1. Physics & Integration Logic
# ==========================================
class LCMOptimizer:
    def __init__(self, checkpoint_dir, alpha=1.0, beta=1.0, gate_length_nm=30.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.gate_len_cm = gate_length_nm * 1e-7
        
        # Load Stats & Model
        stats_path = os.path.join(checkpoint_dir, "stats.pt")
        model_path = os.path.join(checkpoint_dir, "best_model.pth")
        
        try:
            self.stats = torch.load(stats_path, weights_only=False)
            self.model_state = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            self.stats = torch.load(stats_path)
            self.model_state = torch.load(model_path, map_location=self.device)

        self.model = FNO(in_channels=6, out_channels=1, decoder_layers=1, 
                         decoder_layer_size=32, dimension=2, latent_channels=32, 
                         num_fno_layers=4, num_fno_modes=12, padding=8).to(self.device)
        self.model.load_state_dict(self.model_state)
        self.model.eval()
        
        # Pre-compute normalized coordinate grids (H, W) = (64, 128)
        self.H, self.W = 64, 128
        self.grid_x_norm, self.grid_r_norm = np.meshgrid(
            np.linspace(-1, 1, self.W), np.linspace(0, 1, self.H)
        )
        self.grid_x_norm = torch.from_numpy(self.grid_x_norm).float().to(self.device)
        self.grid_r_norm = torch.from_numpy(self.grid_r_norm).float().to(self.device)

    def predict_and_loss(self, x):
        spacing, tl, pgm = x
        
        # --- Batch Prediction (t=0 and t=10000) ---
        # Params T=0
        log_time_0 = 0.0
        p0 = np.array([spacing, tl, pgm, log_time_0])
        p0_norm = (p0 - self.stats['p_mean']) / self.stats['p_std']
        
        # Params T=10000
        log_time_t = np.log10(10000.0)
        pt = np.array([spacing, tl, pgm, log_time_t])
        pt_norm = (pt - self.stats['p_mean']) / self.stats['p_std']
        
        # Prepare Batch Input: (2, 6, H, W)
        batch_input = torch.zeros((2, 6, self.H, self.W), device=self.device)
        
        # Broadcast scalar params
        for i, p_norm in enumerate([p0_norm, pt_norm]):
            for c in range(4):
                batch_input[i, c, :, :] = float(p_norm[c])
            batch_input[i, 4, :, :] = self.grid_x_norm
            batch_input[i, 5, :, :] = self.grid_r_norm
            
        with torch.no_grad():
            preds = self.model(batch_input) # Output (2, 1, H, W)
            
        # Denormalize
        preds_real = (preds.squeeze().cpu().numpy() * self.stats['dcd_std']) + self.stats['dcd_mean']
        map_0, map_t = preds_real[0], preds_real[1]
        
        # --- Integration ---
        # Physical Grids [cm]
        x_half_cm = (4.0 * spacing) * 1e-7
        r_start_cm = (tl * 1e-8)
        r_width_cm = 13.0 * 1e-7
        
        # Differential elements
        dx = (2 * x_half_cm) / self.W
        dr = r_width_cm / self.H
        
        # R vector for volume element (H,)
        r_vec = np.linspace(r_start_cm, r_start_cm + r_width_cm, self.H)
        # Volume factor per row: 2 * pi * r * dr * dx
        # Shape (H, 1) to broadcast across W
        vol_factor = (2 * np.pi * r_vec * dr * dx)[:, None]
        
        # Zone Masks (Indices)
        x_vec = np.linspace(-x_half_cm, x_half_cm, self.W)
        cc_mask = (x_vec >= -self.gate_len_cm/2) & (x_vec <= self.gate_len_cm/2)
        lc_mask = (x_vec < -self.gate_len_cm/2)
        rc_mask = (x_vec > self.gate_len_cm/2)
        
        # Helper to sum charge
        def get_q(map_arr, mask):
            # Sum( Rho * VolFactor ) only where mask is True
            # map_arr is (H, W), vol_factor is (H, 1), mask is (W,)
            return np.sum(map_arr[:, mask] * vol_factor)

        # Compute Delta Q
        dq_cc = (get_q(map_t, cc_mask) - get_q(map_0, cc_mask))/spacing
        dq_lc = (get_q(map_t, lc_mask) - get_q(map_0, lc_mask))/spacing
        dq_rc = (get_q(map_t, rc_mask) - get_q(map_0, rc_mask))/spacing
        
        # Objective
        loss = self.alpha * np.abs(dq_cc) + self.beta * (np.abs(dq_lc) + np.abs(dq_rc))
        return loss

# ==========================================
# 2. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    optimizer_engine = LCMOptimizer(args.checkpoint_dir)
    
    # Storage for all candidates
    # Columns: [Spacing, TL, PGM, Loss, Iteration]
    history = []
    
    # Bounds
    bounds = [(20.0, 30.0), (20.0, 40.0), (10.0, 20.0)]
    
    # --- Wrapper to log every call ---
    # Differential Evolution calls this function for every candidate
    def obj_wrapper(x):
        loss = optimizer_engine.predict_and_loss(x)
        # We append a placeholder iteration 0; we will fix iterations via callback
        history.append([x[0], x[1], x[2], loss])
        return loss
    
    # --- Callback to print progress ---
    # Called after each population evolves
    current_iter = [0]
    def callback(xk, convergence):
        current_iter[0] += 1
        # Calculate loss for best candidate xk
        best_loss = optimizer_engine.predict_and_loss(xk)
        print(f"Step {current_iter[0]}: Best F={best_loss:.4e} | Params: S={xk[0]:.2f}, TL={xk[1]:.2f}, PGM={xk[2]:.2f}")
    
    print("Running optimization with history logging...")
    
    result = differential_evolution(
        obj_wrapper,
        bounds,
        strategy='best1bin',
        maxiter=30,
        popsize=10, 
        callback=callback,
        tol=0.01,
        disp=False
    )
    
    # ==========================================
    # 3. Visualization
    # ==========================================
    df = pd.DataFrame(history, columns=["Spacing", "TL", "PGM", "Loss"])
    
    # Sort by Loss to see the "best" candidates clearly
    df_sorted = df.sort_values("Loss")
    best_candidates = df_sorted.head(100) # Top 100 points tried
    
    print("\nTop 5 Candidates Found:")
    print(df_sorted.head(5))
    
    # --- Plot 1: Parameter Correlation with Loss ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Spacing vs Loss
    sc1 = axes[0].scatter(df["Spacing"], df["Loss"], c=df["Loss"], cmap='viridis_r', alpha=0.6)
    axes[0].set_xlabel("Spacing (nm)")
    axes[0].set_ylabel("Objective F (Coulombs)")
    axes[0].set_title("Spacing vs. Minimal F")
    axes[0].grid(True, alpha=0.3)
    
    # TL vs Loss
    sc2 = axes[1].scatter(df["TL"], df["Loss"], c=df["Loss"], cmap='viridis_r', alpha=0.6)
    axes[1].set_xlabel("Tunnel Thickness (A)")
    axes[1].set_title("Thickness vs. Minimal F")
    axes[1].grid(True, alpha=0.3)
    
    # PGM vs Loss
    sc3 = axes[2].scatter(df["PGM"], df["Loss"], c=df["Loss"], cmap='viridis_r', alpha=0.6)
    axes[2].set_xlabel("Program Voltage (V)")
    axes[2].set_title("PGM vs. Minimal F")
    axes[2].grid(True, alpha=0.3)
    
    cbar = plt.colorbar(sc3, ax=axes.ravel().tolist())
    cbar.set_label('Composite Objective F')
    
    plt.suptitle("Optimization Search Landscape: Trends for Minimal LCM", fontsize=16)
    plt.savefig("optimization_trends.png")
    print("Saved trend plot to optimization_trends.png")
    
    # --- Plot 2: 3D Best Trajectory ---
    # Plot the parameters of the top 10% best solutions to see where they cluster
    cutoff = df["Loss"].quantile(0.10)
    best_cluster = df[df["Loss"] < cutoff]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    p = ax.scatter(best_cluster["Spacing"], best_cluster["TL"], best_cluster["PGM"], 
                   c=best_cluster["Loss"], cmap='jet', s=50, depthshade=True)
    
    ax.set_xlabel('Spacing (nm)')
    ax.set_ylabel('Thickness (A)')
    ax.set_zlabel('PGM Voltage (V)')
    ax.set_title(f'Cluster of Optimal Parameters (Top 10%)')
    
    fig.colorbar(p, label='Objective F')
    plt.savefig("optimization_cluster_3d.png")
    print("Saved 3D cluster plot to optimization_cluster_3d.png")

if __name__ == "__main__":
    main()