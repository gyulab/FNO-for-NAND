import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from physicsnemo.models.fno import FNO

# ==========================================
# 1. Integration & Physics Logic
# ==========================================
def integrate_cylindrical_charge(dcd_map, x_grid, r_grid, x_bounds):
    """
    Integrates charge density over a cylindrical volume.
    Q = integral( rho(x,r) * 2*pi*r * dr * dx )
    
    Args:
        dcd_map: (H, W) array of defect charge density [C/cm^3]
        x_grid: (H, W) array of X coordinates [cm]
        r_grid: (H, W) array of R coordinates [cm]
        x_bounds: Tuple (x_min, x_max) defining the region to integrate over [cm]
    """
    # Create mask for the region of interest (e.g., CC, LC, RC)
    mask = (x_grid >= x_bounds[0]) & (x_grid <= x_bounds[1])
    
    if not np.any(mask):
        return 0.0

    # Differential Elements (assuming uniform grid for simplicity)
    # dx = total_x_width / W
    # dr = total_r_height / H
    # Note: dcd_map is HxW (rows=R, cols=X)
    
    H, W = dcd_map.shape
    dx = np.abs(x_grid[0, 1] - x_grid[0, 0])
    dr = np.abs(r_grid[1, 0] - r_grid[0, 0])
    
    # Volume Element dV = 2 * pi * r * dr * dx
    # We use r_grid for 'r'
    dV = 2 * np.pi * r_grid * dr * dx
    
    # Total Charge in region
    # Q = sum( rho * dV * mask )
    total_charge = np.sum(dcd_map * dV * mask)
    
    return total_charge

class LCMOptimizer:
    def __init__(self, checkpoint_dir, alpha=1.0, beta=1.0, gate_length_nm=30.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.gate_len_cm = gate_length_nm * 1e-7 # nm to cm
        
        # Load Stats
        stats_path = os.path.join(checkpoint_dir, "stats.pt")
        try:
            self.stats = torch.load(stats_path, weights_only=False)
        except TypeError:
            self.stats = torch.load(stats_path)
            
        # Load Model
        self.model = FNO(in_channels=6, out_channels=1, decoder_layers=1, 
                         decoder_layer_size=32, dimension=2, latent_channels=32, 
                         num_fno_layers=4, num_fno_modes=12, padding=8).to(self.device)
        
        model_path = os.path.join(checkpoint_dir, "best_model.pth")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        except TypeError:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Pre-allocate grid tensors for speed
        self.H, self.W = 64, 128
        self.grid_x_norm, self.grid_r_norm = np.meshgrid(
            np.linspace(-1, 1, self.W), np.linspace(0, 1, self.H)
        )

    def predict_map(self, spacing, tl, pgm, time):
        """Runs FNO Inference for a single parameter set."""
        # 1. Normalize Inputs
        log_time = np.log10(time + 1e-6) if time > 0 else 0
        raw_params = np.array([spacing, tl, pgm, log_time])
        norm_params = (raw_params - self.stats['p_mean']) / self.stats['p_std']
        
        # 2. Build Input Tensor
        x_input = np.zeros((1, 6, self.H, self.W), dtype=np.float32)
        for c in range(4):
            x_input[0, c, :, :] = norm_params[c]
            
        x_input[0, 4, :, :] = self.grid_x_norm
        x_input[0, 5, :, :] = self.grid_r_norm
        
        x_tensor = torch.from_numpy(x_input).to(self.device)
        
        # 3. Predict & Denormalize
        with torch.no_grad():
            pred = self.model(x_tensor)
        
        pred_real = (pred.cpu().numpy().squeeze() * self.stats['dcd_std']) + self.stats['dcd_mean']
        return pred_real

    def objective_function(self, x):
        """
        The Cost Function to Minimize.
        x = [Spacing (nm), TL (A), PGM (V)]
        """
        spacing, tl, pgm = x
        
        # 1. Predict Maps at t=0 and t=10000
        map_0 = self.predict_map(spacing, tl, pgm, time=0.0)
        map_t = self.predict_map(spacing, tl, pgm, time=10000.0)
        
        # 2. Define Physical Grid (Dynamic based on Spacing/TL)
        # X Range: +/- 4.0 * Spacing (nm -> cm)
        x_half_cm = (4.0 * spacing) * 1e-7
        # R Range: TL to TL+13nm (A -> cm)
        r_start_cm = (tl * 1e-8)
        r_width_cm = 13.0 * 1e-7
        
        # Create Physical Meshgrid [cm] for integration
        phys_x = np.linspace(-x_half_cm, x_half_cm, self.W)
        phys_r = np.linspace(r_start_cm, r_start_cm + r_width_cm, self.H)
        grid_x_cm, grid_r_cm = np.meshgrid(phys_x, phys_r)
        
        # 3. Define Zones Bounds [cm]
        # CC: Center Cell (-Lg/2 to Lg/2)
        cc_bounds = (-self.gate_len_cm/2, self.gate_len_cm/2)
        # LC: Left Cell ( -Infinity to -Lg/2 )
        lc_bounds = (-999.0, -self.gate_len_cm/2)
        # RC: Right Cell ( Lg/2 to Infinity )
        rc_bounds = (self.gate_len_cm/2, 999.0)
        
        # 4. Integrate Charges
        # Q(0)
        q_cc_0 = integrate_cylindrical_charge(map_0, grid_x_cm, grid_r_cm, cc_bounds)
        q_lc_0 = integrate_cylindrical_charge(map_0, grid_x_cm, grid_r_cm, lc_bounds)
        q_rc_0 = integrate_cylindrical_charge(map_0, grid_x_cm, grid_r_cm, rc_bounds)
        
        # Q(t)
        q_cc_t = integrate_cylindrical_charge(map_t, grid_x_cm, grid_r_cm, cc_bounds)
        q_lc_t = integrate_cylindrical_charge(map_t, grid_x_cm, grid_r_cm, lc_bounds)
        q_rc_t = integrate_cylindrical_charge(map_t, grid_x_cm, grid_r_cm, rc_bounds)
        
        # 5. Compute Deltas (Retention Loss)
        dq_cc = q_cc_t - q_cc_0
        dq_lc = q_lc_t - q_lc_0
        dq_rc = q_rc_t - q_rc_0
        
        # 6. Composite Objective F
        # F = alpha * |dQ_CC| + beta * max(|dQ_LC|, |dQ_RC|)
        # We want to minimize the magnitude of charge loss/gain
        loss = self.alpha * np.abs(dq_cc) + self.beta * max(np.abs(dq_lc), np.abs(dq_rc))
        
        return loss

def main():
    parser = argparse.ArgumentParser(description="Optimize SONOS Parameters for Minimal LCM")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Center Cell Loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for Lateral Migration")
    parser.add_argument("--gate_len", type=float, default=30.0, help="Gate Length in nm (defines CC zone)")
    args = parser.parse_args()
    
    optimizer_engine = LCMOptimizer(args.checkpoint_dir, args.alpha, args.beta, args.gate_len)
    
    # ==========================================
    # Optimization Setup
    # ==========================================
    # Bounds based on training data range + interpolation safety
    # Spacing: 20nm to 30nm
    # TL: 20A to 40A
    # PGM: 10V to 20V
    bounds = [
        (20.0, 30.0), # Spacing
        (20.0, 40.0), # TL
        (10.0, 20.0)  # PGM
    ]
    
    print("==================================================")
    print("Starting Differential Evolution Optimization")
    print(f"Objective: Minimize F = {args.alpha}*|dQ_CC| + {args.beta}*max(|dQ_LC|,|dQ_RC|)")
    print(f"Time: 10,000s | Gate Length: {args.gate_len}nm")
    print(f"Search Space: Spacing[20-30nm], TL[20-40A], PGM[10-20V]")
    print("==================================================")

    # Run Differential Evolution
    result = differential_evolution(
        optimizer_engine.objective_function, 
        bounds, 
        strategy='best1bin', 
        maxiter=50, 
        popsize=15, 
        tol=0.01,
        mutation=(0.5, 1), 
        recombination=0.7,
        disp=True # Print convergence messages
    )
    
    print("\nOptimization Complete!")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print("--------------------------------------------------")
    print(f"Minimum Objective F: {result.fun:.4e} Coulombs")
    print("Optimal Parameters:")
    print(f"  Spacing : {result.x[0]:.4f} nm")
    print(f"  TL      : {result.x[1]:.4f} A")
    print(f"  PGM     : {result.x[2]:.4f} V")
    print("--------------------------------------------------")
    
    # ==========================================
    # Verification & Plotting
    # ==========================================
    print("Verifying optimal solution...")
    # Re-calculate final metrics for display
    opt_x = result.x
    opt_spacing, opt_tl, opt_pgm = opt_x
    
    # Hack to print the specific components
    # We call objective_function but we'd need to modify it to return components.
    # Instead, we just re-run the prediction code briefly here for plotting.
    map_t = optimizer_engine.predict_map(opt_spacing, opt_tl, opt_pgm, 10000.0)
    
    # Plot Optimal Map
    H, W = 64, 128
    x_half = 4.0 * opt_spacing
    tl_nm = opt_tl * 0.1
    
    phys_x = np.linspace(-x_half, x_half, W)
    phys_r = np.linspace(tl_nm, tl_nm+13.0, H)
    X, R = np.meshgrid(phys_x, phys_r)
    
    plt.figure(figsize=(10, 5))
    c = plt.contourf(X, R, map_t, levels=60, cmap='seismic')
    cbar = plt.colorbar(c)
    cbar.set_label('Defect Charge Density ($C \cdot cm^{-3}$)')
    
    # Draw Gate Boundaries
    plt.axvline(args.gate_len/2, color='black', linestyle='--', linewidth=1.5, label='Gate Edge')
    plt.axvline(-args.gate_len/2, color='black', linestyle='--', linewidth=1.5)
    
    plt.title(f"OPTIMAL CONFIGURATION\nF={result.fun:.2e} | S={opt_spacing:.2f}nm, TL={opt_tl:.2f}A, PGM={opt_pgm:.2f}V")
    plt.xlabel('Axial Position X (nm)')
    plt.ylabel('Radius R (nm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimal_lcm_solution.png")
    print("Saved optimal_lcm_solution.png")

if __name__ == "__main__":
    main()
