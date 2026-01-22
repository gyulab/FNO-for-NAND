import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from physicsnemo.models.fno import FNO

def generate_gif():
    # ... (Argparse similar to previous) ...
    # HARDCODED PARAMS for Blind Spot
    SPACING = 25.0
    TL = 25.0
    PGM = 15.0
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load stats/model (same try/except logic as inference.py)
    stats = torch.load("./checkpoints/stats.pt", weights_only=False) # Simplified for brevity
    model = FNO(in_channels=6, out_channels=1, decoder_layers=1, decoder_layer_size=32, 
                dimension=2, latent_channels=32, num_fno_layers=4, num_fno_modes=12, padding=8).to(device)
    model.load_state_dict(torch.load("./checkpoints/best_model.pth", map_location=device, weights_only=False))
    model.eval()

    # Time Sweep
    time_points = np.concatenate([[0], np.logspace(-3, 4, 49)])
    
    # Pre-calc Domain (Same logic as inference)
    H, W = 64, 128
    x_half = 4.0 * SPACING
    x_bounds = (-x_half*1e-9, x_half*1e-9)
    r_start = TL * 1e-10
    r_bounds = (r_start, r_start + 13e-9)
    
    # Grids
    grid_x_norm, grid_r_norm = np.meshgrid(np.linspace(-1, 1, W), np.linspace(0, 1, H))
    phys_x = np.linspace(x_bounds[0], x_bounds[1], W)
    phys_r = np.linspace(r_bounds[0], r_bounds[1], H)
    X, R = np.meshgrid(phys_x, phys_r)

    # Batch Prediction for Scale
    preds = []
    with torch.no_grad():
        for t in time_points:
            log_time = np.log10(t + 1e-6) if t > 0 else 0
            # Construct Input (1, 6, H, W)
            x_in = np.zeros((1, 6, H, W), dtype=np.float32)
            
            # Params
            p = np.array([SPACING, TL, PGM, log_time])
            p_norm = (p - stats['p_mean']) / stats['p_std']
            for c in range(4): x_in[0, c] = p_norm[c]
            
            # Coords
            x_in[0, 4] = grid_x_norm
            x_in[0, 5] = grid_r_norm
            
            out = model(torch.from_numpy(x_in).to(device))
            real = (out.cpu().numpy().squeeze() * stats['dcd_std']) + stats['dcd_mean']
            preds.append(real)

    # Fix Scale
    preds = np.array(preds)
    limit = np.max(np.abs(preds)) * 1.05
    vmin, vmax = -limit, limit
    levels = np.linspace(vmin, vmax, 100)
    
    # Render
    os.makedirs("temp_frames", exist_ok=True)
    frames = []
    for i, (t, val) in enumerate(zip(time_points, preds)):
        fig = plt.figure(figsize=(10,5))
        c = plt.contourf(X*1e9, R*1e9, val, levels=levels, cmap='seismic', vmin=vmin, vmax=vmax)
        plt.colorbar(c, label='Defect Charge Density')
        plt.title(f"Time: {t:.2e}s (S={SPACING}nm)")
        plt.xlabel("X (nm)")
        plt.ylabel("R (nm)")
        fname = f"temp_frames/frame_{i:03d}.png"
        plt.savefig(fname)
        plt.close()
        frames.append(fname)
        
    imageio.mimsave("sonos_evolution.gif", [imageio.imread(f) for f in frames], duration=0.1)
    shutil.rmtree("temp_frames")

if __name__ == "__main__":
    generate_gif()