import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from physicsnemo.models.fno import FNO

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--tl", type=float, required=True)
    parser.add_argument("--pgm", type=float, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Resources
    try:
        stats = torch.load(os.path.join(args.checkpoint_dir, "stats.pt"), weights_only=False)
    except TypeError:
        stats = torch.load(os.path.join(args.checkpoint_dir, "stats.pt"))

    model = FNO(in_channels=6, out_channels=1, decoder_layers=1, decoder_layer_size=32, 
                dimension=2, latent_channels=32, num_fno_layers=4, num_fno_modes=12, padding=8).to(device)
    
    try:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"), map_location=device, weights_only=False))
    except TypeError:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"), map_location=device))
    model.eval()

    # 2. Construct Input
    H, W = 64, 128
    
    # --- PHYSICAL DOMAIN CALCULATION ---
    # Heuristic derived from data: Range is approx +/- 4*Spacing
    # e.g. Spacing=20 -> X=80. Spacing=25 -> X=100.
    x_half_width = 4.0 * args.spacing 
    x_bounds = (-x_half_width * 1e-9, x_half_width * 1e-9) # Convert nm to m

    # R bounds: Start at TL thickness. Assume active region width ~13nm (from 17-4 data)
    r_start = args.tl * 1e-10 # Angstrom to m
    r_width = 13e-9 
    r_bounds = (r_start, r_start + r_width)

    # Prepare Input Tensor
    log_time = np.log10(args.time + 1e-6) if args.time > 0 else 0
    raw_params = np.array([args.spacing, args.tl, args.pgm, log_time])
    norm_params = (raw_params - stats['p_mean']) / stats['p_std']

    x_input = np.zeros((1, 6, H, W), dtype=np.float32)
    
    # Broadcast params
    for c in range(4):
        x_input[0, c, :, :] = norm_params[c]
        
    # Add Coordinates (normalized [-1,1] and [0,1])
    grid_x_norm, grid_r_norm = np.meshgrid(np.linspace(-1, 1, W), np.linspace(0, 1, H))
    x_input[0, 4, :, :] = grid_x_norm
    x_input[0, 5, :, :] = grid_r_norm

    # 3. Predict
    with torch.no_grad():
        pred = model(torch.from_numpy(x_input).to(device))
    
    pred_real = (pred.cpu().numpy().squeeze() * stats['dcd_std']) + stats['dcd_mean']

    # 4. Plot
    # Construct Physical Grid for Plotting
    phys_x = np.linspace(x_bounds[0], x_bounds[1], W)
    phys_r = np.linspace(r_bounds[0], r_bounds[1], H)
    X, R = np.meshgrid(phys_x, phys_r)

    plt.figure(figsize=(10, 5))
    c = plt.contourf(X*1e9, R*1e9, pred_real, levels=60, cmap='seismic')
    cbar = plt.colorbar(c)
    cbar.set_label('Defect Charge Density ($C \cdot cm^{-3}$)')
    
    plt.title(f"Predicted DCD @ {args.time}s\nS={args.spacing}nm (Range $\pm${x_half_width:.1f}nm)")
    plt.xlabel('Axial Position X (nm)')
    plt.ylabel('Radius R (nm)')
    plt.tight_layout()
    plt.savefig(f"inference_S{args.spacing}.png")
    print(f"Saved inference_S{args.spacing}.png")

if __name__ == "__main__":
    inference()