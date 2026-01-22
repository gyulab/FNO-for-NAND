import os
import shutil
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from physicsnemo.models.fno import FNO

def generate_thickness_gif():
    # ==========================================
    # 1. Configuration
    # ==========================================
    parser = argparse.ArgumentParser(description="Generate Thickness (TL) Sweep GIF")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_gif", type=str, default="thickness_sweep_10ks.gif")
    args = parser.parse_args()

    # Fixed Parameters (as requested)
    SPACING = 20.0  # nm
    PGM = 15.0      # Volts
    TIME = 10000.0  # Seconds
    
    # Sweep Parameters: Thickness (TL) 20A -> 40A, 20 frames
    N_FRAMES = 20
    tl_values = np.linspace(20.0, 40.0, N_FRAMES)
    
    print(f"Generating GIF for Thickness Sweep: {tl_values[0]}A -> {tl_values[-1]}A")
    print(f"Fixed Params: Spacing={SPACING}nm, PGM={PGM}V, Time={TIME}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 2. Load Model & Stats
    # ==========================================
    stats_path = os.path.join(args.checkpoint_dir, "stats.pt")
    model_path = os.path.join(args.checkpoint_dir, "best_model.pth")

    if not os.path.exists(stats_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Checkpoints not found. Run training first.")

    # Load Stats
    try:
        stats = torch.load(stats_path, weights_only=False)
    except TypeError:
        stats = torch.load(stats_path)

    # Initialize Model
    model = FNO(
        in_channels=6, 
        out_channels=1, 
        decoder_layers=1, 
        decoder_layer_size=32, 
        dimension=2,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=12,
        padding=8
    ).to(device)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except TypeError:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()

    # ==========================================
    # 3. Batch Inference (Pre-calculate for Scale)
    # ==========================================
    H, W = 64, 128
    results = []
    
    print("Running batch inference...")
    
    with torch.no_grad():
        for tl in tl_values:
            # --- Prepare Input ---
            log_time = np.log10(TIME + 1e-6)
            # Param Order: [Spacing, TL, PGM, Time]
            raw_params = np.array([SPACING, tl, PGM, log_time])
            norm_params = (raw_params - stats['p_mean']) / stats['p_std']
            
            x_input = np.zeros((1, 6, H, W), dtype=np.float32)
            
            # Broadcast params
            for c in range(4):
                x_input[0, c, :, :] = norm_params[c]
                
            # Coordinate Embeddings (Normalized)
            grid_x_norm, grid_r_norm = np.meshgrid(np.linspace(-1, 1, W), np.linspace(0, 1, H))
            x_input[0, 4, :, :] = grid_x_norm
            x_input[0, 5, :, :] = grid_r_norm
            
            # --- Predict ---
            x_tensor = torch.from_numpy(x_input).to(device)
            pred_norm = model(x_tensor)
            pred_real = (pred_norm.cpu().numpy().squeeze() * stats['dcd_std']) + stats['dcd_mean']
            
            # --- Define Physical Domain ---
            # X bounds depend on fixed Spacing (20nm -> +/- 80nm)
            x_half = 4.0 * SPACING
            x_bounds = (-x_half, x_half) 
            
            # R bounds depend on dynamic TL
            # r_start = TL thickness (converted to nm)
            r_start = tl * 0.1 
            # r_width is approx 13nm (active trapping layer width)
            r_width = 13.0     
            r_bounds = (r_start, r_start + r_width)
            
            results.append({
                'tl': tl,
                'map': pred_real,
                'x_bounds': x_bounds,
                'r_bounds': r_bounds
            })

    # ==========================================
    # 4. Generate Frames
    # ==========================================
    # Global Color Scale (Symmetric)
    all_maps = np.concatenate([r['map'].flatten() for r in results])
    limit = np.percentile(np.abs(all_maps), 99) * 1.05
    vmin, vmax = -limit, limit
    levels = np.linspace(vmin, vmax, 100)
    
    print(f"Global Color Scale Locked: [{vmin:.2e}, {vmax:.2e}]")
    
    temp_dir = "thickness_frames"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    print("Rendering frames...")
    
    for idx, res in enumerate(results):
        fig = plt.figure(figsize=(10, 5))
        
        # Grid for current frame
        xb = res['x_bounds']
        rb = res['r_bounds']
        xx = np.linspace(xb[0], xb[1], W)
        rr = np.linspace(rb[0], rb[1], H)
        X, R = np.meshgrid(xx, rr)
        
        # Plot
        c = plt.contourf(X, R, res['map'], levels=levels, cmap='seismic', vmin=vmin, vmax=vmax)
        
        # Labels
        plt.colorbar(c, label='Defect Charge Density ($C \cdot cm^{-3}$)')
        plt.title(f"Tunnel Thickness: {res['tl']:.1f} $\AA$\n(Spacing={int(SPACING)}nm, PGM={int(PGM)}V)")
        plt.xlabel('Axial Position X (nm)')
        plt.ylabel('Radius R (nm)')
        
        # --- Altering Axis ---
        # We explicitly set the limits to the current physical bounds.
        # This causes the Y-axis ticks to "slide" up as TL increases.
        plt.xlim(xb)
        plt.ylim(rb)

        plt.tight_layout()
        fname = os.path.join(temp_dir, f"frame_{idx:03d}.png")
        plt.savefig(fname, dpi=100)
        plt.close(fig)
        frame_files.append(fname)
        
        if idx % 5 == 0: print(f"  Frame {idx}/{N_FRAMES}")

    # ==========================================
    # 5. Compile GIF
    # ==========================================
    print(f"Compiling GIF to {args.output_gif}...")
    with imageio.get_writer(args.output_gif, mode='I', duration=0.2) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    shutil.rmtree(temp_dir)
    print("Done!")

if __name__ == "__main__":
    generate_thickness_gif()
