import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from train_eTrappedCharge import DeviceFNO # Import from training file

# === CONFIGURATION ===
# Range for Log10(Charge) visualization -- Effective eTrapped Charge ranges from 10^10 to 10^20 cm^-3
# Adjust these values based on training data range for better contrast
CHARGE_VMIN = 10.0  
CHARGE_VMAX = 20
# =====================

def create_animation():
    # ==========================================
    # Temperature Sweep Animation Example for eTrapped Charge
    # ==========================================
    MODEL_PATH = "checkpoints_gate/best_model.pt"
    STATS_PATH = "checkpoints_gate/stats.pt"
    OUT_DIR = "animations"
    THICKNESS = 0.007 # 7nm -- adjust this value to observe different ferroelectric thicknesses
    TIME = 1000.0 # Fixed retention time in seconds
    TEMP_RANGE = [27.0, 250.0] # From 27C to 250C
    FRAMES = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run train_eTrappedCharge.py first.")
        return

    # Load Stats (weights_only=False allows numpy arrays)
    stats = torch.load(STATS_PATH, map_location=device, weights_only=False)
    in_mean, in_std = stats['in_mean'], stats['in_std']
    map_mean, map_std = stats['map_mean'], stats['map_std']
    
    # Detect Resolution
    H, W = map_mean.shape[1], map_mean.shape[2]
    print(f"Detected resolution: {H}x{W}")
        
    model = DeviceFNO().to(device)
    # Load weights (weights_only=True is safe for state_dict)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    images = []
    temps = np.linspace(TEMP_RANGE[0], TEMP_RANGE[1], FRAMES)
    
    print(f"Generating Gate Charge Animation ({TEMP_RANGE[0]} -> {TEMP_RANGE[1]} C)...")
    
    for i, temp in enumerate(temps):
        # Prepare Input
        raw_in = torch.tensor([THICKNESS * 1e3, temp, np.log10(TIME + 1.0)]).float().to(device)
        norm_in = (raw_in - in_mean) / in_std
        
        # Broadcast to [1, 3, H, W]
        x = norm_in.view(1, 3, 1, 1).repeat(1, 1, H, W)
        
        with torch.no_grad():
            pred = model(x)
            
        # Denormalize
        pred = pred * map_std + map_mean
        pred = pred.squeeze().cpu().numpy()
        
        # pred is Log10(Charge + 1)
        charge_log = pred
        
        # Plotting
        fig, ax = plt.subplots(figsize=(6, 5))
        
        im = ax.imshow(charge_log, cmap='plasma', origin='lower', aspect='auto',
                       vmin=CHARGE_VMIN, vmax=CHARGE_VMAX)
        
        ax.set_title(f"eTrappedCharge (Log10)\nTemp: {temp:.1f} C (R.Gatesn2)", fontsize=12)
        ax.set_xlabel("Y (Height)")
        ax.set_ylabel("X (Width)")
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Log10(Density) [cm^-3]")
        
        plt.tight_layout()
        
        # Capture
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img)
        plt.close()
        
    gif_path = os.path.join(OUT_DIR, "gate_charge_temperature.gif")
    imageio.mimsave(gif_path, images, fps=5)
    print(f"Saved {gif_path}")

if __name__ == "__main__":
    create_animation()