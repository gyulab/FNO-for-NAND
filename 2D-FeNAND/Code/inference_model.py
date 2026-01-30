import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from train_model import DeviceFNO

# === CONFIGURATION ===
CURR_VMIN = -30.0  
CURR_VMAX = 10.0
# =====================

def animate_temperature(model, stats, out_dir):
    THICKNESS = 0.007 
    TIME = 1000.0
    TEMP_RANGE = [300.0, 523.0] # Kelvin
    FRAMES = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_mean, in_std = stats['in_mean'], stats['in_std']
    map_mean, map_std = stats['map_mean'], stats['map_std']
    
    H, W = map_mean.shape[1], map_mean.shape[2]
    
    images = []
    temps = np.linspace(TEMP_RANGE[0], TEMP_RANGE[1], FRAMES)
    
    print(f"Generating Temperature Animation ({TEMP_RANGE[0]} -> {TEMP_RANGE[1]} K)...")
    
    for i, temp in enumerate(temps):
        raw_in = torch.tensor([THICKNESS * 1e3, temp, np.log10(TIME + 1.0)]).float().to(device)
        norm_in = (raw_in - in_mean) / in_std
        x = norm_in.view(1, 3, 1, 1).repeat(1, 1, H, W)
        
        with torch.no_grad():
            pred = model(x)
            
        pred = pred * map_std + map_mean
        pred = pred.squeeze().cpu().numpy()
        
        # Log Scale Logic
        # pred[0] = Linear (Current) -> Convert to Log
        # pred[1] = Log (Charge) -> Use as is
        # pred[2] = Log (Field) -> Use as is
        
        grid_curr_log = np.log10(np.abs(pred[0]) + 1e-20)
        grid_trap_log = pred[1]
        grid_field_log = pred[2]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. eTrappedCharge
        im0 = axes[0].imshow(grid_trap_log, cmap='magma', origin='lower', aspect='auto')
        axes[0].set_title("Log eTrappedCharge")
        axes[0].set_ylabel("X (Width)")
        axes[0].set_xlabel("Y (Height)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 2. eCurrentDensity
        im1 = axes[1].imshow(grid_curr_log, cmap='plasma', origin='lower', aspect='auto',
                             vmin=CURR_VMIN, vmax=CURR_VMAX)
        axes[1].set_title(f"Log eCurrentDensity\n(Scale: {CURR_VMIN} to {CURR_VMAX})")
        axes[1].set_xlabel("Y (Height)")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. ElectricField
        im2 = axes[2].imshow(grid_field_log, cmap='viridis', origin='lower', aspect='auto')
        axes[2].set_title("Log ElectricField")
        axes[2].set_xlabel("Y (Height)")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        

        
        plt.suptitle(f"Temp: {temp:.1f} K | Thickness: {THICKNESS*1e3:.1f} nm")
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img)
        plt.close()
        
    gif_path = os.path.join(out_dir, "inference_log_all.gif")
    imageio.mimsave(gif_path, images, fps=5)
    print(f"Saved {gif_path}")


def animate_time(model, stats, out_dir):
    THICKNESS = 0.007 
    TEMP = 473.0 # Kelvin
    TIME_RANGE = [0.0, 10000.0]
    FRAMES = 20
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_mean, in_std = stats['in_mean'], stats['in_std']
    map_mean, map_std = stats['map_mean'], stats['map_std']
    
    H, W = map_mean.shape[1], map_mean.shape[2]
    
    images = []
    times = np.linspace(TIME_RANGE[0], TIME_RANGE[1], FRAMES)
    
    print(f"Generating Time Animation ({TIME_RANGE[0]} -> {TIME_RANGE[1]} s)...")
    
    for i, t in enumerate(times):
        raw_in = torch.tensor([THICKNESS * 1e3, TEMP, np.log10(t + 1.0)]).float().to(device)
        norm_in = (raw_in - in_mean) / in_std
        x = norm_in.view(1, 3, 1, 1).repeat(1, 1, H, W)
        
        with torch.no_grad():
            pred = model(x)
            
        pred = pred * map_std + map_mean
        pred = pred.squeeze().cpu().numpy()
        
        # Log Scale Logic
        grid_curr_log = np.log10(np.abs(pred[0]) + 1e-20)
        grid_trap_log = pred[1]
        grid_field_log = pred[2]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. eTrappedCharge
        im0 = axes[0].imshow(grid_trap_log, cmap='magma', origin='lower', aspect='auto')
        axes[0].set_title("Log eTrappedCharge")
        axes[0].set_ylabel("X (Width)")
        axes[0].set_xlabel("Y (Height)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 2. eCurrentDensity
        im1 = axes[1].imshow(grid_curr_log, cmap='plasma', origin='lower', aspect='auto',
                             vmin=CURR_VMIN, vmax=CURR_VMAX)
        axes[1].set_title(f"Log eCurrentDensity\n(Scale: {CURR_VMIN} to {CURR_VMAX})")
        axes[1].set_xlabel("Y (Height)")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. ElectricField
        im2 = axes[2].imshow(grid_field_log, cmap='viridis', origin='lower', aspect='auto')
        axes[2].set_title("Log ElectricField")
        axes[2].set_xlabel("Y (Height)")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        

        
        plt.suptitle(f"Time: {t:.1f} s | Temp: {TEMP} K | Thickness: {THICKNESS*1e3:.1f} nm")
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img)
        plt.close()
        
    gif_path = os.path.join(out_dir, "inference_time_log_all.gif")
    imageio.mimsave(gif_path, images, fps=5)
    print(f"Saved {gif_path}")


def main():
    MODEL_PATH = "checkpoints/best_model.pt"
    STATS_PATH = "checkpoints/stats.pt"
    OUT_DIR = "animations"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        return

    # Load shared assets once
    print(f"Loading model/stats from {MODEL_PATH}...")
    stats = torch.load(STATS_PATH, map_location=device, weights_only=False)
    
    model = DeviceFNO().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # Run Animations
    animate_temperature(model, stats, OUT_DIR)
    animate_time(model, stats, OUT_DIR)

if __name__ == "__main__":
    main()
