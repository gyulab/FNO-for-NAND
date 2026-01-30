import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Import architectures
try:
    from train_model import DeviceFNO
except ImportError:
    from train_eTrappedCharge import DeviceFNO
    
from train_iv_net import ShallowIVNet, CONFIG, parse_plt_iv

def predict_vth(hzo_nm, temp_k, time_s):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Models
    fno = DeviceFNO().to(device)
    fno.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False))
    fno.eval()
    
    iv_net = ShallowIVNet().to(device)
    iv_net.load_state_dict(torch.load("checkpoints/iv_net_best.pt", map_location=device, weights_only=False))
    iv_net.eval()
    
    # 2. Prepare FNO Input
    stats = torch.load("checkpoints/stats.pt", map_location=device, weights_only=False)
    # Handle stats dict format
    in_mean = stats.get('in_mean') if isinstance(stats, dict) else stats['in_mean']
    in_std = stats.get('in_std') if isinstance(stats, dict) else stats['in_std']
    map_mean = stats.get('map_mean').cpu().numpy()
    map_std = stats.get('map_std').cpu().numpy()

    # Normalize scalars for FNO
    raw_in = torch.tensor([hzo_nm, temp_k, np.log10(time_s + 1.0)]).float().to(device)
    norm_in = (raw_in - in_mean) / in_std
    fno_in = norm_in.view(1, 3, 1, 1).repeat(1, 1, 256, 256)
    
    # 3. Generate Physics Map
    with torch.no_grad():
        fno_out = fno(fno_in).squeeze().cpu().numpy()
        
        # FIX: Handle incorrect broadcasting
        if fno_out.ndim == 5 and fno_out.shape[:3] == (3, 1, 3):
            new_map = np.empty((3, 256, 256), dtype=fno_out.dtype)
            for c in range(3):
                new_map[c] = fno_out[c, 0, c]
            fno_out = new_map
        elif fno_out.ndim == 4 and fno_out.shape[:2] == (3, 3):
             new_map = np.empty((3, 256, 256), dtype=fno_out.dtype)
             for c in range(3):
                 new_map[c] = fno_out[c, c]
             fno_out = new_map

        # Denormalize
        phys_map = fno_out * map_std[:, None, None] + map_mean[:, None, None]
        
        # FIX: Handle incorrect broadcasting caused by Stats shape (Post-Denorm)
        if phys_map.ndim == 5 and phys_map.shape[:3] == (3, 1, 3):
             new_map = np.empty((3, 256, 256), dtype=phys_map.dtype)
             for c in range(3):
                 new_map[c] = phys_map[c, 0, c]
             phys_map = new_map
        elif phys_map.ndim == 4 and phys_map.shape[:2] == (3, 3):
             new_map = np.empty((3, 256, 256), dtype=phys_map.dtype)
             for c in range(3):
                 new_map[c] = phys_map[c, c]
             phys_map = new_map
        
    # 4. Predict IV Curve
    map_tensor = torch.tensor(phys_map).unsqueeze(0).float().to(device) 
    scalar_tensor = torch.tensor([hzo_nm, temp_k, np.log10(time_s + 1.0)]).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        iv_pred = iv_net(map_tensor, scalar_tensor).cpu().numpy().flatten()
        
    # 5. Calculate Vth
    vg_axis = np.linspace(0, 10.0, 100)
    target_log = np.log10(1e-9)
    
    idx = np.where(iv_pred > target_log)[0]
    if len(idx) == 0: 
        vth = np.nan
    else:
        i = idx[0]
        y1, y2 = iv_pred[i-1], iv_pred[i]
        x1, x2 = vg_axis[i-1], vg_axis[i]
        vth = x1 + (x2 - x1) * (target_log - y1) / (y2 - y1)

    return vg_axis, iv_pred, vth

def get_ground_truth(hzo_nm, temp_k, time_s):
    csv_path = "./csv/full_node_mapping.csv"
    if not os.path.exists(csv_path): return None, None
    
    df = pd.read_csv(csv_path)
    match = df[
        (np.isclose(df['HZO_Thickness'] * 1e3, hzo_nm, atol=0.1)) &
        (np.isclose(df['Temperature'], temp_k, atol=1.0)) &
        (np.isclose(df['Retention_Time'], time_s, atol=1.0)) &
        (df['State'].str.strip() == 'Erased')
    ]
    
    if not match.empty:
        fname = match.iloc[0]['Filename']
        plt_full_path = os.path.join("./data", os.path.basename(str(fname).strip()))
        gt_vg, gt_id = parse_plt_iv(plt_full_path)
        if gt_vg is not None:
             return gt_vg, np.log10(np.maximum(gt_id, 1e-15))
    return None, None

if __name__ == "__main__":
    HZO = 6.0       # nm
    TIME = 1000     # s
    TEMPS_K = [400.0, 430.0, 473.0]
    COLORS = {400.0: 'b', 430.0: 'g', 473.0: 'r'}
    
    plt.figure(figsize=(10, 8))
    
    for T in TEMPS_K:
        print(f"--- Processing T={T}K ---")
        # AI Prediction
        vg, iv, vth = predict_vth(HZO, T, TIME)
        print(f"  AI Vth: {vth:.3f} V")
        
        plt.plot(vg, iv, '.', color=COLORS[T], markersize=5, 
                 label=f'AI {T:.0f}K (Vth={vth:.2f}V)')
        
        # Ground Truth
        gt_vg, gt_iv = get_ground_truth(HZO, T, TIME)
        if gt_vg is not None:
            plt.plot(gt_vg, gt_iv, '-', color=COLORS[T], linewidth=2, alpha=0.7,
                     label=f'TCAD {T:.0f}K')
        elif T == 400.0:
             print("  (Interpolation Point: No Ground Truth expected)")
        else:
             print("  [WARN] Ground Truth not found")

    plt.axhline(np.log10(1e-9), color='k', linestyle='--', alpha=0.5, label='Vth Threshold')
    plt.xlabel('Gate Voltage (V)', fontsize=12)
    plt.ylabel('Log10 Drain Current (A)', fontsize=12)
    plt.title(f'IV Interpolation Capability Test\n(HZO={HZO}nm, Time={TIME}s)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    out_file = "iv_plots/iv_interpolation_test.png"
    plt.savefig(out_file)
    print(f"\nSaved plot to {out_file}")
