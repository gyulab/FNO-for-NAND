import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from scipy.interpolate import griddata

# Import NVIDIA PhysicsNeMo
from physicsnemo.models.fno import FNO

class PhysicsGuidedLoss(nn.Module):
    """
    PHYSICS LOSS for 2D Electrostatic Potential and Electric Field
    -----------------------------------------------------------
    Implemented a Sobolev (H1) loss that enforces the Electric Field is the gradient of the Potential.
    pred_Ex, pred_Ey = self.gradient(pred_smooth)
    We can implement more complex physics losses as needed. (Monotonic loss, Poission eqn, etc.)
    """
    def __init__(self, stats, h, w, lambda_phys=0.1): 
        super().__init__()
        self.dx = 1.0 / h
        self.dy = 1.0 / w
        self.lambda_phys = lambda_phys
        self.l1 = nn.L1Loss()
        
        # Gaussian Kernel for smoothing derivatives
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
        kernel = kernel / kernel.sum()
        self.register_buffer('smooth_kernel', kernel.view(1, 1, 3, 3))

    def smooth(self, x):
        """Apply Gaussian blur to smooth out griddata"""
        return F.conv2d(x, self.smooth_kernel, padding=1, groups=1)

    def gradient(self, u):
        """Compute First Derivative"""
        u_x = (u[..., 1:] - u[..., :-1]) / self.dx
        pad_x = torch.zeros_like(u_x[..., :1])
        u_x = torch.cat([u_x, pad_x], dim=3)
        
        u_y = (u[..., 1:, :] - u[..., :-1, :]) / self.dy
        pad_y = torch.zeros_like(u_y[..., :1, :])
        u_y = torch.cat([u_y, pad_y], dim=2)
        return u_x, u_y

    def forward(self, pred_norm, target_norm):
        loss_data = self.l1(pred_norm, target_norm)

        # 1. Smooth the inputs -- only smooth for the gradient calculation
        pred_smooth = self.smooth(pred_norm[:, 0:1, :, :])   # Potential
        targ_smooth = self.smooth(target_norm[:, 0:1, :, :]) # Potential

        # 2. Compute Gradients on smoothed maps
        pred_Ex, pred_Ey = self.gradient(pred_smooth)
        targ_Ex, targ_Ey = self.gradient(targ_smooth)
        
        # 3. Physics Loss: Match the Electric Field Trend
        loss_h1 = self.l1(pred_Ex, targ_Ex) + self.l1(pred_Ey, targ_Ey)
        
        total_loss = loss_data + self.lambda_phys * loss_h1
        
        return total_loss, loss_data.item(), loss_h1.item(), 0.0

# ==========================================
# 2. DATASET
# ==========================================
class DeviceMapDataset(Dataset):
    FIELDS = ["ElectrostaticPotential", "eTrappedCharge", "eCurrentDensity"]
    
    def __init__(self, csv_file, data_dir=".", map_resolution=(256, 256), normalize=True, cache=True):
        self.data_dir = data_dir
        self.H, self.W = map_resolution
        self.normalize = normalize
        
        # Load CSV
        print(f"Loading mapping from: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.df.columns = self.df.columns.str.strip()
        required = ['Node', 'HZO_Thickness', 'Temperature', 'Retention_Time']
        self.df = self.df.dropna(subset=required)
        self.df['Node'] = self.df['Node'].astype(int)
        
        self.dx = 1.0
        self.dy = 1.0
        self.cache_data = []
        
        if cache:
            self._build_cache()
            if self.normalize:
                self._compute_stats()

    def _build_cache(self):
        print(f"Caching data ({self.H}x{self.W})...")
        for idx, row in self.df.iterrows():
            try:
                node_id = int(row['Node'])
                scalars = self._get_scalars(row)
                maps = np.zeros((3, self.H, self.W), dtype=np.float32)
                
                valid_sample = True
                for i, qty in enumerate(self.FIELDS):
                    search_pattern = os.path.join(self.data_dir, f"n{node_id}_des_*_{qty}.dat")
                    files = glob.glob(search_pattern)
                    if not files: 
                        valid_sample = False; break
                    
                    fpath = max(files, key=os.path.getsize)
                    data = np.loadtxt(fpath, comments='#')
                    
                    x, y, v = data[:, 1], data[:, 2], data[:, 3]
                    
                    # Compute Grid Spacing
                    if self.dx == 1.0:
                        x_range = x.max() - x.min()
                        y_range = y.max() - y.min()
                        self.dx = x_range / self.H
                        self.dy = y_range / self.W
                        print(f"Grid Physical Spacing: dx={self.dx:.2e}, dy={self.dy:.2e}")

                    # Interpolate
                    gx, gy = np.mgrid[x.min():x.max():self.H*1j, y.min():y.max():self.W*1j]
                    grid_z = griddata((x, y), v, (gx, gy), method='linear', fill_value=0)
                    if np.isnan(grid_z).any():
                        grid_z[np.isnan(grid_z)] = 0
                    maps[i] = grid_z

                if valid_sample:
                    self.cache_data.append({'scalars': scalars, 'maps': maps})
            except Exception: continue
        print(f"Cached {len(self.cache_data)} samples.")

    def _get_scalars(self, row):
        t = float(row['HZO_Thickness']) * 1e3 
        temp = float(row['Temperature'])
        time = float(row['Retention_Time'])
        log_time = np.log10(time + 1.0)
        return np.array([t, temp, log_time], dtype=np.float32)

    def _compute_stats(self):
        all_scalars = np.stack([x['scalars'] for x in self.cache_data])
        all_maps = np.stack([x['maps'] for x in self.cache_data])
        
        # Log transform Charge/Current
        all_maps[:, 1] = np.log10(np.abs(all_maps[:, 1]) + 1.0)
        all_maps[:, 2] = np.log10(np.abs(all_maps[:, 2]) + 1e-20)
        
        # Update cache
        for i in range(len(self.cache_data)):
            self.cache_data[i]['maps'] = all_maps[i]

        self.in_mean = np.mean(all_scalars, axis=0)
        self.in_std = np.std(all_scalars, axis=0) + 1e-8
        self.map_mean = np.mean(all_maps, axis=0)
        self.map_std = np.std(all_maps, axis=0) + 1e-8

    def get_stats(self):
        return {
            'map_mean': torch.from_numpy(self.map_mean).float(),
            'map_std': torch.from_numpy(self.map_std).float(),
            'dx': self.dx,
            'dy': self.dy
        }

    def __len__(self): return len(self.cache_data)

    def __getitem__(self, idx):
        item = self.cache_data[idx]
        norm_scalars = (item['scalars'] - self.in_mean) / self.in_std
        norm_maps = (item['maps'] - self.map_mean) / self.map_std
        
        input_tensor = np.zeros((3, self.H, self.W), dtype=np.float32)
        input_tensor[0, :, :] = norm_scalars[0]
        input_tensor[1, :, :] = norm_scalars[1]
        input_tensor[2, :, :] = norm_scalars[2]
        
        return torch.from_numpy(input_tensor), torch.from_numpy(norm_maps)

# ==========================================
# 3. MODEL
# ==========================================
class DeviceFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.fno = FNO(
            in_channels=3,      # Inputs: Thickness, Temp, Time
            out_channels=3,     # Outputs: Potential, Charge, Current
            decoder_layers=1,
            decoder_layer_size=32,
            dimension=2,
            latent_channels=64,
            num_fno_layers=4,
            num_fno_modes=16, 
            padding=[0, 0]
        )

    def forward(self, x):
        return self.fno(x)

# ==========================================
# 4. TRAINING LOOP WITH PINO
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='./csv/updated_node_mapping.csv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_phys', type=float, default=0.1, help="Weight for physics loss")
    parser.add_argument('--out_dir', type=str, default='checkpoints_pino')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Data
    full_ds = DeviceMapDataset(args.csv, map_resolution=(256, 256))
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Setup PINO Components
    stats = full_ds.get_stats()
    torch.save(stats, os.path.join(args.out_dir, "stats.pt"))

    # Pass grid size (256, 256) explicitly
    criterion = PhysicsGuidedLoss(
        stats=stats, 
        h=256, 
        w=256, 
        lambda_phys = 0.05
    ).to(device)
    
    model = DeviceFNO().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs*len(train_loader))
    
    print(f"Starting PINO Training (Lambda={args.lambda_phys})...")
    
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0
        avg_mse = 0
        avg_phys = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)
            
            # PINO Loss Calculation
            loss, mse_val, h1_val, h2_val = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            avg_loss += loss.item()
            avg_mse += mse_val
            avg_phys += (h1_val + h2_val)
            
        avg_loss /= len(train_loader)
        avg_mse /= len(train_loader)
        avg_phys /= len(train_loader)
        
        # Validation
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                v_loss, v_mse, _, _ = criterion(pred, y)
                val_mse += v_mse
        val_mse /= len(val_loader)
        
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1} | Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f} + Phys: {avg_phys:.4f}) | Val MSE: {val_mse:.4f}")
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_ep{epoch+1}.pt"))

if __name__ == "__main__":
    train()
