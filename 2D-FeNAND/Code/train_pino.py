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

# ==========================================
# 1. HELPER: FINITE DIFFERENCE OPS
# ==========================================
def dx(u, dx_val, dim, padding="replication"):
    """
    Computes first derivative using central difference.
    u: [Batch, Channels, H, W]
    dim: 2 for H (x), 3 for W (y)
    """
    # Kernel: [-1/2, 0, 1/2]
    kernel = torch.tensor([-0.5, 0.0, 0.5], device=u.device).view(1, 1, 3)
    if dim == 2: # Vertical (x)
        kernel = kernel.view(1, 1, 3, 1)
        pad = (0, 0, 1, 1) # Pad H
    else: # Horizontal (y)
        kernel = kernel.view(1, 1, 1, 3)
        pad = (1, 1, 0, 0) # Pad W
        
    # Replicate padding to handle boundaries
    u_pad = F.pad(u, pad, mode='replicate')
    
    # Apply convolution per channel
    ch = u.shape[1]
    grad = F.conv2d(u_pad, kernel.repeat(ch, 1, 1, 1), groups=ch)
    return grad / dx_val

def ddx(u, dx_val, dim, padding="replication"):
    """
    Computes second derivative (Laplacian component).
    u: [Batch, Channels, H, W]
    dim: 2 for H (x), 3 for W (y)
    """
    # Kernel: [1, -2, 1]
    kernel = torch.tensor([1.0, -2.0, 1.0], device=u.device).view(1, 1, 3)
    if dim == 2:
        kernel = kernel.view(1, 1, 3, 1)
        pad = (0, 0, 1, 1)
    else:
        kernel = kernel.view(1, 1, 1, 3)
        pad = (1, 1, 0, 0)
        
    u_pad = F.pad(u, pad, mode='replicate')
    
    ch = u.shape[1]
    lap = F.conv2d(u_pad, kernel.repeat(ch, 1, 1, 1), groups=ch)
    return lap / (dx_val**2)

# ==========================================
# 2. PDE MODULE (Poisson Equation)
# ==========================================
class Poisson(nn.Module):
    """
    Custom Poisson PDE definition for PINO
    Equation: ε∇²φ + ρ = 0
    """
    def __init__(self, stats, epsilon_r=25.0):
        super().__init__()
        # Physics Constants
        # epsilon_0 = 8.854e-14 F/cm
        self.epsilon = 8.854e-14 * epsilon_r
        
        # Grid stats for Un-normalization
        self.register_buffer('map_mean', stats['map_mean'].view(1, 3, 1, 1))
        self.register_buffer('map_std',  stats['map_std'].view(1, 3, 1, 1))
        self.dx_val = stats['dx']
        self.dy_val = stats['dy']

    def unnormalize(self, x):
        return (x * self.map_std) + self.map_mean

    def inverse_log_transform(self, log_tensor):
        # Revert log10(|x| + 1)
        magnitude = torch.pow(10.0, log_tensor) - 1.0
        return -1.0 * magnitude

    def forward(self, input_tensor):
        """
        Input: Normalized Tensor [Batch, 3, H, W]
        Returns: Residual map [Batch, 1, H, W]
        """
        # 1. Recover Physical Units
        real_tensor = self.unnormalize(input_tensor)
        
        phi = real_tensor[:, 0:1, :, :]      # Potential (V)
        rho_log = real_tensor[:, 1:2, :, :]  # Charge (Log Q/cm^3)
        rho = self.inverse_log_transform(rho_log) # Charge (C/cm^3)

        # 2. Compute Derivatives (FDM Method)
        # d²φ/dx²
        phi_xx = ddx(phi, self.dx_val, dim=2)
        # d²φ/dy²
        phi_yy = ddx(phi, self.dy_val, dim=3)
        
        # 3. Compute Residual
        # Poisson: ε(φ_xx + φ_yy) = -ρ  ->  ε(φ_xx + φ_yy) + ρ = 0
        laplacian = phi_xx + phi_yy
        residual = (self.epsilon * laplacian) + rho

        # 4. Zero out boundaries
        # Pad with 0s for the outer 2 pixels
        mask = torch.ones_like(residual)
        mask[:, :, :2, :] = 0
        mask[:, :, -2:, :] = 0
        mask[:, :, :, :2] = 0
        mask[:, :, :, -2:] = 0
        
        return residual * mask

# ==========================================
# 3. DATASET
# ==========================================
class DeviceMapDataset(Dataset):
    FIELDS = ["ElectrostaticPotential", "eTrappedCharge", "eCurrentDensity"]
    
    def __init__(self, csv_file, data_dir=".", map_resolution=(256, 256), normalize=True, cache=True):
        self.data_dir = data_dir
        self.H, self.W = map_resolution
        self.normalize = normalize
        
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
                    
                    # Calculate Physical Grid Spacing (cm)
                    if self.dx == 1.0:
                        # Assuming microns in file, convert to cm: * 1e-4
                        x_cm = x * 1e-4
                        y_cm = y * 1e-4
                        self.dx = (x_cm.max() - x_cm.min()) / self.H
                        self.dy = (y_cm.max() - y_cm.min()) / self.W
                        print(f"Physical Grid (cm): dx={self.dx:.2e}, dy={self.dy:.2e}")

                    gx, gy = np.mgrid[x.min():x.max():self.H*1j, y.min():y.max():self.W*1j]
                    grid_z = griddata((x, y), v, (gx, gy), method='linear', fill_value=0)
                    if np.isnan(grid_z).any(): grid_z[np.isnan(grid_z)] = 0
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
# 4. MODEL
# ==========================================
class DeviceFNO(nn.Module):
    def __init__(self):
        super().__init__()
        self.fno = FNO(
            in_channels=3,
            out_channels=3,
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
# 5. TRAINING LOOP
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='./csv/updated_node_mapping.csv')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_phys', type=float, default=0.1, help="Physics loss weight")
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
    
    stats = full_ds.get_stats()
    torch.save(stats, os.path.join(args.out_dir, "stats.pt"))

    # 2. Initialize Models
    model = DeviceFNO().to(device)
    
    # Initialize PDE Physics Module
    physics_eqn = Poisson(stats=stats, epsilon_r=25.0).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs*len(train_loader))
    
    # Loss Functions
    l1_loss = nn.L1Loss()
    
    print(f"Starting PINO Training (Lambda={args.lambda_phys})...")
    
    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0
        avg_data = 0
        avg_pde = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Predict
            pred = model(x)
            
            # A. Data Loss (Supervised)
            loss_data = l1_loss(pred, y)
            
            # B. Physics Loss (Unsupervised/Consistency)
            # Calculate residual of Poisson's equation
            pde_residual = physics_eqn(pred)
            loss_pde = torch.mean(torch.abs(pde_residual))
            
            # Total Loss
            total_loss = loss_data + (args.lambda_phys * loss_pde)
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            avg_loss += total_loss.item()
            avg_data += loss_data.item()
            avg_pde += loss_pde.item()
            
        avg_loss /= len(train_loader)
        avg_data /= len(train_loader)
        avg_pde /= len(train_loader)
        
        # Validation (Pure Data Accuracy)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += l1_loss(pred, y).item()
        val_loss /= len(val_loader)
        
        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1} | Tot: {avg_loss:.4f} (Data: {avg_data:.4f} + PDE: {avg_pde:.4f}) | Val L1: {val_loss:.4f}")
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model_best.pt"))

if __name__ == "__main__":
    train()