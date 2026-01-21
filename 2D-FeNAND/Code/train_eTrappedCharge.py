import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from scipy.interpolate import griddata

# Import NVIDIA PhysicsNeMo
from physicsnemo.models.fno import FNO

# ==========================================
# 1. DATASET (Region: R.Gatesn2 Only)
# Focus on the gate region "R.Gatesn2" where the significant electron trapped charge (eTrappedChange) occurs.
# ==========================================
class GateRegionDataset(Dataset):
    # Updated to match your file naming convention
    TARGET_REGION = "R.Gatesn2"
    TARGET_QTY = "eTrappedCharge"
    
    def __init__(self, csv_file, data_dir=".", map_resolution=(64, 64), normalize=True, cache=True):
        self.data_dir = data_dir
        self.H, self.W = map_resolution
        self.normalize = normalize
        
        print(f"Loading mapping from: {csv_file}")
        self.df = pd.read_csv(csv_file)
        self.df.columns = self.df.columns.str.strip()
        
        required = ['Node', 'HZO_Thickness', 'Temperature', 'Retention_Time']
        self.df = self.df.dropna(subset=required)
        self.df['Node'] = self.df['Node'].astype(int)
        
        print(f"Found {len(self.df)} samples in CSV.")
        
        self.grid_x, self.grid_y = None, None
        
        if cache:
            self.cache_data = self._build_cache()
            if len(self.cache_data) == 0:
                # Fallback check for R.Gate_Sn_2 if R.Gatesn2 not found
                print(f"Warning: No files found for {self.TARGET_REGION}. Checking 'R.Gate_Sn_2'...")
                self.TARGET_REGION = "R.Gate_Sn_2"
                self.cache_data = self._build_cache()
                
            if len(self.cache_data) == 0:
                raise ValueError(f"No valid data found for {self.TARGET_QTY} in region R.Gatesn2 or R.Gate_Sn_2.")
                
            if self.normalize:
                self._compute_stats()
        else:
            self.cache_data = None

    def _build_cache(self):
        cache = []
        print(f"Caching {self.TARGET_REGION} data to RAM ({self.H}x{self.W})...")
        
        for idx, row in self.df.iterrows():
            try:
                node_id = int(row['Node'])
                scalars = self._get_scalars(row)
                
                # Flexible pattern matching for region name
                search_pattern = os.path.join(
                    self.data_dir, 
                    f"n{node_id}_des_*{self.TARGET_REGION}*_{self.TARGET_QTY}.dat"
                )
                files = glob.glob(search_pattern)
                
                if not files:
                    continue
                
                fpath = files[0] # Take first match
                
                # Load Data
                data = np.loadtxt(fpath, comments='#')
                if data.ndim == 1: data = data[None, :]
                if data.shape[0] < 10: continue
                    
                x, y, v = data[:, 1], data[:, 2], data[:, 3]
                
                # Initialize Grid
                if self.grid_x is None:
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()
                    gx, gy = np.mgrid[x_min:x_max:self.H*1j, y_min:y_max:self.W*1j]
                    self.grid_x, self.grid_y = gx, gy
                    print(f"Initialized Grid for {self.TARGET_REGION}: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}]")

                # Interpolate
                grid_z = griddata((x, y), v, (self.grid_x, self.grid_y), method='linear', fill_value=0)
                
                if np.isnan(grid_z).any():
                    grid_z_near = griddata((x, y), v, (self.grid_x, self.grid_y), method='nearest')
                    mask = np.isnan(grid_z)
                    grid_z[mask] = grid_z_near[mask]
                
                # Store as (1, H, W)
                map_data = grid_z[np.newaxis, :, :] 
                
                cache.append({'scalars': scalars, 'maps': map_data})
            
            except Exception as e:
                continue
                
        print(f"Cached {len(cache)} valid samples.")
        return cache

    def _get_scalars(self, row):
        t = float(row['HZO_Thickness']) * 1e3 
        temp = float(row['Temperature'])
        time = float(row['Retention_Time'])
        log_time = np.log10(time + 1.0)
        return np.array([t, temp, log_time], dtype=np.float32)

    def _compute_stats(self):
        print("Computing normalization stats...")
        all_scalars = np.stack([x['scalars'] for x in self.cache_data])
        all_maps = np.stack([x['maps'] for x in self.cache_data])
        
        # Log10 transform
        all_maps = np.log10(np.abs(all_maps) + 1.0)
        
        # Update cache
        for i in range(len(self.cache_data)):
            self.cache_data[i]['maps'] = all_maps[i]

        self.in_mean = np.mean(all_scalars, axis=0)
        self.in_std = np.std(all_scalars, axis=0) + 1e-8
        
        self.map_mean = np.mean(all_maps, axis=0)
        self.map_std = np.std(all_maps, axis=0) + 1e-8
        print("Stats computed.")

    def get_stats(self):
        return {
            'in_mean': torch.from_numpy(self.in_mean).float(),
            'in_std': torch.from_numpy(self.in_std).float(),
            'map_mean': torch.from_numpy(self.map_mean).float(),
            'map_std': torch.from_numpy(self.map_std).float(),
            'grid_x': self.grid_x,
            'grid_y': self.grid_y
        }

    def __len__(self):
        return len(self.cache_data)

    def __getitem__(self, idx):
        item = self.cache_data[idx]
        norm_scalars = (item['scalars'] - self.in_mean) / self.in_std
        norm_maps = (item['maps'] - self.map_mean) / self.map_std
        
        # Broadcast to (3, H, W)
        input_tensor = np.zeros((3, self.H, self.W), dtype=np.float32)
        input_tensor[0, :, :] = norm_scalars[0]
        input_tensor[1, :, :] = norm_scalars[1]
        input_tensor[2, :, :] = norm_scalars[2]
        
        # === FIX: Explicit .float() cast to prevent Double vs Float error ===
        return torch.from_numpy(input_tensor).float(), torch.from_numpy(norm_maps).float()

# ==========================================
# 2. MODEL
# ==========================================
class DeviceFNO(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 inputs -> 1 output (Charge)
        self.fno = FNO(
            in_channels=3,
            out_channels=1, 
            decoder_layers=1,
            decoder_layer_size=32,
            dimension=2,
            latent_channels=32,
            num_fno_layers=4,
            num_fno_modes=12,
            padding=[0, 0]
        )

    def forward(self, x):
        return self.fno(x)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='./csv/updated_node_mapping.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out_dir', type=str, default='checkpoints_gate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        full_ds = GateRegionDataset(args.csv, map_resolution=(256, 256))
    except ValueError as e:
        print(f"Error: {e}")
        return

    train_size = int(0.85 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    torch.save(full_ds.get_stats(), os.path.join(args.out_dir, "stats.pt"))
    
    model = DeviceFNO().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs*len(train_loader))
    criterion = nn.MSELoss()
    
    print("Starting training (Region: R.Gatesn2)...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

if __name__ == "__main__":
    train()
