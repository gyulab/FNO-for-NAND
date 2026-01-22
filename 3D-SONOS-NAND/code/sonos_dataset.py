import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import griddata
from io import StringIO

class SonosGridder:
    """
    Interpolates scattered Ginestra outputs onto a fixed size computational grid.
    The PHYSICAL domain varies, but the COMPUTATIONAL grid is always (H, W).
    """
    def __init__(self, resolution=(64, 128)): # H=R, W=X
        self.H, self.W = resolution
        # We define a normalized canonical grid [-1, 1] for X and [0, 1] for R
        # This will be used as input features to the FNO
        self.grid_x_norm, self.grid_r_norm = np.meshgrid(
            np.linspace(-1, 1, self.W),
            np.linspace(0, 1, self.H)
        )

    def interpolate(self, df, spacing, tl):
        """
        Interpolates data onto the grid.
        Dynamic Bounds Logic:
          - X Range: Dependent on Spacing. We assume the file covers roughly +/- 4*Spacing.
                     However, we trust the data bounds if available.
          - R Range: Dependent on TL.
        """
        # 1. Clean Data
        df['X (m)'] = pd.to_numeric(df['X (m)'], errors='coerce')
        df['Ryz (m)'] = pd.to_numeric(df['Ryz (m)'], errors='coerce')
        df['Defect Charge Density (C \cdot cm^{-3})'] = pd.to_numeric(df['Defect Charge Density (C \cdot cm^{-3})'], errors='coerce')
        df = df.dropna()

        if df.empty:
            return None, None, None

        points = df[['X (m)', 'Ryz (m)']].values
        values = df['Defect Charge Density (C \cdot cm^{-3})'].values

        # 2. Determine Physical Bounds from Data (Training Phase)
        # We add a small buffer to ensure points on the edge are included
        x_min_data, x_max_data = points[:, 0].min(), points[:, 0].max()
        r_min_data, r_max_data = points[:, 1].min(), points[:, 1].max()

        # Enforce symmetry for X if it looks centered (e.g. -77.5 to 77.5)
        max_abs_x = max(abs(x_min_data), abs(x_max_data))
        x_bounds = (-max_abs_x, max_abs_x)
        r_bounds = (r_min_data, r_max_data)

        # 3. Create Physical Grid for this sample
        grid_x_phys = np.linspace(x_bounds[0], x_bounds[1], self.W)
        grid_r_phys = np.linspace(r_bounds[0], r_bounds[1], self.H)
        grid_x, grid_r = np.meshgrid(grid_x_phys, grid_r_phys)

        # 4. Interpolate
        grid_z = griddata(points, values, (grid_x, grid_r), method='linear', fill_value=0)
        
        # Replace NaNs (extrapolation) with 0
        grid_z = np.nan_to_num(grid_z)

        return grid_z, x_bounds, r_bounds

class SonosDataset(Dataset):
    def __init__(self, data_dir=".", resolution=(64, 128), normalize=True):
        self.data_dir = data_dir
        self.gridder = SonosGridder(resolution)
        self.normalize = normalize
        self.samples = [] 
        
        fname_pattern = re.compile(r"spacing(\d+)nm-tl(\d+)am-pgm(\d+)v-dcd\.csv")
        files = glob.glob(os.path.join(data_dir, "spacing*nm-tl*am-pgm*v-dcd.csv"))
        
        print(f"Found {len(files)} CSV files. Parsing data...")

        self.stats = {
            'dcd_mean': 0, 'dcd_std': 1,
            'p_mean': np.zeros(4), 'p_std': np.ones(4)
        }
        
        all_dcds = []
        all_params = []

        for fpath in files:
            fname = os.path.basename(fpath)
            match = fname_pattern.search(fname)
            if not match: continue
                
            spacing = float(match.group(1)) # nm
            tl = float(match.group(2))      # angstrom
            pgm = float(match.group(3))     # Volts
            
            timesteps_data = self._parse_ginestra_csv(fpath)
            
            for t_val, df in timesteps_data.items():
                if df.empty: continue
                if len(df) < 10: continue 

                # Interpolate
                grid_dcd, x_b, r_b = self.gridder.interpolate(df, spacing, tl)
                if grid_dcd is None: continue
                
                # Parameters: [Spacing, TL, PGM, Log(Time)]
                log_time = np.log10(t_val + 1e-6) if t_val > 0 else 0
                params = np.array([spacing, tl, pgm, log_time], dtype=np.float32)
                
                self.samples.append({
                    'map': grid_dcd,
                    'params': params
                })
                
                all_dcds.append(grid_dcd)
                all_params.append(params)

        if len(self.samples) == 0:
            print("Warning: No valid data found!")
            return

        # Compute Stats
        if self.normalize:
            all_dcds = np.stack(all_dcds)
            all_params = np.stack(all_params)
            
            self.stats['dcd_mean'] = np.mean(all_dcds)
            self.stats['dcd_std'] = np.std(all_dcds) + 1e-9
            self.stats['p_mean'] = np.mean(all_params, axis=0)
            self.stats['p_std'] = np.std(all_params, axis=0) + 1e-9
            
            print(f"Stats computed on {len(self.samples)} samples.")
            print(f"DCD Mean: {self.stats['dcd_mean']:.2e}, Std: {self.stats['dcd_std']:.2e}")

    def _parse_ginestra_csv(self, fpath):
        with open(fpath, 'r') as f:
            lines = f.readlines()
            
        data_blocks = {}
        current_time = None
        buffer = []
        capture = False
        
        for line in lines:
            line = line.strip()
            if "Time (s)" in line:
                if current_time is not None and buffer:
                    data_blocks[current_time] = self._buffer_to_df(buffer)
                try:
                    current_time = float(line.split(',')[1])
                except:
                    current_time = 0.0
                buffer = []
                capture = False
                
            elif "#DATA" in line:
                capture = True
                continue
            elif "#HEADER" in line:
                capture = False
            elif capture and line:
                buffer.append(line)
                
        if current_time is not None and buffer:
            data_blocks[current_time] = self._buffer_to_df(buffer)
        return data_blocks

    def _buffer_to_df(self, buffer):
        if not buffer: return pd.DataFrame()
        
        # Remove Ginestra's internal header if present in data block
        if "Ryz" in buffer[0] or "Defect" in buffer[0]:
            buffer.pop(0)

        header = "Ryz (m),X (m),Defect Charge Density (C \cdot cm^{-3})"
        csv_str = header + "\n" + "\n".join(buffer)
        try:
            return pd.read_csv(StringIO(csv_str))
        except:
            return pd.DataFrame()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        dcd_map = item['map']
        params = item['params']
        
        H, W = dcd_map.shape
        # Input Channels: 6
        # 0: Spacing (norm)
        # 1: TL (norm)
        # 2: PGM (norm)
        # 3: Time (norm)
        # 4: Grid X (coordinate embedding -1 to 1)
        # 5: Grid R (coordinate embedding 0 to 1)
        
        x_input = np.zeros((6, H, W), dtype=np.float32)
        
        # Normalize scalar params
        norm_params = (params - self.stats['p_mean']) / self.stats['p_std']
        
        # Broadcast scalars
        for c in range(4):
            x_input[c, :, :] = norm_params[c]
            
        # Add Coordinate Channels
        # Use meshgrid for coordinates
        grid_x_norm, grid_r_norm = self.gridder.grid_x_norm, self.gridder.grid_r_norm
        x_input[4, :, :] = grid_x_norm
        x_input[5, :, :] = grid_r_norm
        
        # Target
        if self.normalize:
            norm_dcd = (dcd_map - self.stats['dcd_mean']) / self.stats['dcd_std']
        else:
            norm_dcd = dcd_map
            
        y_target = norm_dcd[np.newaxis, :, :].astype(np.float32)
        
        return torch.from_numpy(x_input), torch.from_numpy(y_target)

    def get_stats(self):
        return self.stats