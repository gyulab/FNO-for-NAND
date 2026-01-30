import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CONFIG = {
    "CSV_PATH": "./csv/full_node_mapping.csv",
    "MAP_DIR": "./data/physics_cache_npy",
    "PLT_DIR": "./data",                 # Folder where .plt files are
    "VG_RANGE": (0, 10),             # Voltage range to interpolate
    "VG_POINTS": 100,                    # Size of the output vector
    "BATCH_SIZE": 16,
    "LR": 1e-3,
    "EPOCHS": 2500,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SAVE_PATH": "checkpoints/iv_net_best.pt"
}

# ==============================================================================
# 1. ROBUST PARSER (Fixed: Strict Dataset Block Parsing)
# ==============================================================================
def parse_plt_iv(plt_path):
    if not os.path.exists(plt_path): return None, None
    try:
        with open(plt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 1. Locate Data Section
        # Find where numerical data starts
        data_start_match = re.search(r'Data\s*\{', content)
        if not data_start_match:
            return None, None
        
        data_start_idx = data_start_match.end()
        header_text = content[:data_start_match.start()]
        
        # 2. Extract Column Names (STRICT)
        # Only look for strings inside the datasets = [ ... ] block
        # This prevents picking up stray quotes from metadata
        dataset_match = re.search(r'datasets\s*=\s*\[(.*?)\]', header_text, re.DOTALL)
        if not dataset_match:
            return None, None
            
        dataset_content = dataset_match.group(1)
        cols = re.findall(r'"([^"]+)"', dataset_content)
        
        # 3. Identify Indices
        vg_idx, id_idx = -1, -1
        
        for idx, col_name in enumerate(cols):
            c_lower = col_name.lower()
            if "gate" in c_lower and "outervoltage" in c_lower:
                vg_idx = idx
            if "drain" in c_lower and "totalcurrent" in c_lower:
                id_idx = idx
                
        if vg_idx == -1 or id_idx == -1:
            return None, None

        if vg_idx == -1 or id_idx == -1:
            return None, None

        # 4. Parse Numerical Data (Flatten -> Reshape)
        data_end_idx = content.find('}', data_start_idx)
        if data_end_idx == -1: data_end_idx = len(content)
        
        data_str = content[data_start_idx:data_end_idx]
        
        # Fast tokenization
        tokens = data_str.split()
        values = []
        for t in tokens:
            try:
                values.append(float(t))
            except ValueError:
                pass
                
        num_cols = len(cols)
        if num_cols == 0: return None, None
        
        num_rows = len(values) // num_cols
        if num_rows == 0: return None, None
        
        # Truncate to full rows and reshape
        arr = np.array(values[:num_rows*num_cols]).reshape(num_rows, num_cols)
        
        return arr[:, vg_idx], np.abs(arr[:, id_idx])

    except Exception as e:
        print(f"DEBUG: Error parsing {plt_path}: {e}")
        return None, None

def interpolate_iv(vg, id_curr):
    try:
        # Sort by Vg
        idx = np.argsort(vg)
        vg, id_curr = vg[idx], id_curr[idx]
        
        # Remove duplicates
        vg, u_idx = np.unique(vg, return_index=True)
        id_curr = id_curr[u_idx]

        if len(vg) < 2: return None

        # Log10 (clip at 1e-15)
        log_id = np.log10(np.maximum(id_curr, 1e-15))

        # Interpolate
        f = interp1d(vg, log_id, kind='linear', fill_value="extrapolate")
        target_vg = np.linspace(CONFIG['VG_RANGE'][0], CONFIG['VG_RANGE'][1], CONFIG['VG_POINTS'])
        return f(target_vg).astype(np.float32)
    except Exception:
        return None

# ==============================================================================
# 2. DATASET
# ==============================================================================
class RetentionDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df.columns = self.df.columns.str.strip()
        self.samples = []
        
        # Filter for Erased State only (ERS)
        self.df = self.df[self.df['State'].str.strip() == 'Erased']
        
        print(f"Indexing Data (checking {len(self.df)} files)...")
        
        valid_count = 0
        skipped_count = 0
        
        for _, row in self.df.iterrows():
            node = int(row['Node'])
            map_path = os.path.join(CONFIG['MAP_DIR'], f"node_{node}.npy")
            plt_path = os.path.join(CONFIG['PLT_DIR'], os.path.basename(str(row['Filename']).strip()))
            
            if not os.path.exists(map_path) or not os.path.exists(plt_path):
                skipped_count += 1
                continue
            
            # Check Parsing
            vg, id_curr = parse_plt_iv(plt_path)
            if vg is None:
                print(f"  [WARN] Parse Failed: Node {node}")
                skipped_count += 1
                continue
                
            # Check Interpolation
            label = interpolate_iv(vg, id_curr)
            if label is None:
                print(f"  [WARN] Interp Failed: Node {node}")
                skipped_count += 1
                continue

            scalars = np.array([
                row['HZO_Thickness'] * 1e3, # nm
                row['Temperature'],         # Kelvin
                np.log10(row['Retention_Time'] + 1.0)
            ], dtype=np.float32)
            
            self.samples.append({
                'map_path': map_path,
                'scalars': scalars,
                'label': label
            })
            valid_count += 1
        
        print(f"Dataset Ready: {valid_count} Valid | {skipped_count} Skipped")
        if valid_count == 0:
            raise ValueError("No valid samples found! Check paths and logic.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        phys_map = np.load(item['map_path'])
        
        # FIX: Handle incorrect broadcasting in cache generation (3, 1, 3, 256, 256)
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
            
        return torch.tensor(phys_map), torch.tensor(item['scalars']), torch.tensor(item['label'])

# ==============================================================================
# 3. MODEL
# ==============================================================================
class ShallowIVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 + 3, 128), nn.ReLU(),
            nn.Linear(128, CONFIG['VG_POINTS'])
        )

    def forward(self, x_map, x_scal):
        # x_scal: [HZO, Temp, Time]
        # 3 scalar physical conditions that are injected into the model alongside the features extracted from the 2D map
        feats = self.enc(x_map).view(x_map.size(0), -1)
        combined = torch.cat([feats, x_scal], dim=1)
        return self.head(combined)

# ==============================================================================
# 4. MAIN
# ==============================================================================
def main():
    if not os.path.exists(CONFIG['MAP_DIR']):
        print(f"ERROR: Map Directory {CONFIG['MAP_DIR']} missing.")
        return

    ds = RetentionDataset(CONFIG['CSV_PATH'])
    loader = DataLoader(ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    model = ShallowIVNet().to(CONFIG['DEVICE'])
    opt = optim.Adam(model.parameters(), lr=CONFIG['LR'])
    crit = nn.MSELoss()
    
    print("--- Starting Training ---")
    best_loss = float('inf')
    
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        for maps, scalars, labels in loader:
            maps, scalars, labels = maps.to(CONFIG['DEVICE']), scalars.to(CONFIG['DEVICE']), labels.to(CONFIG['DEVICE'])
            
            opt.zero_grad()
            preds = model(maps, scalars)
            loss = crit(preds, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.5f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG['SAVE_PATH'])
            
    print(f"Done. Best model saved to {CONFIG['SAVE_PATH']}")

if __name__ == "__main__":
    main()