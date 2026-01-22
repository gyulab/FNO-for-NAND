# Checkpoints Directory

This directory contains trained model weights and normalization statistics for the SONOS NAND surrogate model.

## Contents

### `best_model.pth`
**Size**: ~9.5 MB

PyTorch state dictionary containing the trained weights of the Fourier Neural Operator (FNO) model.

**Model Architecture**:
```python
FNO(
    in_channels=6,          # Input features
    out_channels=1,         # Defect charge density
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=2,            # 2D spatial problem
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
    padding=8
)
```

**Total Parameters**: ~2.4 million

**Training Details**:
- Optimizer: AdamW with weight decay 1e-4
- Learning rate: 1e-3 with OneCycleLR scheduler
- Loss function: MSE (Mean Squared Error)
- Training epochs: 1000
- Validation split: 90/10 train/val

**Loading the Model**:
```python
import torch
from physicsnemo.models.fno import FNO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()
```

---

### `stats.pt`
**Size**: ~2 KB

PyTorch dictionary containing normalization statistics computed from the training data.

**Structure**:
```python
{
    'dcd_mean': float,      # Mean of defect charge density
    'dcd_std': float,       # Std dev of defect charge density
    'p_mean': np.array,     # Mean of [Spacing, TL, PGM, Log(Time)]
    'p_std': np.array       # Std dev of [Spacing, TL, PGM, Log(Time)]
}
```

**Parameter Statistics** (typical values):
```python
p_mean = [25.0, 30.0, 15.0, 3.0]  # [nm, Å, V, log10(s)]
p_std  = [5.0,  10.0, 5.0,  2.0]  # [nm, Å, V, log10(s)]
```

**DCD Statistics** (typical values):
```python
dcd_mean = 0.0          # C·cm⁻³ (approximately zero after normalization)
dcd_std  = 2.5e0        # C·cm⁻³
```

**Loading Statistics**:
```python
import torch

stats = torch.load('stats.pt', weights_only=False)

print(f"DCD Mean: {stats['dcd_mean']:.2e}")
print(f"DCD Std:  {stats['dcd_std']:.2e}")
print(f"Param Mean: {stats['p_mean']}")
print(f"Param Std:  {stats['p_std']}")
```

**Using for Normalization**:
```python
import numpy as np

# Normalize input parameters
spacing, tl, pgm, time = 25.0, 30.0, 15.0, 10000.0
log_time = np.log10(time + 1e-6)
raw_params = np.array([spacing, tl, pgm, log_time])
norm_params = (raw_params - stats['p_mean']) / stats['p_std']

# Denormalize model output
pred_norm = model(input_tensor)
pred_real = (pred_norm * stats['dcd_std']) + stats['dcd_mean']
```

---

## File Generation

These files are automatically created by `train_model_sonos.py`:

```bash
python train_model_sonos.py --data_dir ../data --out_dir ../checkpoints
```

The training script:
1. Computes normalization statistics from the training set
2. Saves `stats.pt` before training begins
3. Saves `best_model.pth` whenever validation loss improves
4. Final checkpoint corresponds to the epoch with lowest validation loss

---

## Model Performance

**Typical Metrics** (on validation set):
- **MSE Loss**: ~1e-3 (normalized units)
- **Relative Error**: <5% on average
- **Inference Speed**: ~10ms per prediction (GPU)
- **Speedup vs TCAD**: >1000× faster

---

## Version Control

**Recommended Practice**:
- Track `stats.pt` in Git (small file)
- Use Git LFS for `best_model.pth` (large file)
- Tag checkpoints with version numbers for reproducibility

**Example `.gitattributes`**:
```
checkpoints/*.pth filter=lfs diff=lfs merge=lfs -text
```

---

## Retraining

To retrain from scratch:

1. Delete existing checkpoints:
   ```bash
   rm best_model.pth stats.pt
   ```

2. Run training script:
   ```bash
   cd ../code
   python train_model_sonos.py --epochs 1000
   ```

3. New checkpoints will be generated in this directory

---

## Checkpoint Compatibility

**Important**: The model architecture must match exactly when loading weights. If you modify:
- Number of FNO layers
- Number of modes
- Latent channels
- Input/output channels

You must retrain the model. The checkpoint will not be compatible.

---

## Backup Recommendations

These files represent significant computational investment:
- **Training time**: ~1-2 hours on modern GPU
- **Backup**: Store copies in cloud storage or external drives
- **Validation**: Periodically verify checkpoints load correctly

---

## Troubleshooting

**Issue**: `RuntimeError: Error(s) in loading state_dict`
- **Cause**: Model architecture mismatch
- **Solution**: Ensure FNO initialization matches training configuration

**Issue**: `FileNotFoundError: stats.pt not found`
- **Cause**: Training incomplete or files moved
- **Solution**: Retrain model or restore from backup

**Issue**: Poor inference results
- **Cause**: Using wrong normalization statistics
- **Solution**: Ensure `stats.pt` matches the training run for `best_model.pth`
