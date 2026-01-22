# Code Directory

This directory contains all Python scripts for training, inference, visualization, and optimization of the SONOS NAND flash memory surrogate model.

## Scripts Overview

### Training

#### `train_model_sonos.py`
Trains the Fourier Neural Operator (FNO) model on the SONOS dataset.

**Usage:**
```bash
python train_model_sonos.py --data_dir ../data --epochs 1000 --batch_size 16 --lr 1e-3 --out_dir ../checkpoints
```

**Arguments:**
- `--data_dir`: Directory containing training CSV files (default: `./data`)
- `--epochs`: Number of training epochs (default: 1000)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--out_dir`: Output directory for checkpoints (default: `./checkpoints`)

**Outputs:**
- `checkpoints/best_model.pth`: Trained model weights
- `checkpoints/stats.pt`: Normalization statistics (mean/std for inputs and outputs)

---

### Dataset

#### `sonos_dataset.py`
PyTorch Dataset class for loading and preprocessing SONOS simulation data.

**Key Classes:**
- `SonosGridder`: Interpolates scattered Ginestra outputs onto a fixed computational grid
- `SonosDataset`: Dataset class that loads CSV files and prepares training samples

**Features:**
- Automatic parsing of Ginestra CSV format with time-series data
- Dynamic physical domain calculation based on device parameters
- Grid interpolation from scattered points to regular 64×128 grid
- Data normalization for stable training
- Coordinate embedding for spatial awareness

---

### Inference

#### `inference_single_point_interpolation_sonos.py`
Performs inference for a single set of device parameters and generates a visualization.

**Usage:**
```bash
python inference_single_point_interpolation_sonos.py \
    --spacing 25 \
    --tl 30 \
    --pgm 15 \
    --time 10000 \
    --checkpoint_dir ../checkpoints
```

**Arguments:**
- `--spacing`: Spacing between cells in nm (required)
- `--tl`: Tunnel layer thickness in Angstroms (required)
- `--pgm`: Program voltage in Volts (required)
- `--time`: Time in seconds (required)
- `--checkpoint_dir`: Directory containing model and stats (default: `./checkpoints`)

**Output:**
- `inference_S{spacing}.png`: Contour plot of predicted defect charge density

---

### Visualization

#### `inference_time_sweep.py`
Generates an animated GIF showing the time evolution of defect charge density.

**Usage:**
```bash
python inference_time_sweep.py
```

**Configuration:**
- Fixed parameters: Spacing=25nm, TL=25Å, PGM=15V
- Time range: 0 to 10,000 seconds (50 logarithmically-spaced points)

**Output:**
- `sonos_evolution.gif`: Animated visualization of DCD evolution over time

---

#### `inference_spacing_sweep.py`
Generates an animated GIF showing how DCD changes with varying cell spacing.

**Usage:**
```bash
python inference_spacing_sweep.py --checkpoint_dir ../checkpoints --output_gif spacing_sweep_10ks.gif
```

**Arguments:**
- `--checkpoint_dir`: Directory containing model checkpoints (default: `./checkpoints`)
- `--output_gif`: Output GIF filename (default: `spacing_sweep_10ks.gif`)

**Configuration:**
- Fixed parameters: TL=25Å, PGM=15V, Time=10,000s
- Spacing range: 20nm to 30nm (20 frames)

**Output:**
- `spacing_sweep_10ks.gif`: Animated visualization with dynamically changing axes

---

#### `inference_tl_thickness_sweep.py`
Generates an animated GIF showing how DCD changes with varying tunnel layer thickness.

**Usage:**
```bash
python inference_tl_thickness_sweep.py --checkpoint_dir ../checkpoints --output_gif thickness_sweep_10ks.gif
```

**Arguments:**
- `--checkpoint_dir`: Directory containing model checkpoints (default: `./checkpoints`)
- `--output_gif`: Output GIF filename (default: `thickness_sweep_10ks.gif`)

**Configuration:**
- Fixed parameters: Spacing=20nm, PGM=15V, Time=10,000s
- TL range: 20Å to 40Å (20 frames)

**Output:**
- `thickness_sweep_10ks.gif`: Animated visualization with sliding Y-axis

---

### Optimization

#### `optimize_sonos_lcm.py`
Performs multi-objective optimization to find optimal device parameters that minimize charge and maximize uniformity.

**Usage:**
```bash
python optimize_sonos_lcm.py \
    --checkpoint_dir ../checkpoints \
    --alpha 1.0 \
    --beta 1.0 \
    --gate_length_nm 30.0 \
    --time 10000
```

**Arguments:**
- `--checkpoint_dir`: Directory containing model checkpoints (default: `./checkpoints`)
- `--alpha`: Weight for total charge objective (default: 1.0)
- `--beta`: Weight for uniformity objective (default: 1.0)
- `--gate_length_nm`: Gate length in nm for integration bounds (default: 30.0)
- `--time`: Fixed time point for optimization (default: 10,000s)

**Optimization Variables:**
- Spacing: 20-30 nm
- Tunnel Layer Thickness: 20-40 Å
- Program Voltage: 10-20 V

**Objective Function:**
```
Cost = α × |Q_total| + β × σ(DCD)
```
where Q_total is the integrated charge and σ(DCD) is the standard deviation of charge density.

**Output:**
- `optimal_lcm_solution.png`: Visualization of optimal DCD distribution
- Console output with optimal parameters and objective values

---

## Model Architecture

All inference scripts use the same FNO architecture:

```python
FNO(
    in_channels=6,          # Spacing, TL, PGM, Time, X_grid, R_grid
    out_channels=1,         # Defect Charge Density
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=2,            # 2D spatial problem
    latent_channels=32,
    num_fno_layers=4,
    num_fno_modes=12,
    padding=8
)
```

## Data Flow

1. **Input Preparation**: Parameters are normalized using training statistics
2. **Coordinate Embedding**: Normalized grid coordinates (X, R) are added as channels
3. **FNO Forward Pass**: Model predicts normalized DCD on 64×128 grid
4. **Denormalization**: Output is scaled back to physical units (C·cm⁻³)
5. **Visualization**: Results are plotted on physical domain with proper axes

## Dependencies

```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import differential_evolution
import imageio.v2 as imageio
from physicsnemo.models.fno import FNO
```

## Tips

- **GPU Acceleration**: All scripts automatically use CUDA if available
- **Batch Processing**: For multiple predictions, modify the inference scripts to loop over parameters
- **Custom Visualizations**: Adjust colormap, contour levels, and plot settings in the visualization scripts
- **Optimization Bounds**: Modify the `bounds` parameter in `optimize_sonos_lcm.py` to explore different design spaces
