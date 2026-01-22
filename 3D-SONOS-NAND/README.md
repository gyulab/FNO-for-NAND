# 3D-SONOS-NAND: Machine Learning for SONOS NAND Flash Memory Modeling

This project implements a machine learning-based surrogate model for predicting defect charge density (DCD) in 3D SONOS (Silicon-Oxide-Nitride-Oxide-Silicon) NAND flash memory devices. The model uses a Fourier Neural Operator (FNO) to accelerate physics-based simulations and enable rapid design space exploration.

## Overview

The project trains a neural network to predict the spatial distribution of defect charge density in SONOS structures as a function of:
- **Spacing** (nm): Distance between memory cells
- **Tunnel Layer Thickness (TL)** (Angstroms): Thickness of the tunnel oxide layer
- **Program Voltage (PGM)** (Volts): Applied programming voltage
- **Time** (seconds): Duration of the programming operation

The trained model can perform inference orders of magnitude faster than traditional TCAD simulations while maintaining high accuracy.

## Project Structure

```
3D-SONOS-NAND/
├── code/                   # Python scripts for training and inference
├── data/                   # Training data from Ginestra simulations (Hidden due to EULA/NDA Agreement)
├── checkpoints/            # Trained model weights and normalization statistics
├── results/                # Generated visualizations and optimization results
└── LCM Modeling for SONOS 3D NAND.pdf  # Technical documentation
```

## Quick Start

### 1. Training the Model

```bash
cd code
python train_model_sonos.py --data_dir ../data --epochs 1000 --batch_size 16
```

### 2. Running Inference

Single point prediction:
```bash
python inference_single_point_interpolation_sonos.py --spacing 25 --tl 30 --pgm 15 --time 10000
```

### 3. Generating Visualizations

Time evolution GIF:
```bash
python inference_time_sweep.py
```

Spacing sweep GIF:
```bash
python inference_spacing_sweep.py
```

Thickness sweep GIF:
```bash
python inference_tl_thickness_sweep.py
```

### 4. Design Optimization

Find optimal device parameters:
```bash
python optimize_sonos_lcm.py --alpha 1.0 --beta 1.0
```

## Key Features

- **Fast Surrogate Modeling**: FNO-based neural network replaces expensive TCAD simulations
- **Multi-Parameter Inference**: Predicts DCD for arbitrary combinations of design parameters
- **Design Optimization**: Automated search for optimal device configurations
- **Visualization Tools**: Generate animations showing parameter sweeps and time evolution
- **Cylindrical Geometry**: Handles 2D cylindrical coordinate system (R, X)

## Requirements

- Python 3.8+
- PyTorch
- physicsnemo (FNO implementation)
- NumPy, Pandas, Matplotlib
- SciPy (for interpolation)
- imageio (for GIF generation)

## Data Format

Training data consists of CSV files from Ginestra TCAD simulations with the naming convention:
```
spacing{XX}nm-tl{YY}am-pgm{ZZ}v-dcd.csv
```

Each file contains time-series data with columns:
- `Ryz (m)`: Radial coordinate
- `X (m)`: Axial coordinate  
- `Defect Charge Density (C·cm⁻³)`: Target variable

## Model Architecture

- **Input**: 6 channels (Spacing, TL, PGM, Time, X_grid, R_grid)
- **Output**: 1 channel (Defect Charge Density)
- **Resolution**: 64 × 128 (R × X)
- **Architecture**: Fourier Neural Operator with 4 layers, 12 modes
- **Training**: AdamW optimizer with OneCycleLR scheduler

## Citation
- A. Padovani et al., "Understanding and Variability of Lateral Charge Migration in 3D CT-NAND Flash with and Without Band-Gap Engineered Barriers," 2019 IEEE International Reliability Physics Symposium (IRPS), Monterey, CA, USA, 2019, pp. 1-8, doi: 10.1109/IRPS.2019.8720566.
