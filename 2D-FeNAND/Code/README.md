# 2D-FeNAND Code Documentation

This directory contains Python scripts for training and evaluating Neural Operators for 2D FeNAND devices.

## Requirements
- Python 3.8+
- PyTorch
- NumPy, Pandas, SciPy
- `physicsnemo` (NVIDIA Physics-Informed Neural Operator library)

## Scripts

### 1. Training

#### `train_model.py`
Standard FNO training using MSE Loss.
- **Inputs**: HZO Thickness, Temperature, Retention Time.
- **Outputs**: Electrostatic Potential, eTrapped Charge, Electric Field.
- **Usage**:
  ```bash
  python train_model.py --csv ./csv/node_mapping.csv --epochs 100
  ```

#### `train_pino.py`
Physics-Informed Neural Operator (PINO) training.
- **Features**: Adds a Physics-Guided Loss (H1 Sobolev Loss) to enforce that the Electric Field matches the gradient of the Potential.
- **Arguments**: `--lambda_phys` controls the weight of the physics loss.
- **Usage**:
  ```bash
  python train_pino.py --lambda_phys 0.1
  ```

#### `train_eTrappedCharge.py`
Specialized training for Electron Trapped Charge in the Gate Region (`R.Gatesn2`).
- **Purpose**: Focuses high-resolution training on the critical gate oxide region.
- **Usage**:
  ```bash
  python train_eTrappedCharge.py --out_dir checkpoints_gate
  ```

### 2. Inference & Visualization

#### `inference_eTrappedCharge.py`
Generates animations of trapped charge evolution over temperature sweeps.
- **Output**: GIFs saved to `animations/`.
- **Usage**:
  ```bash
  python inference_eTrappedCharge.py
  ```
