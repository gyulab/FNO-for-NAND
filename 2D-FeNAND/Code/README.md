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
Physics-Informed Neural Operator (PINO) training that explicitly enforces the **Poisson Equation**.

**Methodology**:
The model output is constrained by a custom `Poisson` PDE module that calculates residuals within the computational graph:
1. **Un-normalization**: Model outputs (Potential $\phi$, Charge $\rho$) are converted back to physical units.
2. **Finite Difference Calculation**: The Laplacian of the potential $\nabla^2\phi$ is computed using fixed 3x3 finite difference kernels (convolutions).
3. **Residual Computation**: The PDE residual $r$ is calculated as:
   $$ r = \epsilon \nabla^2\phi + \rho $$
   *(Where $\epsilon$ is the permittivity and $\rho$ is the charge density)* used as a regulariation term.
4. **Boundary Masking**: Boundary pixels are masked out to avoid padding artifacts.

- **Arguments**:
  - `--lambda_phys`: Weight of the PDE residual loss (default: 0.1).
  - `--epochs`: Default 500 (increased for convergence).
- **Usage**:
  ```bash
  python train_pino.py --lambda_phys 0.1 --epochs 500
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
