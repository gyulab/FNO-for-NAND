# FNO for NAND: AI-Powered Semiconductor Device Simulation with NVIDIA PhysicsNeMo

> **Powered by NVIDIA PhysicsNeMo**
> 
> This project leverages the **NVIDIA PhysicsNeMo** framework to implement Fourier Neural Operators (FNO) for high-speed, physics-preserving surrogate modeling of AI-powered Ferroelectric NAND (FeNAND) devices. By utilizing NVIDIA's optimized operator implementations, we can achieve significant acceleration over traditional TCAD tools while maintaining physical accuracy.

## Overview

The `FNO_for_NAND` project aims to accelerate semiconductor device modeling by replacing computationally expensive Technology Computer-Aided Design (TCAD) simulations with Physics-Informed Neural Operators (PINO).

At the core of our approach is **NVIDIA PhysicsNeMo**, which provides the robust, differentiable, and GPU-accelerated spectral operators necessary to solve the complex PDEs governing device physics (Poisson, Continuity, and Charge Trap equations).

## Key Features

- **NVIDIA PhysicsNeMo Integration**:
  - Utilizes `physicsnemo.models.fno.FNO` for highly efficient spectral convolutions.
  - Leverages optimized FFT implementations for training on high-resolution physics fields.

- **Physics-Informed Learning (PINO)**:
  - Enforces the **Poisson Equation** (`ε∇²φ + ρ = 0`) directly within the training loop.
  - Uses **Finite Difference Methods (FDM)** implemented as differentiable convolutions in PyTorch to compute gradients and Laplacians ($\nabla^2\phi$) on the fly, minimizing the PDE residual alongside data loss.

- **High-Fidelity Surrogate Modeling**:
  - Accurate prediction of **Electrostatic Potential**, **Electron Trapped Charge**, and **Current Density** profiles.
  - Specialized high-resolution modeling for critical gate oxide regions.

## Project Structure

- **`2D-FeNAND/`**: 
  - **Code**: Training scripts utilizing `physicsnemo` for 2D Ferroelectric NAND devices.
  - **Results**: Checkpoints and physics-informed animations.
  - **Docs**: `FeNAND-AI-Surrogate.pdf` detailing the methodology.
  
- **`3D-SONOS-NAND/`**:
  - Future development for 3D SONOS NAND architectures, scaling the PhysicsNeMo approach to 3D domains.

## References

- **NVIDIA PhysicsNeMo**: [https://github.com/ NVIDIA/PhysicsNeMo](https://github.com/NVIDIA/PhysicsNeMo) 
