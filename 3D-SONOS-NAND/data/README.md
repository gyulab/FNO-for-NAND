# Data Directory

This directory contains training data from Ginestra TCAD simulations of 3D SONOS NAND flash memory devices.

## Data Overview

The dataset consists of **23 CSV files** containing defect charge density (DCD) simulations for various device configurations and operating conditions.

### File Naming Convention

```
spacing{XX}nm-tl{YY}am-pgm{ZZ}v-dcd.csv
```

Where:
- `{XX}`: Cell spacing in nanometers (20, 25, or 30 nm)
- `{YY}`: Tunnel layer thickness in Angstroms (20, 30, or 40 Å)
- `{ZZ}`: Program voltage in Volts (10, 15, or 20 V)

### Parameter Space Coverage

| Parameter | Values | Unit |
|-----------|--------|------|
| **Spacing** | 20, 25, 30 | nm |
| **Tunnel Layer (TL)** | 20, 30, 40 | Å |
| **Program Voltage (PGM)** | 10, 15, 20 | V |

**Total combinations**: 3 × 3 × 3 = 27 possible configurations  
**Available files**: 23 (some combinations may be missing)

## File Format

Each CSV file contains time-series data with multiple timesteps. The structure is:

```
#HEADER
Time (s),<timestamp>
#DATA
Ryz (m),X (m),Defect Charge Density (C \cdot cm^{-3})
<data_point_1>
<data_point_2>
...
#HEADER
Time (s),<next_timestamp>
#DATA
Ryz (m),X (m),Defect Charge Density (C \cdot cm^{-3})
<data_point_1>
...
```

### Column Descriptions

1. **Ryz (m)**: Radial coordinate in the cylindrical geometry (meters)
   - Represents the distance from the center axis
   - Typically ranges from ~4nm to ~17nm depending on tunnel layer thickness

2. **X (m)**: Axial coordinate along the device length (meters)
   - Represents position along the channel
   - Range depends on cell spacing (typically ±4× spacing)

3. **Defect Charge Density (C·cm⁻³)**: Target variable
   - Charge density trapped in defect states
   - Units: Coulombs per cubic centimeter
   - Can be positive or negative

### Time Points

Each file contains multiple timesteps representing the temporal evolution of charge trapping during the programming operation. Time values are typically:
- 0 seconds (initial state)
- Logarithmically spaced from ~1ms to ~10,000s

## Data Characteristics

### Spatial Resolution
- **Scattered points**: Data is provided on an unstructured mesh
- **Typical point count**: 800-2000 points per timestep
- **Grid interpolation**: The `SonosDataset` class interpolates to a regular 64×128 grid

### Physical Domain

The physical domain varies with device parameters:

**X-axis (Axial)**:
- Range: Approximately ±4× Spacing
- Example: For 20nm spacing → X ∈ [-80nm, +80nm]

**R-axis (Radial)**:
- Start: Tunnel layer thickness
- Width: ~13nm (active trapping region)
- Example: For TL=20Å → R ∈ [2nm, 15nm]

### Data Statistics

Typical defect charge density values:
- **Range**: -5 to +5 C·cm⁻³
- **Peak values**: Occur near the gate region (center of X-axis)
- **Symmetry**: Generally symmetric about X=0

## Usage in Training

The `SonosDataset` class in `code/sonos_dataset.py` automatically:

1. **Discovers files**: Scans this directory for matching CSV files
2. **Parses format**: Extracts timesteps and data blocks
3. **Interpolates**: Maps scattered points to regular 64×128 grid
4. **Normalizes**: Computes and applies z-score normalization
5. **Augments**: Adds coordinate embeddings for spatial awareness

### Loading Example

```python
from sonos_dataset import SonosDataset

dataset = SonosDataset(
    data_dir='./data',
    resolution=(64, 128),
    normalize=True
)

print(f"Total samples: {len(dataset)}")
print(f"DCD mean: {dataset.stats['dcd_mean']:.2e}")
print(f"DCD std: {dataset.stats['dcd_std']:.2e}")
```

## Data Preprocessing Pipeline

1. **File Discovery**: Regex matching on filenames
2. **Parameter Extraction**: Parse spacing, TL, PGM from filename
3. **CSV Parsing**: Custom parser handles Ginestra format with headers
4. **Time-series Extraction**: Each timestep becomes a separate training sample
5. **Grid Interpolation**: 
   - Delaunay triangulation of scattered points
   - Linear interpolation to regular grid
   - Fill extrapolated regions with zeros
6. **Normalization**:
   - Parameters: Z-score normalization (mean=0, std=1)
   - DCD: Z-score normalization across all samples
7. **Coordinate Embedding**:
   - X: Normalized to [-1, 1]
   - R: Normalized to [0, 1]

## File Size

- **Individual files**: ~2-3 MB each
- **Total dataset**: ~50 MB
- **Compressed**: Suitable for version control with Git LFS

## Data Quality Notes

- **Missing combinations**: Not all 27 parameter combinations are present
- **Scattered mesh**: Original data is on unstructured grid (requires interpolation)
- **Time coverage**: Varies by file, but typically includes 50+ timesteps
- **Boundary effects**: Edge regions may have interpolation artifacts

## Adding New Data

To add new simulation data:

1. Export from Ginestra in the same CSV format
2. Name file following the convention: `spacing{XX}nm-tl{YY}am-pgm{ZZ}v-dcd.csv`
3. Place in this directory
4. Re-run training - the dataset class will automatically discover it

## Data Provenance

This data was generated using:
- **Simulator**: Ginestra TCAD
- **Device**: 3D SONOS NAND flash memory
- **Physics**: Drift-diffusion with trap-assisted tunneling
- **Geometry**: 2D cylindrical cross-section

For more details, see: `LCM Modeling for SONOS 3D NAND.pdf`
