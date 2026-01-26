# FNO Data Generation Module

**Synthetic Dataset Generator for 2D Heterogeneous Poisson Equation**

This module generates training data for a Fourier Neural Operator (FNO) surrogate model. It produces pairs of input fields (diffusion coefficient `a(x,y)` and source term `f(x,y)`) along with their corresponding PDE solutions `u(x,y)`.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Module Reference](#module-reference)
7. [Data Format](#data-format)
8. [Visualization](#visualization)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

---

## Overview

### The Problem

We solve the **heterogeneous 2D Poisson equation**:

$$-\nabla \cdot (a(x,y) \nabla u(x,y)) = f(x,y)$$

Where:
- **`a(x,y)`** — Spatially varying diffusion coefficient (material property)
- **`f(x,y)`** — Source term (external forcing)
- **`u(x,y)`** — Solution field (the quantity we predict)

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────┐    ┌──────────────────┐                      │
│   │  Generate a(x,y) │    │  Generate f(x,y) │                      │
│   │  (Coefficients)  │    │  (Sources)       │                      │
│   └────────┬─────────┘    └────────┬─────────┘                      │
│            │                       │                                 │
│            └───────────┬───────────┘                                 │
│                        ▼                                             │
│            ┌───────────────────────┐                                 │
│            │   Solve PDE for u     │                                 │
│            │  (Finite Difference)  │                                 │
│            └───────────┬───────────┘                                 │
│                        │                                             │
│            ┌───────────┴───────────┐                                 │
│            ▼                       ▼                                 │
│   ┌─────────────────┐    ┌─────────────────┐                        │
│   │  Sample Sensors │    │   Save Dataset  │                        │
│   │   (Optional)    │    │   (.npz/.h5)    │                        │
│   └─────────────────┘    └─────────────────┘                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Background

### The Heterogeneous Poisson Equation

The equation `−∇·(a∇u) = f` describes diffusion with spatially varying material properties.

**Expanded form in 2D:**

$$-\frac{\partial}{\partial x}\left(a \frac{\partial u}{\partial x}\right) - \frac{\partial}{\partial y}\left(a \frac{\partial u}{\partial y}\right) = f$$

**Physical interpretations:**
- **Heat conduction:** `a` = thermal conductivity, `u` = temperature, `f` = heat source
- **Electrostatics:** `a` = permittivity, `u` = electric potential, `f` = charge density
- **Groundwater flow:** `a` = hydraulic conductivity, `u` = pressure head, `f` = recharge

### Finite Difference Discretization

We discretize on a uniform grid with spacing `h = 1/(N-1)` and use a **5-point stencil** that accounts for variable coefficients:

For interior point `(i,j)`:

```
                    a_{i,j+1/2} * (u_{i,j+1} - u_{i,j})
                                    │
                                    ▼
a_{i-1/2,j} * (u_{i,j} - u_{i-1,j}) ← u_{i,j} → a_{i+1/2,j} * (u_{i+1,j} - u_{i,j})
                                    │
                                    ▼
                    a_{i,j-1/2} * (u_{i,j} - u_{i,j-1})
```

The coefficients at half-points are approximated by **harmonic averaging**:

$$a_{i+1/2,j} = \frac{2 \cdot a_{i,j} \cdot a_{i+1,j}}{a_{i,j} + a_{i+1,j}}$$

This produces a sparse linear system `Au = b` solved with `scipy.sparse.linalg.spsolve`.

### Coefficient Generation: Smoothed Random Fields

1. Generate white noise `n(x,y) ~ N(0,1)` on the grid
2. Apply Gaussian filter with `σ ∈ [3, 8]` pixels
3. Rescale linearly to `[a_min, a_max] = [0.1, 3.0]`

This produces smooth, physically plausible heterogeneous media.

### Source Term Generation

**Method 1: Gaussian Bumps**
$$f(x,y) = \sum_{k=1}^{K} A_k \exp\left(-\frac{(x-x_k)^2 + (y-y_k)^2}{2\sigma_k^2}\right)$$

where `K ∈ [1, 5]`, with random centers, amplitudes, and widths.

**Method 2: Low-Frequency Fourier Series**
$$f(x,y) = \sum_{m=1}^{M} \sum_{n=1}^{N} c_{mn} \sin(m\pi x) \sin(n\pi y)$$

with random coefficients `c_mn` and low frequency cutoff.

---

## Installation

### Dependencies

```bash
pip install torch numpy scipy matplotlib tqdm h5py pyyaml
```

**Required versions:**
- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4

### Verify Installation

```bash
cd data_generation
python -c "from config import DataGenConfig; print('✓ Installation OK')"
```

---

## Quick Start

### Generate Prototype Dataset (1400 samples)

```bash
python generate_dataset.py \
    --train_samples 1000 \
    --val_samples 200 \
    --test_samples 200 \
    --output ../data/prototype/
```

### Generate Single Sample (for testing)

```python
from config import DataGenConfig
from generate_coefficients import generate_coefficient
from generate_sources import generate_source
from solve_poisson import solve_poisson

config = DataGenConfig(grid_size=64)
a = generate_coefficient(config)
f = generate_source(config)
u = solve_poisson(a, f, config)

print(f"a shape: {a.shape}, range: [{a.min():.2f}, {a.max():.2f}]")
print(f"f shape: {f.shape}, range: [{f.min():.2f}, {f.max():.2f}]")
print(f"u shape: {u.shape}, range: [{u.min():.2f}, {u.max():.2f}]")
```

---

## Configuration

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_size` | int | 64 | Grid resolution (H = W) |
| `a_min` | float | 0.1 | Minimum diffusion coefficient |
| `a_max` | float | 3.0 | Maximum diffusion coefficient |
| `sigma_min` | float | 3.0 | Min Gaussian smoothing σ |
| `sigma_max` | float | 8.0 | Max Gaussian smoothing σ |
| `source_method` | str | "gaussian" | Source type: "gaussian" or "fourier" |
| `num_bumps_min` | int | 1 | Min Gaussian bumps (if gaussian) |
| `num_bumps_max` | int | 5 | Max Gaussian bumps (if gaussian) |
| `fourier_modes` | int | 5 | Max Fourier modes (if fourier) |
| `num_sensors` | int | 50 | Sparse sensor count |
| `sensor_noise` | float | 0.01 | Sensor noise std (relative) |
| `seed` | int | 42 | Random seed for reproducibility |

### Using a YAML Config File

```yaml
# config.yaml
grid_size: 64
a_min: 0.1
a_max: 3.0
sigma_min: 3.0
sigma_max: 8.0
source_method: "gaussian"
num_bumps_min: 1
num_bumps_max: 5
num_sensors: 50
sensor_noise: 0.01
seed: 42
```

```bash
python generate_dataset.py --config config.yaml --output ../data/
```

---

## Module Reference

### `config.py`

```python
@dataclass
class DataGenConfig:
    """Central configuration for data generation."""
    grid_size: int = 64
    a_min: float = 0.1
    a_max: float = 3.0
    # ... (see source for all options)
```

### `generate_coefficients.py`

```python
def generate_coefficient(config: DataGenConfig, seed: int = None) -> torch.Tensor:
    """
    Generate a single diffusion coefficient field a(x,y).
    
    Returns:
        torch.Tensor: Shape (H, W), values in [a_min, a_max]
    """
```

### `generate_sources.py`

```python
def generate_source(config: DataGenConfig, seed: int = None) -> torch.Tensor:
    """
    Generate a single source term field f(x,y).
    
    Args:
        config: Generation configuration
        seed: Optional random seed
    
    Returns:
        torch.Tensor: Shape (H, W)
    """
```

### `solve_poisson.py`

```python
def solve_poisson(
    a: torch.Tensor, 
    f: torch.Tensor, 
    config: DataGenConfig
) -> torch.Tensor:
    """
    Solve -∇·(a∇u) = f with Dirichlet BC (u=0 on boundary).
    
    Args:
        a: Diffusion coefficient, shape (H, W)
        f: Source term, shape (H, W)
        config: Configuration with grid parameters
    
    Returns:
        torch.Tensor: Solution u, shape (H, W)
    """
```

### `generate_sensors.py`

```python
def sample_sensors(
    u: torch.Tensor, 
    config: DataGenConfig,
    seed: int = None
) -> dict:
    """
    Sample sparse sensor readings from solution field.
    
    Returns:
        dict: {
            'positions': (N, 2) array of (x, y) coordinates,
            'values': (N,) array of u values (with noise)
        }
    """
```

### `generate_dataset.py`

Main CLI tool for batch generation. See `python generate_dataset.py --help`.

---

## Data Format

### Output Structure

```
output_directory/
├── train.npz       # Training set
├── val.npz         # Validation set
├── test.npz        # Test set
└── metadata.yaml   # Generation config and statistics
```

### NPZ File Contents

Each `.npz` file contains:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `a` | (N, H, W) | float32 | Diffusion coefficients |
| `f` | (N, H, W) | float32 | Source terms |
| `u` | (N, H, W) | float32 | PDE solutions |
| `sensors_pos` | (N, S, 2) | float32 | Sensor (x,y) positions |
| `sensors_val` | (N, S) | float32 | Sensor readings |

Where:
- `N` = number of samples
- `H`, `W` = grid dimensions  
- `S` = number of sensors per sample

### Loading Data

```python
import numpy as np

data = np.load('train.npz')
a = data['a']  # (N, H, W)
f = data['f']  # (N, H, W)
u = data['u']  # (N, H, W)

print(f"Loaded {len(a)} samples with shape {a.shape[1:]}")
```

### PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset

class PoissonDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.a = torch.from_numpy(data['a']).float()
        self.f = torch.from_numpy(data['f']).float()
        self.u = torch.from_numpy(data['u']).float()
    
    def __len__(self):
        return len(self.a)
    
    def __getitem__(self, idx):
        # Stack a and f as 2-channel input
        x = torch.stack([self.a[idx], self.f[idx]], dim=0)  # (2, H, W)
        y = self.u[idx].unsqueeze(0)  # (1, H, W)
        return x, y
```

---

## Visualization

### Visualize Samples

```bash
python visualize.py --input ../data/train.npz --num_samples 5 --output figures/
```

### Programmatic Visualization

```python
from visualize import plot_sample, plot_dataset_statistics

# Single sample
plot_sample(a, f, u, save_path='sample.png')

# Dataset statistics
plot_dataset_statistics('../data/train.npz', save_path='stats.png')
```

### Expected Output

A typical visualization shows three panels:

```
┌─────────────────┬─────────────────┬─────────────────┐
│                 │                 │                 │
│    a(x,y)       │    f(x,y)       │    u(x,y)       │
│  Coefficient    │   Source        │   Solution      │
│                 │                 │                 │
│  [Smooth,       │  [Localized     │  [Satisfies     │
│   0.1 to 3.0]   │   bumps]        │   BC, u=0 edge] │
│                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘
```

---

## Advanced Usage

### Parallel Generation

Use multiple workers for faster generation:

```bash
python generate_dataset.py \
    --train_samples 10000 \
    --num_workers 8 \
    --output ../data/full/
```

### Custom Coefficient Distribution

```python
from generate_coefficients import generate_coefficient_custom

def my_coefficient_fn(grid_size, seed):
    """Custom: layered medium with horizontal bands."""
    a = np.ones((grid_size, grid_size))
    for i in range(0, grid_size, 16):
        a[i:i+8, :] = 2.0
    return a + 0.1 * np.random.randn(grid_size, grid_size)

# Use custom function
a = my_coefficient_fn(64, seed=42)
```

### HDF5 Format (for Large Datasets)

For datasets > 10GB, use HDF5:

```bash
python generate_dataset.py \
    --train_samples 100000 \
    --format h5 \
    --output ../data/large/
```

### Reproducibility

All generation is fully reproducible with seeds:

```bash
# These produce identical datasets
python generate_dataset.py --seed 42 --output run1/
python generate_dataset.py --seed 42 --output run2/

# Verify
diff run1/train.npz run2/train.npz  # Should be identical
```

---

## Troubleshooting

### Common Issues

**Q: Solver fails with "Matrix is singular"**

This can happen if `a(x,y)` has values too close to zero. Ensure `a_min >= 0.1`.

```python
config = DataGenConfig(a_min=0.1)  # ✓ Safe
config = DataGenConfig(a_min=0.0)  # ✗ May cause singularity
```

**Q: Generated `u` has values outside expected range**

Check that `f(x,y)` isn't too large. The solution magnitude scales with source magnitude.

```python
# Normalize source term if needed
f = f / f.abs().max()
```

**Q: Memory error with large grid sizes**

The sparse solver uses O(N²) memory for grid size N. For N > 256:

```bash
# Generate in smaller batches
python generate_dataset.py --batch_size 100 --grid_size 256
```

**Q: Slow generation speed**

1. Use multiprocessing: `--num_workers 8`
2. Reduce grid size for prototyping: `--grid_size 32`
3. Use SSD storage for output

### Validation Checks

Run the test suite to verify installation:

```bash
python -m pytest tests/test_generation.py -v
```

Expected output:
```
tests/test_generation.py::test_coefficient_shape PASSED
tests/test_generation.py::test_coefficient_range PASSED
tests/test_generation.py::test_source_shape PASSED
tests/test_generation.py::test_solver_boundary_conditions PASSED
tests/test_generation.py::test_solver_known_solution PASSED
tests/test_generation.py::test_sensor_sampling PASSED
```

---

## References

1. Li, Z., et al. "Fourier Neural Operator for Parametric Partial Differential Equations." ICLR 2021.
2. LeVeque, R.J. "Finite Difference Methods for Ordinary and Partial Differential Equations." SIAM, 2007.
3. Trefethen, L.N. "Spectral Methods in MATLAB." SIAM, 2000.

---

## License

MIT License. See repository root for details.
