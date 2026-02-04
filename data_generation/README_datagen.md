# Data Generation

Generates training data for FNO: solves `-∇·(a∇u) = f` with Dirichlet BC.

## Quick Start

```bash
cd data_generation

# Generate dataset
python generate_dataset.py --train_samples 1000 --val_samples 200 --test_samples 200 --output ../data/

# Visualize
python visualize.py --input ../data/train.npz --output figures/

# Run tests
python -m pytest tests/test_generation.py -v
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | DataGenConfig dataclass |
| `generate_coefficients.py` | Gaussian-smoothed random fields for `a` |
| `generate_sources.py` | Gaussian bumps or Fourier series for `f` |
| `solve_poisson.py` | Finite difference solver with harmonic averaging |
| `generate_dataset.py` | Main CLI for batch generation |
| `visualize.py` | Plotting utilities |
| `utils.py` | I/O, normalization, statistics |

## Output Format

```
output/
├── train.npz    # 'a', 'f', 'u' arrays (N, H, W)
├── val.npz
├── test.npz
└── metadata.yaml
```

## Configuration

Key params in `DataGenConfig`:

| Param | Default | Notes |
|-------|---------|-------|
| `grid_size` | 64 | Square grid resolution |
| `a_min`, `a_max` | 0.1, 3.0 | Coefficient range |
| `source_method` | "gaussian" | or "fourier" |
| `seed` | 42 | Reproducibility |

## Python API

```python
from config import DataGenConfig
from generate_coefficients import generate_coefficient
from generate_sources import generate_source
from solve_poisson import solve_poisson

config = DataGenConfig(grid_size=64)
a = generate_coefficient(config, seed=42)
f = generate_source(config, seed=42)
u = solve_poisson(a, f, config)
```

## Solver Details

5-point stencil with harmonic averaging for variable coefficient:
```
a_{i+1/2,j} = 2 * a_{i,j} * a_{i+1,j} / (a_{i,j} + a_{i+1,j})
```

Sparse system solved with `scipy.sparse.linalg.spsolve`.
