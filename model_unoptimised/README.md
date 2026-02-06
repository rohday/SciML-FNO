# FNO Model

Fourier Neural Operator for 2D Poisson equation: maps `(a, f) -> u`.

## Architecture

```
(a, f) → Lift → Fourier Layers (×L) → Project → u
```

Each Fourier layer: `H(x) = σ(Wx + SpectralConv(x))`

## Files

| File | Purpose |
|------|---------|
| `config.py` | Model config dataclass |
| `layers.py` | SpectralConv2d layer |
| `fno2d.py` | Main FNO2d model |
| `train.py` | Training script |
| `evaluate.py` | Evaluation + benchmarking |
| `utils.py` | Data loading, checkpointing |
| `plotter.py` | Live training dashboard |

## Usage

### Train
```bash
python model/train.py --data_path data/train.npz --epochs 100 --output_dir checkpoints/
```

### Evaluate
```bash
python model/evaluate.py --checkpoint checkpoints/ --data_path data/test.npz
```

## Key Parameters

| Param | Default | Notes |
|-------|---------|-------|
| `modes` | 12 | Fourier modes kept |
| `width` | 32 | Hidden channels |
| `depth` | 4 | Number of Fourier layers |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-3 | With cosine annealing |

## Input/Output

- **Input**: `a` (diffusion coef) + `f` (source), shape `(B, 2, H, W)`
- **Output**: `u` (solution), shape `(B, 1, H, W)`
- **Loss**: Relative L2 error
