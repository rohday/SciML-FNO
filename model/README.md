# FNO Model Module

**Fourier Neural Operator for 2D Poisson Equation**

This module implements a Fourier Neural Operator (FNO) that learns the mapping from input fields `(a, f)` to the PDE solution `u`.

---

## Architecture Overview

The FNO architecture consists of three stages:

```
Input (a, f)     Lifting        Fourier Layers (×L)       Projection      Output u
[2, H, W]    →   [d, H, W]   →   [d, H, W] × L layers   →   [1, H, W]   →   [H, W]
                    P                   H                       Q
```

- **Purpose:** Project input channels to higher-dimensional latent space
- **Implementation:** Point-wise 1×1 convolution or MLP
- **Input:** 2 channels (coefficient `a` + source `f`)
- **Output:** `hidden_channels` (default: 64)

### 2. Fourier Layers (H) — The Core
Each Fourier layer performs:
```
H(x) = σ(Wx + K(x))
```

Where:
- **W:** Point-wise linear transformation (1×1 conv)
- **K:** Spectral convolution operator
  1. FFT: Transform to frequency domain
  2. Multiply by learnable complex weights (only for low-frequency modes)
  3. IFFT: Transform back to spatial domain
- **σ:** Activation function (GELU)

**Key insight:** By truncating high-frequency modes, FNO learns only the low-frequency structure, which is computationally efficient and acts as a regularizer.

### 3. Projection Layer (Q)
- **Purpose:** Map latent representation back to output
- **Implementation:** Point-wise 1×1 convolution or MLP
- **Input:** `hidden_channels`
- **Output:** 1 channel (solution `u`)

---

## Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_channels` | 2 | Input channels: `a` (coefficient) + `f` (source) |
| `out_channels` | 1 | Output channels: `u` (solution) |
| `hidden_channels` | 64 | Width of hidden layers |
| `num_layers` | 4 | Number of Fourier layers |
| `modes_x` | 12 | Fourier modes kept in x-direction |
| `modes_y` | 12 | Fourier modes kept in y-direction |
| `activation` | GELU | Activation function |
| `padding` | 9 | Zero-padding for non-periodic BC |

### Parameter Counts (Approximate)

| Configuration | Parameters | Notes |
|---------------|------------|-------|
| Small (d=32, L=4) | ~500K | Fast prototyping |
| Default (d=64, L=4) | ~2M | Standard training |
| Large (d=128, L=6) | ~8M | High accuracy |

---

## Input/Output Specification

### Input
```python
x = torch.stack([a, f], dim=1)  # Shape: (batch, 2, H, W)
```
- `a`: Diffusion coefficient field, normalized to [0, 1]
- `f`: Source term field, normalized

### Output
```python
u = model(x)  # Shape: (batch, 1, H, W)
```
- `u`: Predicted solution field

---

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-3 | Initial LR (with cosine decay) |
| `batch_size` | 32 | Batch size |
| `epochs` | 500 | Training epochs |
| `weight_decay` | 1e-4 | L2 regularization |
| `scheduler` | CosineAnnealing | LR scheduler |
| `loss` | Relative L2 | Loss function |

### Loss Function
**Relative L2 Error:**
$$\mathcal{L} = \frac{\|u_{pred} - u_{true}\|_2}{\|u_{true}\|_2}$$

This is scale-invariant and matches the evaluation metric.

---

## Resolution Invariance

A key property of FNO is **discretization invariance**:
- Train on 64×64 grid
- Evaluate on 128×128 or 256×256 with **zero-shot generalization**

This works because FNO learns the *continuous operator*, not a fixed-resolution mapping.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | Model configuration dataclass |
| `layers.py` | Spectral convolution and Fourier layer implementations |
| `fno2d.py` | Main FNO2D model class |
| `train.py` | Training script with logging |
| `evaluate.py` | Evaluation and metrics |
| `utils.py` | Data loading, normalization, checkpointing |

---

## Quick Start

### Training
```bash
python train.py \
    --data_dir ../data/prototype \
    --epochs 100 \
    --hidden_channels 64 \
    --num_layers 4 \
    --output checkpoints/
```

### Evaluation
```bash
python evaluate.py \
    --checkpoint checkpoints/best.pth \
    --data_dir ../data/prototype \
    --split test
```

---

## References

1. Li, Z., et al. "Fourier Neural Operator for Parametric Partial Differential Equations." ICLR 2021.
2. [neuraloperator library](https://github.com/neuraloperator/neuraloperator)
3. [NVIDIA Modulus](https://developer.nvidia.com/modulus)
