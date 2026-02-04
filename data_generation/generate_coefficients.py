"""Diffusion coefficient generation using Gaussian-smoothed noise."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from config import DataGenConfig


def gaussian_kernel_2d(sigma: float, kernel_size: Optional[int] = None) -> torch.Tensor:
    """Create 2D Gaussian kernel for smoothing."""
    if kernel_size is None:
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
    return gauss_2d.unsqueeze(0).unsqueeze(0)


def smooth_field(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian smoothing."""
    kernel = gaussian_kernel_2d(sigma)
    padding = kernel.shape[-1] // 2
    field_4d = field.unsqueeze(0).unsqueeze(0)
    smoothed = F.conv2d(field_4d, kernel, padding=padding)
    return smoothed.squeeze(0).squeeze(0)


def generate_coefficient(config: DataGenConfig, seed: Optional[int] = None) -> torch.Tensor:
    """Generate coefficient field a(x,y) in [a_min, a_max]."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    
    noise = torch.randn(H, W)
    sigma = np.random.uniform(config.sigma_min, config.sigma_max)
    smoothed = smooth_field(noise, sigma)
    
    min_val = smoothed.min()
    max_val = smoothed.max()
    
    if max_val - min_val > 1e-8:
        normalized = (smoothed - min_val) / (max_val - min_val)
    else:
        normalized = torch.ones_like(smoothed) * 0.5
    
    a = config.a_min + normalized * (config.a_max - config.a_min)
    return a


def generate_coefficient_batch(config: DataGenConfig, batch_size: int, base_seed: Optional[int] = None) -> torch.Tensor:
    """Generate batch of coefficient fields."""
    coefficients = []
    for i in range(batch_size):
        seed = None if base_seed is None else base_seed + i
        a = generate_coefficient(config, seed=seed)
        coefficients.append(a)
    return torch.stack(coefficients, dim=0)


def generate_coefficient_layered(config: DataGenConfig, num_layers: int = 4, seed: Optional[int] = None) -> torch.Tensor:
    """Generate layered coefficient field (horizontal bands)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    
    layer_values = torch.rand(num_layers) * (config.a_max - config.a_min) + config.a_min
    a = torch.zeros(H, W)
    layer_height = H // num_layers
    
    for i in range(num_layers):
        start = i * layer_height
        end = (i + 1) * layer_height if i < num_layers - 1 else H
        a[start:end, :] = layer_values[i]
    
    noise = torch.randn(H, W) * 0.1
    smoothed_noise = smooth_field(noise, config.sigma_min)
    a = torch.clamp(a + smoothed_noise, config.a_min, config.a_max)
    
    return a


if __name__ == "__main__":
    config = DataGenConfig(grid_size=64)
    
    a = generate_coefficient(config, seed=42)
    print(f"Single: {a.shape}, range: [{a.min():.3f}, {a.max():.3f}]")
    
    batch = generate_coefficient_batch(config, batch_size=10, base_seed=0)
    print(f"Batch: {batch.shape}")
    
    a1 = generate_coefficient(config, seed=123)
    a2 = generate_coefficient(config, seed=123)
    assert torch.allclose(a1, a2)
    print("✓ Coefficient generation OK")
