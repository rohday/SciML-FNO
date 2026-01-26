"""
Diffusion coefficient (a) generation module.

Generates smooth, heterogeneous coefficient fields for the Poisson equation
using Gaussian-smoothed random noise.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

from config import DataGenConfig


def gaussian_kernel_2d(sigma: float, kernel_size: Optional[int] = None) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for smoothing.
    
    Args:
        sigma: Standard deviation of the Gaussian
        kernel_size: Size of the kernel (must be odd). If None, computed from sigma.
        
    Returns:
        torch.Tensor: 2D Gaussian kernel of shape (1, 1, kernel_size, kernel_size)
    """
    if kernel_size is None:
        # Kernel size should be at least 6*sigma to capture the Gaussian
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # Create 1D Gaussian
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Create 2D Gaussian via outer product
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
    
    # Reshape for conv2d: (out_channels, in_channels, H, W)
    return gauss_2d.unsqueeze(0).unsqueeze(0)


def smooth_field(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian smoothing to a 2D field.
    
    Args:
        field: Input field of shape (H, W)
        sigma: Gaussian smoothing sigma
        
    Returns:
        torch.Tensor: Smoothed field of shape (H, W)
    """
    kernel = gaussian_kernel_2d(sigma)
    padding = kernel.shape[-1] // 2
    
    # Add batch and channel dimensions for conv2d
    field_4d = field.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution with same padding
    smoothed = F.conv2d(field_4d, kernel, padding=padding)
    
    return smoothed.squeeze(0).squeeze(0)


def generate_coefficient(
    config: DataGenConfig,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a single diffusion coefficient field a(x,y).
    
    The generation process:
    1. Generate white noise on the grid
    2. Apply Gaussian smoothing with random sigma in [sigma_min, sigma_max]
    3. Rescale to [a_min, a_max]
    
    Args:
        config: Generation configuration
        seed: Optional random seed for reproducibility
        
    Returns:
        torch.Tensor: Coefficient field of shape (H, W) with values in [a_min, a_max]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    
    # Step 1: Generate white noise
    noise = torch.randn(H, W)
    
    # Step 2: Random smoothing sigma
    sigma = np.random.uniform(config.sigma_min, config.sigma_max)
    smoothed = smooth_field(noise, sigma)
    
    # Step 3: Rescale to [a_min, a_max]
    # Normalize to [0, 1] first
    min_val = smoothed.min()
    max_val = smoothed.max()
    
    if max_val - min_val > 1e-8:
        normalized = (smoothed - min_val) / (max_val - min_val)
    else:
        # Edge case: constant field
        normalized = torch.ones_like(smoothed) * 0.5
    
    # Scale to [a_min, a_max]
    a = config.a_min + normalized * (config.a_max - config.a_min)
    
    return a


def generate_coefficient_batch(
    config: DataGenConfig,
    batch_size: int,
    base_seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a batch of diffusion coefficient fields.
    
    Args:
        config: Generation configuration
        batch_size: Number of samples to generate
        base_seed: Base random seed (each sample uses base_seed + i)
        
    Returns:
        torch.Tensor: Batch of coefficients of shape (batch_size, H, W)
    """
    coefficients = []
    
    for i in range(batch_size):
        seed = None if base_seed is None else base_seed + i
        a = generate_coefficient(config, seed=seed)
        coefficients.append(a)
    
    return torch.stack(coefficients, dim=0)


def generate_coefficient_layered(
    config: DataGenConfig,
    num_layers: int = 4,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a layered coefficient field (horizontal or vertical layers).
    
    This can model stratified media like geological layers.
    
    Args:
        config: Generation configuration
        num_layers: Number of horizontal layers
        seed: Optional random seed
        
    Returns:
        torch.Tensor: Layered coefficient field of shape (H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    
    # Generate random layer values
    layer_values = torch.rand(num_layers) * (config.a_max - config.a_min) + config.a_min
    
    # Create layered field
    a = torch.zeros(H, W)
    layer_height = H // num_layers
    
    for i in range(num_layers):
        start = i * layer_height
        end = (i + 1) * layer_height if i < num_layers - 1 else H
        a[start:end, :] = layer_values[i]
    
    # Add small smooth perturbation
    noise = torch.randn(H, W) * 0.1
    smoothed_noise = smooth_field(noise, config.sigma_min)
    a = a + smoothed_noise
    
    # Clamp to valid range
    a = torch.clamp(a, config.a_min, config.a_max)
    
    return a


if __name__ == "__main__":
    # Test coefficient generation
    config = DataGenConfig(grid_size=64)
    
    # Single sample
    a = generate_coefficient(config, seed=42)
    print(f"Single coefficient shape: {a.shape}")
    print(f"Range: [{a.min():.3f}, {a.max():.3f}]")
    
    # Batch
    batch = generate_coefficient_batch(config, batch_size=10, base_seed=0)
    print(f"Batch shape: {batch.shape}")
    
    # Reproducibility test
    a1 = generate_coefficient(config, seed=123)
    a2 = generate_coefficient(config, seed=123)
    assert torch.allclose(a1, a2), "Reproducibility failed!"
    print("\n✓ Coefficient generation working correctly")
