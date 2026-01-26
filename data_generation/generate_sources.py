"""
Source term (f) generation module.

Generates source term fields for the Poisson equation using either:
1. Sum of Gaussian bumps (localized sources)
2. Low-frequency Fourier series (smooth global sources)
"""

import torch
import numpy as np
from typing import Optional, Tuple, List

from config import DataGenConfig


def generate_gaussian_bumps(
    config: DataGenConfig,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a source term as a sum of Gaussian bumps.
    
    f(x,y) = sum_k A_k * exp(-((x-x_k)^2 + (y-y_k)^2) / (2*sigma_k^2))
    
    Args:
        config: Generation configuration
        seed: Optional random seed
        
    Returns:
        torch.Tensor: Source field of shape (H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    
    # Create coordinate grids (normalized [0, 1])
    y_coords = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    x_coords = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    
    # Random number of bumps
    num_bumps = np.random.randint(config.num_bumps_min, config.num_bumps_max + 1)
    
    # Initialize source field
    f = torch.zeros(H, W)
    
    for _ in range(num_bumps):
        # Random center (avoid boundaries)
        cx = np.random.uniform(0.1, 0.9)
        cy = np.random.uniform(0.1, 0.9)
        
        # Random width (as fraction of domain)
        sigma = np.random.uniform(config.bump_sigma_min, config.bump_sigma_max)
        
        # Random amplitude (can be positive or negative)
        amplitude = np.random.uniform(config.bump_amplitude_min, config.bump_amplitude_max)
        if np.random.random() < 0.3:  # 30% chance of negative (sink)
            amplitude = -amplitude
        
        # Compute Gaussian bump
        dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
        bump = amplitude * torch.exp(-dist_sq / (2 * sigma**2))
        
        f = f + bump
    
    return f


def generate_fourier_source(
    config: DataGenConfig,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a source term as a low-frequency Fourier series.
    
    f(x,y) = sum_{m,n} c_mn * sin(m*pi*x) * sin(n*pi*y)
    
    Using sine basis to naturally satisfy zero boundary conditions.
    
    Args:
        config: Generation configuration
        seed: Optional random seed
        
    Returns:
        torch.Tensor: Source field of shape (H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    M = config.fourier_modes
    
    # Create coordinate grids (normalized [0, 1])
    y_coords = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    x_coords = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    
    # Initialize source field
    f = torch.zeros(H, W)
    
    # Generate random coefficients with decay for higher frequencies
    for m in range(1, M + 1):
        for n in range(1, M + 1):
            # Coefficient decays with frequency (1/(m^2 + n^2))
            decay = 1.0 / (m**2 + n**2)
            c_mn = np.random.randn() * decay
            
            # Add Fourier mode
            mode = c_mn * torch.sin(m * np.pi * x_coords) * torch.sin(n * np.pi * y_coords)
            f = f + mode
    
    # Scale to reasonable range
    f = f / f.abs().max() * config.bump_amplitude_max
    
    return f


def generate_source(
    config: DataGenConfig,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a source term field f(x,y) using the configured method.
    
    Args:
        config: Generation configuration (source_method determines which generator)
        seed: Optional random seed
        
    Returns:
        torch.Tensor: Source field of shape (H, W)
    """
    if config.source_method == "gaussian":
        return generate_gaussian_bumps(config, seed)
    elif config.source_method == "fourier":
        return generate_fourier_source(config, seed)
    else:
        raise ValueError(f"Unknown source method: {config.source_method}")


def generate_source_batch(
    config: DataGenConfig,
    batch_size: int,
    base_seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a batch of source term fields.
    
    Args:
        config: Generation configuration
        batch_size: Number of samples to generate
        base_seed: Base random seed
        
    Returns:
        torch.Tensor: Batch of sources of shape (batch_size, H, W)
    """
    sources = []
    
    for i in range(batch_size):
        seed = None if base_seed is None else base_seed + i
        f = generate_source(config, seed=seed)
        sources.append(f)
    
    return torch.stack(sources, dim=0)


def generate_point_sources(
    config: DataGenConfig,
    num_sources: int = 3,
    spread: float = 0.02,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate near-point sources (very localized Gaussians).
    
    Useful for testing and visualization.
    
    Args:
        config: Generation configuration
        num_sources: Number of point sources
        spread: Width of the Gaussians (as fraction of domain)
        seed: Optional random seed
        
    Returns:
        torch.Tensor: Source field of shape (H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    
    # Create coordinate grids
    y_coords = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    x_coords = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    
    f = torch.zeros(H, W)
    
    for _ in range(num_sources):
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        amplitude = np.random.uniform(0.5, 2.0)
        
        dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
        bump = amplitude * torch.exp(-dist_sq / (2 * spread**2))
        f = f + bump
    
    return f


if __name__ == "__main__":
    # Test source generation
    config = DataGenConfig(grid_size=64)
    
    # Gaussian bumps
    f_gauss = generate_gaussian_bumps(config, seed=42)
    print(f"Gaussian source shape: {f_gauss.shape}")
    print(f"Range: [{f_gauss.min():.3f}, {f_gauss.max():.3f}]")
    
    # Fourier source
    config_fourier = DataGenConfig(grid_size=64, source_method="fourier")
    f_fourier = generate_fourier_source(config_fourier, seed=42)
    print(f"Fourier source shape: {f_fourier.shape}")
    print(f"Range: [{f_fourier.min():.3f}, {f_fourier.max():.3f}]")
    
    # Batch
    batch = generate_source_batch(config, batch_size=10, base_seed=0)
    print(f"Batch shape: {batch.shape}")
    
    # Reproducibility
    f1 = generate_source(config, seed=123)
    f2 = generate_source(config, seed=123)
    assert torch.allclose(f1, f2), "Reproducibility failed!"
    print("\n✓ Source generation working correctly")
