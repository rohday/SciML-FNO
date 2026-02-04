"""Source term generation using Gaussian bumps or Fourier series."""

import torch
import numpy as np
from typing import Optional

from config import DataGenConfig


def generate_gaussian_bumps(config: DataGenConfig, seed: Optional[int] = None) -> torch.Tensor:
    """Generate source as sum of Gaussian bumps."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    
    y_coords = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    x_coords = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    
    num_bumps = np.random.randint(config.num_bumps_min, config.num_bumps_max + 1)
    f = torch.zeros(H, W)
    
    for _ in range(num_bumps):
        cx = np.random.uniform(0.1, 0.9)
        cy = np.random.uniform(0.1, 0.9)
        sigma = np.random.uniform(config.bump_sigma_min, config.bump_sigma_max)
        amplitude = np.random.uniform(config.bump_amplitude_min, config.bump_amplitude_max)
        if np.random.random() < 0.3:
            amplitude = -amplitude
        
        dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
        bump = amplitude * torch.exp(-dist_sq / (2 * sigma**2))
        f = f + bump
    
    return f


def generate_fourier_source(config: DataGenConfig, seed: Optional[int] = None) -> torch.Tensor:
    """Generate source as low-frequency Fourier series."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    M = config.fourier_modes
    
    y_coords = torch.linspace(0, 1, H).unsqueeze(1).expand(H, W)
    x_coords = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    
    f = torch.zeros(H, W)
    
    for m in range(1, M + 1):
        for n in range(1, M + 1):
            decay = 1.0 / (m**2 + n**2)
            c_mn = np.random.randn() * decay
            mode = c_mn * torch.sin(m * np.pi * x_coords) * torch.sin(n * np.pi * y_coords)
            f = f + mode
    
    f = f / f.abs().max() * config.bump_amplitude_max
    return f


def generate_source(config: DataGenConfig, seed: Optional[int] = None) -> torch.Tensor:
    """Generate source term using configured method."""
    if config.source_method == "gaussian":
        return generate_gaussian_bumps(config, seed)
    elif config.source_method == "fourier":
        return generate_fourier_source(config, seed)
    else:
        raise ValueError(f"Unknown source method: {config.source_method}")


def generate_source_batch(config: DataGenConfig, batch_size: int, base_seed: Optional[int] = None) -> torch.Tensor:
    """Generate batch of source fields."""
    sources = []
    for i in range(batch_size):
        seed = None if base_seed is None else base_seed + i
        f = generate_source(config, seed=seed)
        sources.append(f)
    return torch.stack(sources, dim=0)


def generate_point_sources(config: DataGenConfig, num_sources: int = 3, spread: float = 0.02, seed: Optional[int] = None) -> torch.Tensor:
    """Generate near-point sources (very localized Gaussians)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = config.grid_size, config.grid_size
    
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
    config = DataGenConfig(grid_size=64)
    
    f_gauss = generate_gaussian_bumps(config, seed=42)
    print(f"Gaussian: {f_gauss.shape}, range: [{f_gauss.min():.3f}, {f_gauss.max():.3f}]")
    
    config_fourier = DataGenConfig(grid_size=64, source_method="fourier")
    f_fourier = generate_fourier_source(config_fourier, seed=42)
    print(f"Fourier: {f_fourier.shape}, range: [{f_fourier.min():.3f}, {f_fourier.max():.3f}]")
    
    batch = generate_source_batch(config, batch_size=10, base_seed=0)
    print(f"Batch: {batch.shape}")
    
    f1 = generate_source(config, seed=123)
    f2 = generate_source(config, seed=123)
    assert torch.allclose(f1, f2)
    print("✓ Source generation OK")
