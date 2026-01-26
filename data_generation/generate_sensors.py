"""
Sparse sensor sampling module.

Samples random sensor locations from the solution field and records
(x, y, u) readings with optional noise perturbation.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple

from config import DataGenConfig


def sample_sensors(
    u: torch.Tensor,
    config: DataGenConfig,
    seed: Optional[int] = None,
    avoid_boundary: bool = True,
    boundary_margin: int = 2
) -> Dict[str, torch.Tensor]:
    """
    Sample sparse sensor readings from the solution field.
    
    Args:
        u: Solution field of shape (H, W)
        config: Configuration with sensor parameters
        seed: Optional random seed
        avoid_boundary: If True, avoid sampling near boundaries
        boundary_margin: Number of grid points to avoid near boundary
        
    Returns:
        dict with keys:
            'positions': (N, 2) tensor of (x, y) normalized coordinates [0, 1]
            'indices': (N, 2) tensor of (i, j) grid indices
            'values': (N,) tensor of u values at sensor locations
            'values_noisy': (N,) tensor of u values with noise
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    H, W = u.shape
    N = config.num_sensors
    
    # Determine valid sampling region
    if avoid_boundary:
        i_min, i_max = boundary_margin, H - boundary_margin
        j_min, j_max = boundary_margin, W - boundary_margin
    else:
        i_min, i_max = 0, H
        j_min, j_max = 0, W
    
    # Check if we have enough valid locations
    n_valid = (i_max - i_min) * (j_max - j_min)
    if N > n_valid:
        raise ValueError(f"Not enough valid locations ({n_valid}) for {N} sensors")
    
    # Sample random locations without replacement
    valid_i = torch.arange(i_min, i_max)
    valid_j = torch.arange(j_min, j_max)
    
    # Create all valid (i, j) pairs
    ii, jj = torch.meshgrid(valid_i, valid_j, indexing='ij')
    all_indices = torch.stack([ii.flatten(), jj.flatten()], dim=1)
    
    # Random sample without replacement
    perm = torch.randperm(len(all_indices))[:N]
    indices = all_indices[perm]  # (N, 2)
    
    # Get values at sensor locations
    i_idx = indices[:, 0]
    j_idx = indices[:, 1]
    values = u[i_idx, j_idx]
    
    # Add noise
    if config.sensor_noise > 0:
        # Relative noise based on solution magnitude
        noise_std = config.sensor_noise * values.abs().clamp(min=1e-6)
        noise = torch.randn(N) * noise_std
        values_noisy = values + noise
    else:
        values_noisy = values.clone()
    
    # Convert indices to normalized coordinates
    positions = torch.stack([
        i_idx.float() / (H - 1),  # y coordinate (row)
        j_idx.float() / (W - 1),  # x coordinate (col)
    ], dim=1)
    
    return {
        'positions': positions,      # (N, 2) normalized [0, 1]
        'indices': indices,          # (N, 2) integer grid indices
        'values': values,            # (N,) true values
        'values_noisy': values_noisy # (N,) noisy values
    }


def sample_sensors_grid(
    u: torch.Tensor,
    config: DataGenConfig,
    grid_density: int = 8,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Sample sensors on a regular grid pattern.
    
    Useful for visualization and systematic coverage.
    
    Args:
        u: Solution field of shape (H, W)
        config: Configuration
        grid_density: Number of sensors per axis
        seed: Random seed (for noise only)
        
    Returns:
        Same format as sample_sensors()
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    H, W = u.shape
    
    # Create regular grid
    i_grid = torch.linspace(1, H - 2, grid_density).long()
    j_grid = torch.linspace(1, W - 2, grid_density).long()
    
    ii, jj = torch.meshgrid(i_grid, j_grid, indexing='ij')
    indices = torch.stack([ii.flatten(), jj.flatten()], dim=1)
    
    N = len(indices)
    i_idx = indices[:, 0]
    j_idx = indices[:, 1]
    
    values = u[i_idx, j_idx]
    
    # Add noise
    if config.sensor_noise > 0:
        noise_std = config.sensor_noise * values.abs().clamp(min=1e-6)
        noise = torch.randn(N) * noise_std
        values_noisy = values + noise
    else:
        values_noisy = values.clone()
    
    positions = torch.stack([
        i_idx.float() / (H - 1),
        j_idx.float() / (W - 1),
    ], dim=1)
    
    return {
        'positions': positions,
        'indices': indices,
        'values': values,
        'values_noisy': values_noisy
    }


def create_sensor_mask(
    H: int,
    W: int,
    sensor_indices: torch.Tensor
) -> torch.Tensor:
    """
    Create a binary mask indicating sensor locations.
    
    Args:
        H, W: Grid dimensions
        sensor_indices: (N, 2) tensor of (i, j) indices
        
    Returns:
        torch.Tensor: Binary mask of shape (H, W)
    """
    mask = torch.zeros(H, W)
    mask[sensor_indices[:, 0], sensor_indices[:, 1]] = 1.0
    return mask


def create_sensor_field(
    H: int,
    W: int,
    sensor_data: Dict[str, torch.Tensor],
    use_noisy: bool = True
) -> torch.Tensor:
    """
    Create a sparse field with sensor values at their locations, zeros elsewhere.
    
    Args:
        H, W: Grid dimensions
        sensor_data: Output from sample_sensors()
        use_noisy: Whether to use noisy values
        
    Returns:
        torch.Tensor: Sparse sensor field of shape (H, W)
    """
    field = torch.zeros(H, W)
    indices = sensor_data['indices']
    values = sensor_data['values_noisy'] if use_noisy else sensor_data['values']
    
    field[indices[:, 0], indices[:, 1]] = values
    return field


if __name__ == "__main__":
    # Test sensor sampling
    from generate_coefficients import generate_coefficient
    from generate_sources import generate_source
    from solve_poisson import solve_poisson
    
    config = DataGenConfig(grid_size=64, num_sensors=50, sensor_noise=0.01)
    
    # Generate sample data
    a = generate_coefficient(config, seed=42)
    f = generate_source(config, seed=42)
    u = solve_poisson(a, f, config)
    
    # Sample sensors
    sensors = sample_sensors(u, config, seed=42)
    
    print(f"Sensor positions shape: {sensors['positions'].shape}")
    print(f"Sensor indices shape: {sensors['indices'].shape}")
    print(f"Sensor values shape: {sensors['values'].shape}")
    print(f"Values range: [{sensors['values'].min():.4f}, {sensors['values'].max():.4f}]")
    
    # Check noise
    noise_diff = (sensors['values_noisy'] - sensors['values']).abs()
    print(f"Max noise magnitude: {noise_diff.max():.6f}")
    
    # Grid sensors
    grid_sensors = sample_sensors_grid(u, config, grid_density=8, seed=42)
    print(f"\nGrid sensors: {len(grid_sensors['values'])} points")
    
    # Create mask
    mask = create_sensor_mask(64, 64, sensors['indices'])
    print(f"Mask sum: {mask.sum().int().item()} (should be {config.num_sensors})")
    
    print("\n✓ Sensor sampling working correctly")
