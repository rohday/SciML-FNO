"""
Configuration module for FNO data generation.

This module provides a central dataclass for all generation parameters,
supporting both programmatic configuration and YAML file loading.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import yaml
from pathlib import Path


@dataclass
class DataGenConfig:
    """
    Central configuration for Poisson equation data generation.
    
    Attributes:
        grid_size: Resolution of the square grid (H = W = grid_size)
        
        # Coefficient (a) parameters
        a_min: Minimum value for diffusion coefficient
        a_max: Maximum value for diffusion coefficient
        sigma_min: Minimum Gaussian smoothing sigma (in pixels)
        sigma_max: Maximum Gaussian smoothing sigma (in pixels)
        
        # Source (f) parameters
        source_method: Generation method - "gaussian" (bumps) or "fourier" (low-freq)
        num_bumps_min: Minimum number of Gaussian bumps (if method="gaussian")
        num_bumps_max: Maximum number of Gaussian bumps (if method="gaussian")
        bump_sigma_min: Minimum bump width as fraction of grid size
        bump_sigma_max: Maximum bump width as fraction of grid size
        bump_amplitude_min: Minimum bump amplitude
        bump_amplitude_max: Maximum bump amplitude
        fourier_modes: Maximum Fourier mode index (if method="fourier")
        
        # Sensor parameters
        num_sensors: Number of sparse sensor locations per sample
        sensor_noise: Relative noise std for sensor readings
        
        # Generation parameters
        seed: Base random seed for reproducibility
        dtype: Data type for tensors ("float32" or "float64")
    """
    
    # Grid parameters
    grid_size: int = 64
    
    # Coefficient (a) parameters
    a_min: float = 0.1
    a_max: float = 3.0
    sigma_min: float = 3.0
    sigma_max: float = 8.0
    
    # Source (f) parameters
    source_method: Literal["gaussian", "fourier"] = "gaussian"
    num_bumps_min: int = 1
    num_bumps_max: int = 5
    bump_sigma_min: float = 0.05  # Fraction of grid size
    bump_sigma_max: float = 0.15  # Fraction of grid size
    bump_amplitude_min: float = 0.5
    bump_amplitude_max: float = 2.0
    fourier_modes: int = 5
    
    # Sensor parameters
    num_sensors: int = 50
    sensor_noise: float = 0.01
    
    # Generation parameters
    seed: int = 42
    dtype: Literal["float32", "float64"] = "float32"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.grid_size > 0, "grid_size must be positive"
        assert self.a_min > 0, "a_min must be positive (prevents singular matrix)"
        assert self.a_max > self.a_min, "a_max must be greater than a_min"
        assert self.sigma_min > 0, "sigma_min must be positive"
        assert self.sigma_max >= self.sigma_min, "sigma_max must be >= sigma_min"
        assert self.num_bumps_min >= 1, "num_bumps_min must be at least 1"
        assert self.num_bumps_max >= self.num_bumps_min, "num_bumps_max must be >= num_bumps_min"
        assert self.num_sensors >= 0, "num_sensors must be non-negative"
        assert self.sensor_noise >= 0, "sensor_noise must be non-negative"
        assert self.source_method in ("gaussian", "fourier"), "Invalid source_method"
    
    @classmethod
    def from_yaml(cls, path: str) -> "DataGenConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            DataGenConfig instance with loaded parameters
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Output path for YAML file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def get_grid_spacing(self) -> float:
        """
        Get the grid spacing h for a unit domain [0, 1]^2.
        
        Returns:
            Grid spacing h = 1 / (grid_size - 1)
        """
        return 1.0 / (self.grid_size - 1)
    
    def get_coordinates(self):
        """
        Get x and y coordinate arrays for the grid.
        
        Returns:
            Tuple of (x, y) 1D arrays of shape (grid_size,)
        """
        import torch
        coords = torch.linspace(0, 1, self.grid_size)
        return coords, coords


# Default configurations for common use cases
PROTOTYPE_CONFIG = DataGenConfig(
    grid_size=64,
    seed=42,
)

FULL_CONFIG = DataGenConfig(
    grid_size=64,
    seed=42,
)

HIGHRES_CONFIG = DataGenConfig(
    grid_size=128,
    seed=42,
)


if __name__ == "__main__":
    # Test configuration
    config = DataGenConfig()
    print("Default configuration:")
    print(f"  Grid size: {config.grid_size}")
    print(f"  a range: [{config.a_min}, {config.a_max}]")
    print(f"  sigma range: [{config.sigma_min}, {config.sigma_max}]")
    print(f"  Source method: {config.source_method}")
    print(f"  Grid spacing: {config.get_grid_spacing():.4f}")
    
    # Test YAML export/import
    config.to_yaml("/tmp/test_config.yaml")
    loaded = DataGenConfig.from_yaml("/tmp/test_config.yaml")
    assert config.to_dict() == loaded.to_dict(), "YAML roundtrip failed"
    print("\n✓ Configuration module working correctly")
