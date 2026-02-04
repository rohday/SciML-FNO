"""Data generation configuration."""

from dataclasses import dataclass, asdict
from typing import Literal
import yaml
from pathlib import Path


@dataclass
class DataGenConfig:
    """Config for Poisson equation data generation."""
    
    grid_size: int = 64
    
    # Coefficient (a) params
    a_min: float = 0.1
    a_max: float = 3.0
    sigma_min: float = 3.0
    sigma_max: float = 8.0
    
    # Source (f) params
    source_method: Literal["gaussian", "fourier"] = "gaussian"
    num_bumps_min: int = 1
    num_bumps_max: int = 5
    bump_sigma_min: float = 0.05
    bump_sigma_max: float = 0.15
    bump_amplitude_min: float = 0.5
    bump_amplitude_max: float = 2.0
    fourier_modes: int = 5
    
    seed: int = 42
    dtype: Literal["float32", "float64"] = "float32"
    
    def __post_init__(self):
        assert self.grid_size > 0
        assert self.a_min > 0
        assert self.a_max > self.a_min
        assert self.sigma_min > 0
        assert self.sigma_max >= self.sigma_min
        assert self.num_bumps_min >= 1
        assert self.num_bumps_max >= self.num_bumps_min
        assert self.source_method in ("gaussian", "fourier")
    
    @classmethod
    def from_yaml(cls, path: str) -> "DataGenConfig":
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def get_grid_spacing(self) -> float:
        return 1.0 / (self.grid_size - 1)
    
    def get_coordinates(self):
        import torch
        coords = torch.linspace(0, 1, self.grid_size)
        return coords, coords


PROTOTYPE_CONFIG = DataGenConfig(grid_size=64, seed=42)
FULL_CONFIG = DataGenConfig(grid_size=64, seed=42)
HIGHRES_CONFIG = DataGenConfig(grid_size=128, seed=42)


if __name__ == "__main__":
    config = DataGenConfig()
    print(f"Grid: {config.grid_size}, a: [{config.a_min}, {config.a_max}], method: {config.source_method}")
    
    config.to_yaml("/tmp/test_config.yaml")
    loaded = DataGenConfig.from_yaml("/tmp/test_config.yaml")
    assert config.to_dict() == loaded.to_dict()
    print("✓ Config OK")
