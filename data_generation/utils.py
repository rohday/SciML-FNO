"""Utility functions for data generation."""

import torch
import numpy as np
import h5py
import yaml
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from dataclasses import asdict

from config import DataGenConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_dataset_npz(path: str, a, f, u, sensors_pos=None, sensors_val=None) -> None:
    """Save dataset to NPZ format."""
    def to_numpy(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
        return x
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'a': to_numpy(a).astype(np.float32),
        'f': to_numpy(f).astype(np.float32),
        'u': to_numpy(u).astype(np.float32),
    }
    
    if sensors_pos is not None:
        data['sensors_pos'] = to_numpy(sensors_pos).astype(np.float32)
    if sensors_val is not None:
        data['sensors_val'] = to_numpy(sensors_val).astype(np.float32)
    
    np.savez_compressed(path, **data)


def load_dataset_npz(path: str) -> Dict[str, np.ndarray]:
    """Load dataset from NPZ format."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def save_dataset_h5(path: str, a, f, u, sensors_pos=None, sensors_val=None, compression: str = "gzip") -> None:
    """Save dataset to HDF5 format."""
    def to_numpy(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
        return x
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('a', data=to_numpy(a).astype(np.float32), compression=compression)
        hf.create_dataset('f', data=to_numpy(f).astype(np.float32), compression=compression)
        hf.create_dataset('u', data=to_numpy(u).astype(np.float32), compression=compression)
        
        if sensors_pos is not None:
            hf.create_dataset('sensors_pos', data=to_numpy(sensors_pos).astype(np.float32), compression=compression)
        if sensors_val is not None:
            hf.create_dataset('sensors_val', data=to_numpy(sensors_val).astype(np.float32), compression=compression)


def load_dataset_h5(path: str) -> Dict[str, np.ndarray]:
    """Load dataset from HDF5 format."""
    data = {}
    with h5py.File(path, 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][:]
    return data


def save_metadata(path: str, config: DataGenConfig, stats: Optional[Dict] = None, extra: Optional[Dict] = None) -> None:
    """Save generation metadata to YAML."""
    metadata = {
        'config': asdict(config),
        'stats': stats or {},
        'extra': extra or {},
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def compute_statistics(a: np.ndarray, f: np.ndarray, u: np.ndarray) -> Dict[str, float]:
    """Compute dataset statistics."""
    def field_stats(x, name):
        return {
            f'{name}_min': float(x.min()),
            f'{name}_max': float(x.max()),
            f'{name}_mean': float(x.mean()),
            f'{name}_std': float(x.std()),
        }
    
    stats = {}
    stats.update(field_stats(a, 'a'))
    stats.update(field_stats(f, 'f'))
    stats.update(field_stats(u, 'u'))
    stats['num_samples'] = int(a.shape[0])
    stats['grid_size'] = int(a.shape[1])
    
    return stats


def normalize_field(x: torch.Tensor, method: str = "minmax", stats: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalize field using minmax or standard method."""
    if method == "minmax":
        if stats is None:
            min_val = x.min()
            max_val = x.max()
            stats = {'min': min_val.item(), 'max': max_val.item()}
        else:
            min_val = stats['min']
            max_val = stats['max']
        x_norm = (x - min_val) / (max_val - min_val + 1e-8)
        
    elif method == "standard":
        if stats is None:
            mean = x.mean()
            std = x.std()
            stats = {'mean': mean.item(), 'std': std.item()}
        else:
            mean = stats['mean']
            std = stats['std']
        x_norm = (x - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return x_norm, stats


def denormalize_field(x_norm: torch.Tensor, method: str, stats: Dict[str, float]) -> torch.Tensor:
    """Reverse normalization."""
    if method == "minmax":
        return x_norm * (stats['max'] - stats['min']) + stats['min']
    elif method == "standard":
        return x_norm * stats['std'] + stats['mean']
    else:
        raise ValueError(f"Unknown method: {method}")


class ProgressTracker:
    """Simple progress tracking."""
    
    def __init__(self, total: int, desc: str = "Generating"):
        self.total = total
        self.desc = desc
        self.current = 0
        
    def update(self, n: int = 1) -> None:
        self.current += n
        pct = 100 * self.current / self.total
        print(f"\r{self.desc}: {self.current}/{self.total} ({pct:.1f}%)", end="", flush=True)
        
    def close(self) -> None:
        print()


if __name__ == "__main__":
    print("Testing utilities...")
    
    set_seed(42)
    
    a = np.random.randn(10, 64, 64).astype(np.float32)
    f = np.random.randn(10, 64, 64).astype(np.float32)
    u = np.random.randn(10, 64, 64).astype(np.float32)
    
    save_dataset_npz("/tmp/test_dataset.npz", a, f, u)
    loaded = load_dataset_npz("/tmp/test_dataset.npz")
    assert np.allclose(loaded['a'], a)
    print("✓ NPZ OK")
    
    save_dataset_h5("/tmp/test_dataset.h5", a, f, u)
    loaded_h5 = load_dataset_h5("/tmp/test_dataset.h5")
    assert np.allclose(loaded_h5['a'], a)
    print("✓ HDF5 OK")
    
    stats = compute_statistics(a, f, u)
    print(f"✓ Stats: {len(stats)} fields")
    
    x = torch.randn(5, 64, 64)
    x_norm, norm_stats = normalize_field(x, method="minmax")
    x_denorm = denormalize_field(x_norm, "minmax", norm_stats)
    assert torch.allclose(x, x_denorm, atol=1e-6)
    print("✓ Normalization OK")
    
    print("\n✓ All utility tests passed")
