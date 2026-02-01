#!/usr/bin/env python3
"""
Main dataset generation script.

Orchestrates the generation of (a, f, u) samples for training the FNO model.
Supports parallel generation and multiple output formats.

Usage:
    python generate_dataset.py --train_samples 1000 --val_samples 200 --test_samples 200 --output ../data/
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

import torch
import numpy as np
from tqdm import tqdm

from config import DataGenConfig
from generate_coefficients import generate_coefficient
from generate_sources import generate_source
from solve_poisson import solve_poisson
from utils import (
    set_seed,
    save_dataset_npz,
    save_dataset_h5,
    save_metadata,
    compute_statistics,
)


def generate_single_sample(
    idx: int,
    config: DataGenConfig,
    base_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single (a, f, u) sample.
    
    Args:
        idx: Sample index
        config: Generation configuration
        base_seed: Base seed (actual seed = base_seed + idx)
        
    Returns:
        Tuple of (a, f, u) arrays
    """
    seed = base_seed + idx
    
    # Generate fields
    a = generate_coefficient(config, seed=seed)
    f = generate_source(config, seed=seed + 1000000)  # Different seed for f
    
    # Solve PDE
    u = solve_poisson(a, f, config)
    
    # Convert to numpy
    a_np = a.numpy().astype(np.float32)
    f_np = f.numpy().astype(np.float32)
    u_np = u.numpy().astype(np.float32)
    
    return a_np, f_np, u_np


def generate_dataset_split(
    num_samples: int,
    config: DataGenConfig,
    base_seed: int,
    num_workers: int = 1,
    desc: str = "Generating"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a dataset split (train/val/test).
    
    Args:
        num_samples: Number of samples to generate
        config: Generation configuration
        base_seed: Base random seed
        num_workers: Number of parallel workers
        desc: Progress bar description
        
    Returns:
        Tuple of stacked arrays (a, f, u)
    """
    if num_samples == 0:
        H = config.grid_size
        return (
            np.empty((0, H, H), dtype=np.float32),
            np.empty((0, H, H), dtype=np.float32),
            np.empty((0, H, H), dtype=np.float32),
        )
    
    # Generate samples
    if num_workers > 1:
        # Parallel generation
        worker_fn = partial(generate_single_sample, config=config, base_seed=base_seed)
        
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_fn, range(num_samples)),
                total=num_samples,
                desc=desc
            ))
    else:
        # Sequential generation
        results = []
        for idx in tqdm(range(num_samples), desc=desc):
            result = generate_single_sample(idx, config, base_seed)
            results.append(result)
    
    # Stack results
    a_all = np.stack([r[0] for r in results], axis=0)
    f_all = np.stack([r[1] for r in results], axis=0)
    u_all = np.stack([r[2] for r in results], axis=0)
    
    return a_all, f_all, u_all


def generate_full_dataset(
    train_samples: int,
    val_samples: int,
    test_samples: int,
    config: DataGenConfig,
    output_dir: str,
    format: str = "npz",
    num_workers: int = 1
) -> Dict[str, str]:
    """
    Generate complete train/val/test dataset.
    
    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        config: Generation configuration
        output_dir: Output directory path
        format: Output format ("npz" or "h5")
        num_workers: Number of parallel workers
        
    Returns:
        Dictionary of output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_fn = save_dataset_npz if format == "npz" else save_dataset_h5
    ext = ".npz" if format == "npz" else ".h5"
    
    output_files = {}
    all_stats = {}
    
    # Generate each split with different base seeds
    splits = [
        ("train", train_samples, config.seed),
        ("val", val_samples, config.seed + 10000000),
        ("test", test_samples, config.seed + 20000000),
    ]
    
    total_start = time.time()
    
    for split_name, n_samples, base_seed in splits:
        if n_samples == 0:
            continue
        
        print(f"Generating {split_name} split...")
        
        start_time = time.time()
        
        a, f, u = generate_dataset_split(
            num_samples=n_samples,
            config=config,
            base_seed=base_seed,
            num_workers=num_workers,
            desc=f"{split_name}"
        )
        
        elapsed = time.time() - start_time
        print(f"Generated in {elapsed:.1f}s ({n_samples/elapsed:.1f} samples/sec)")
        
        # Save dataset
        filepath = output_path / f"{split_name}{ext}"
        save_fn(str(filepath), a, f, u)
        output_files[split_name] = str(filepath)

        # Compute statistics
        stats = compute_statistics(a, f, u)
        all_stats[split_name] = stats
    
    total_elapsed = time.time() - total_start
    total_samples = train_samples + val_samples + test_samples
    
    # Save metadata
    metadata_path = output_path / "metadata.yaml"
    save_metadata(
        str(metadata_path),
        config,
        stats=all_stats,
        extra={
            'total_samples': total_samples,
            'generation_time_seconds': total_elapsed,
            'format': format,
            'files': list(output_files.values()),
        }
    )
    output_files['metadata'] = str(metadata_path)
    
    print(f"\n{'-'*60}")
    print(f"Generation complete.")
    print(f"{'-'*60}")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data for FNO Poisson surrogate model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset size
    parser.add_argument('--train_samples', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=200,
                        help='Number of validation samples')
    parser.add_argument('--test_samples', type=int, default=200,
                        help='Number of test samples')
    
    # Grid and physics
    parser.add_argument('--grid_size', type=int, default=64,
                        help='Grid resolution (H = W)')
    parser.add_argument('--a_min', type=float, default=0.1,
                        help='Minimum diffusion coefficient')
    parser.add_argument('--a_max', type=float, default=3.0,
                        help='Maximum diffusion coefficient')
    parser.add_argument('--source_method', type=str, default='gaussian',
                        choices=['gaussian', 'fourier'],
                        help='Source term generation method')
    

    
    # Output
    parser.add_argument('--output', type=str, default='../data/prototype',
                        help='Output directory')
    parser.add_argument('--format', type=str, default='npz',
                        choices=['npz', 'h5'],
                        help='Output file format')
    
    # Generation
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers (1 = sequential)')
    
    # Config file (overrides other options)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = DataGenConfig.from_yaml(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = DataGenConfig(
            grid_size=args.grid_size,
            a_min=args.a_min,
            a_max=args.a_max,
            source_method=args.source_method,
            seed=args.seed,
        )
    
    # Print configuration
    print("\n" + "="*60)
    print("FNO DATA GENERATION")
    print("="*60)
    print(f"Grid size: {config.grid_size}x{config.grid_size}")
    print(f"Samples: train={args.train_samples}, val={args.val_samples}, test={args.test_samples}")
    print(f"Coefficient range: [{config.a_min}, {config.a_max}]")
    print(f"Source method: {config.source_method}")
    print(f"Output: {args.output} ({args.format})")
    print(f"Seed: {config.seed}")
    print(f"Workers: {args.num_workers}")
    
    # Generate dataset
    set_seed(config.seed)
    
    output_files = generate_full_dataset(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        config=config,
        output_dir=args.output,
        format=args.format,
        num_workers=args.num_workers,
    )
    
    return output_files


if __name__ == "__main__":
    main()
