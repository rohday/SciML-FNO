#!/usr/bin/env python3
"""Main dataset generation script for FNO training data."""

import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
from multiprocessing import Pool
from functools import partial

import torch
import numpy as np
from tqdm import tqdm

from config import DataGenConfig
from generate_coefficients import generate_coefficient
from generate_sources import generate_source
from solve_poisson import solve_poisson
from utils import set_seed, save_dataset_npz, save_dataset_h5, save_metadata, compute_statistics


def generate_single_sample(idx: int, config: DataGenConfig, base_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate single (a, f, u) sample."""
    seed = base_seed + idx
    
    a = generate_coefficient(config, seed=seed)
    f = generate_source(config, seed=seed + 1000000)
    u = solve_poisson(a, f, config)
    
    return a.numpy().astype(np.float32), f.numpy().astype(np.float32), u.numpy().astype(np.float32)


def generate_dataset_split(num_samples: int, config: DataGenConfig, base_seed: int, num_workers: int = 1, desc: str = "Generating") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dataset split (train/val/test)."""
    if num_samples == 0:
        H = config.grid_size
        return (np.empty((0, H, H), dtype=np.float32),) * 3
    
    if num_workers > 1:
        worker_fn = partial(generate_single_sample, config=config, base_seed=base_seed)
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(worker_fn, range(num_samples)), total=num_samples, desc=desc))
    else:
        results = []
        for idx in tqdm(range(num_samples), desc=desc):
            results.append(generate_single_sample(idx, config, base_seed))
    
    a_all = np.stack([r[0] for r in results], axis=0)
    f_all = np.stack([r[1] for r in results], axis=0)
    u_all = np.stack([r[2] for r in results], axis=0)
    
    return a_all, f_all, u_all


def generate_full_dataset(train_samples: int, val_samples: int, test_samples: int, config: DataGenConfig, output_dir: str, format: str = "npz", num_workers: int = 1) -> Dict[str, str]:
    """Generate complete train/val/test dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_fn = save_dataset_npz if format == "npz" else save_dataset_h5
    ext = ".npz" if format == "npz" else ".h5"
    
    output_files = {}
    all_stats = {}
    
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
        
        filepath = output_path / f"{split_name}{ext}"
        save_fn(str(filepath), a, f, u)
        output_files[split_name] = str(filepath)

        stats = compute_statistics(a, f, u)
        all_stats[split_name] = stats
    
    total_elapsed = time.time() - total_start
    total_samples = train_samples + val_samples + test_samples
    
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
    parser = argparse.ArgumentParser(description='Generate FNO training data')
    
    parser.add_argument('--train_samples', type=int, default=1000)
    parser.add_argument('--val_samples', type=int, default=200)
    parser.add_argument('--test_samples', type=int, default=200)
    
    parser.add_argument('--grid_size', type=int, default=64)
    parser.add_argument('--a_min', type=float, default=0.1)
    parser.add_argument('--a_max', type=float, default=3.0)
    parser.add_argument('--source_method', type=str, default='gaussian', choices=['gaussian', 'fourier'])
    
    parser.add_argument('--output', type=str, default='../data/prototype')
    parser.add_argument('--format', type=str, default='npz', choices=['npz', 'h5'])
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--config', type=str, default=None)
    
    args = parser.parse_args()
    
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
    
    print("\n" + "="*60)
    print("FNO DATA GENERATION")
    print("="*60)
    print(f"Grid: {config.grid_size}x{config.grid_size}")
    print(f"Samples: train={args.train_samples}, val={args.val_samples}, test={args.test_samples}")
    print(f"Coefficient: [{config.a_min}, {config.a_max}]")
    print(f"Source: {config.source_method}")
    print(f"Output: {args.output} ({args.format})")
    print(f"Seed: {config.seed}, Workers: {args.num_workers}")
    
    set_seed(config.seed)
    
    return generate_full_dataset(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        config=config,
        output_dir=args.output,
        format=args.format,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
