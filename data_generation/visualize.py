"""Visualization utilities for generated data."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List

from config import DataGenConfig


def setup_plotting_style():
    plt.rcParams.update({
        'figure.figsize': (12, 4),
        'figure.dpi': 100,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'image.cmap': 'viridis',
    })


def plot_sample(a, f, u, save_path: Optional[str] = None, title: Optional[str] = None, show: bool = True):
    """Plot single (a, f, u) sample."""
    setup_plotting_style()
    
    def to_np(x):
        if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
        return x
    
    a, f, u = to_np(a), to_np(f), to_np(u)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    im0 = axes[0].imshow(a, origin='lower', cmap='plasma')
    axes[0].set_title(f'Coefficient a [{a.min():.2f}, {a.max():.2f}]')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    vmax_f = max(abs(f.min()), abs(f.max()))
    im1 = axes[1].imshow(f, origin='lower', cmap='RdBu_r', vmin=-vmax_f, vmax=vmax_f)
    axes[1].set_title(f'Source f [{f.min():.2f}, {f.max():.2f}]')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(u, origin='lower', cmap='viridis')
    axes[2].set_title(f'Solution u [{u.min():.4f}, {u.max():.4f}]')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    if title: fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show: plt.show()
    else: plt.close()
    
    return fig


def plot_samples_grid(a_batch, f_batch, u_batch, num_samples: int = 4, save_path: Optional[str] = None, show: bool = True):
    """Plot multiple samples in grid layout."""
    setup_plotting_style()
    
    def to_np(x):
        if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
        return x
    
    a_batch, f_batch, u_batch = to_np(a_batch), to_np(f_batch), to_np(u_batch)
    num_samples = min(num_samples, len(a_batch))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        a, f, u = a_batch[i], f_batch[i], u_batch[i]
        
        im0 = axes[i, 0].imshow(a, origin='lower', cmap='plasma')
        axes[i, 0].set_ylabel(f'Sample {i}')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        vmax_f = max(abs(f.min()), abs(f.max()))
        im1 = axes[i, 1].imshow(f, origin='lower', cmap='RdBu_r', vmin=-vmax_f, vmax=vmax_f)
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        im2 = axes[i, 2].imshow(u, origin='lower', cmap='viridis')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
    
    for j, label in enumerate(['Coefficient a', 'Source f', 'Solution u']):
        axes[0, j].set_title(label)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show: plt.show()
    else: plt.close()
    
    return fig


def plot_dataset_statistics(npz_path: str, save_path: Optional[str] = None, show: bool = True):
    """Plot dataset statistics."""
    setup_plotting_style()
    
    data = np.load(npz_path)
    a, f, u = data['a'], data['f'], data['u']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    axes[0, 0].hist(a.flatten(), bins=100, alpha=0.7, color='blue')
    axes[0, 0].set_title(f'a: μ={a.mean():.3f}, σ={a.std():.3f}')
    
    axes[0, 1].hist(f.flatten(), bins=100, alpha=0.7, color='green')
    axes[0, 1].set_title(f'f: μ={f.mean():.3f}, σ={f.std():.3f}')
    
    axes[0, 2].hist(u.flatten(), bins=100, alpha=0.7, color='orange')
    axes[0, 2].set_title(f'u: μ={u.mean():.3f}, σ={u.std():.3f}')
    
    axes[1, 0].plot(a.mean(axis=(1, 2)), 'b-', alpha=0.7)
    axes[1, 0].set_title('a mean per sample')
    
    axes[1, 1].plot(np.abs(f).max(axis=(1, 2)), 'g-', alpha=0.7)
    axes[1, 1].set_title('|f| max per sample')
    
    axes[1, 2].plot(np.abs(u).max(axis=(1, 2)), 'orange', alpha=0.7)
    axes[1, 2].set_title('|u| max per sample')
    
    plt.suptitle(f'Dataset: {Path(npz_path).name} ({len(a)} samples, {a.shape[1]}x{a.shape[2]})', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show: plt.show()
    else: plt.close()
    
    return fig


def plot_solver_convergence(grid_sizes: List[int] = [16, 32, 64, 128], save_path: Optional[str] = None, show: bool = True):
    """Plot solver convergence with grid refinement."""
    from solve_poisson import verify_solver_accuracy
    
    setup_plotting_style()
    
    errors = []
    for gs in grid_sizes:
        error = verify_solver_accuracy(grid_size=gs, verbose=False)
        errors.append(error)
        print(f"Grid {gs}: error = {error:.6e}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(grid_sizes, errors, 'bo-', markersize=8, linewidth=2, label='Numerical error')
    
    h_ref = np.array(grid_sizes)
    c = errors[0] * grid_sizes[0]**2
    ref_errors = c / h_ref**2
    ax.loglog(grid_sizes, ref_errors, 'r--', alpha=0.7, label='O(h²) reference')
    
    ax.set_xlabel('Grid size N')
    ax.set_ylabel('Relative L2 error')
    ax.set_title('Solver Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show: plt.show()
    else: plt.close()
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        print("Running visualization tests...")
        
        from generate_coefficients import generate_coefficient
        from generate_sources import generate_source
        from solve_poisson import solve_poisson
        
        config = DataGenConfig(grid_size=64)
        
        a = generate_coefficient(config, seed=42)
        f = generate_source(config, seed=42)
        u = solve_poisson(a, f, config)
        
        plot_sample(a, f, u, save_path='/tmp/test_sample.png', show=False)
        print("✓ Sample plot OK")
        
        a_batch = torch.stack([generate_coefficient(config, seed=i) for i in range(4)])
        f_batch = torch.stack([generate_source(config, seed=i) for i in range(4)])
        u_batch = torch.stack([solve_poisson(a_batch[i], f_batch[i], config) for i in range(4)])
        
        plot_samples_grid(a_batch, f_batch, u_batch, num_samples=4, save_path='/tmp/test_grid.png', show=False)
        print("✓ Grid plot OK")
    
    elif args.input:
        if args.output: Path(args.output).mkdir(parents=True, exist_ok=True)
        
        data = np.load(args.input)
        
        plot_samples_grid(
            data['a'], data['f'], data['u'],
            num_samples=args.num_samples,
            save_path=f"{args.output}/samples.png" if args.output else None
        )
        
        plot_dataset_statistics(
            args.input,
            save_path=f"{args.output}/statistics.png" if args.output else None
        )
    else:
        print("Use --test or --input <file.npz>")
