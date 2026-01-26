"""
Visualization utilities for generated data.

Provides plotting functions for coefficient, source, and solution fields,
as well as dataset statistics visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Union, List, Dict

from config import DataGenConfig


def setup_plotting_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.rcParams.update({
        'figure.figsize': (12, 4),
        'figure.dpi': 100,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'image.cmap': 'viridis',
    })


def plot_sample(
    a: Union[torch.Tensor, np.ndarray],
    f: Union[torch.Tensor, np.ndarray],
    u: Union[torch.Tensor, np.ndarray],
    sensors: Optional[Dict] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot a single (a, f, u) sample as three panels.
    
    Args:
        a: Diffusion coefficient, shape (H, W)
        f: Source term, shape (H, W)
        u: Solution, shape (H, W)
        sensors: Optional sensor data dict from sample_sensors()
        save_path: Optional path to save figure
        title: Optional figure title
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    setup_plotting_style()
    
    # Convert to numpy
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    a, f, u = to_np(a), to_np(f), to_np(u)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot coefficient a
    im0 = axes[0].imshow(a, origin='lower', cmap='plasma')
    axes[0].set_title(f'Coefficient a(x,y)\n[{a.min():.2f}, {a.max():.2f}]')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Plot source f
    vmax_f = max(abs(f.min()), abs(f.max()))
    im1 = axes[1].imshow(f, origin='lower', cmap='RdBu_r', vmin=-vmax_f, vmax=vmax_f)
    axes[1].set_title(f'Source f(x,y)\n[{f.min():.2f}, {f.max():.2f}]')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Plot solution u
    im2 = axes[2].imshow(u, origin='lower', cmap='viridis')
    axes[2].set_title(f'Solution u(x,y)\n[{u.min():.4f}, {u.max():.4f}]')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Overlay sensors if provided
    if sensors is not None:
        indices = sensors['indices']
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        axes[2].scatter(indices[:, 1], indices[:, 0], c='red', s=10, alpha=0.7, 
                       marker='x', label=f'{len(indices)} sensors')
        axes[2].legend(loc='upper right')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_samples_grid(
    a_batch: Union[torch.Tensor, np.ndarray],
    f_batch: Union[torch.Tensor, np.ndarray],
    u_batch: Union[torch.Tensor, np.ndarray],
    num_samples: int = 4,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot multiple samples in a grid layout.
    
    Args:
        a_batch, f_batch, u_batch: Batches of shape (N, H, W)
        num_samples: Number of samples to plot
        save_path: Optional save path
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    setup_plotting_style()
    
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    a_batch = to_np(a_batch)
    f_batch = to_np(f_batch)
    u_batch = to_np(u_batch)
    
    num_samples = min(num_samples, len(a_batch))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        a, f, u = a_batch[i], f_batch[i], u_batch[i]
        
        # Coefficient
        im0 = axes[i, 0].imshow(a, origin='lower', cmap='plasma')
        axes[i, 0].set_title(f'Sample {i}: a(x,y)' if i == 0 else '')
        axes[i, 0].set_ylabel(f'Sample {i}')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Source
        vmax_f = max(abs(f.min()), abs(f.max()))
        im1 = axes[i, 1].imshow(f, origin='lower', cmap='RdBu_r', vmin=-vmax_f, vmax=vmax_f)
        if i == 0:
            axes[i, 1].set_title('f(x,y)')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Solution
        im2 = axes[i, 2].imshow(u, origin='lower', cmap='viridis')
        if i == 0:
            axes[i, 2].set_title('u(x,y)')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
    
    # Column labels
    for j, label in enumerate(['Coefficient a', 'Source f', 'Solution u']):
        axes[0, j].set_title(label)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_dataset_statistics(
    npz_path: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot dataset statistics (histograms and distributions).
    
    Args:
        npz_path: Path to NPZ dataset file
        save_path: Optional save path
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    setup_plotting_style()
    
    data = np.load(npz_path)
    a = data['a']
    f = data['f']
    u = data['u']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Top row: histograms
    axes[0, 0].hist(a.flatten(), bins=100, alpha=0.7, color='blue')
    axes[0, 0].set_title(f'a(x,y) distribution\nmean={a.mean():.3f}, std={a.std():.3f}')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(f.flatten(), bins=100, alpha=0.7, color='green')
    axes[0, 1].set_title(f'f(x,y) distribution\nmean={f.mean():.3f}, std={f.std():.3f}')
    axes[0, 1].set_xlabel('Value')
    
    axes[0, 2].hist(u.flatten(), bins=100, alpha=0.7, color='orange')
    axes[0, 2].set_title(f'u(x,y) distribution\nmean={u.mean():.3f}, std={u.std():.3f}')
    axes[0, 2].set_xlabel('Value')
    
    # Bottom row: per-sample statistics
    a_means = a.mean(axis=(1, 2))
    f_maxs = np.abs(f).max(axis=(1, 2))
    u_maxs = np.abs(u).max(axis=(1, 2))
    
    axes[1, 0].plot(a_means, 'b-', alpha=0.7)
    axes[1, 0].set_title('a mean per sample')
    axes[1, 0].set_xlabel('Sample index')
    axes[1, 0].set_ylabel('Mean')
    
    axes[1, 1].plot(f_maxs, 'g-', alpha=0.7)
    axes[1, 1].set_title('|f| max per sample')
    axes[1, 1].set_xlabel('Sample index')
    axes[1, 1].set_ylabel('Max |f|')
    
    axes[1, 2].plot(u_maxs, 'orange', alpha=0.7)
    axes[1, 2].set_title('|u| max per sample')
    axes[1, 2].set_xlabel('Sample index')
    axes[1, 2].set_ylabel('Max |u|')
    
    plt.suptitle(f'Dataset Statistics: {Path(npz_path).name}\n({len(a)} samples, {a.shape[1]}x{a.shape[2]} grid)', 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_solver_convergence(
    grid_sizes: List[int] = [16, 32, 64, 128],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot solver convergence with grid refinement.
    
    Args:
        grid_sizes: List of grid sizes to test
        save_path: Optional save path
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    from solve_poisson import verify_solver_accuracy
    
    setup_plotting_style()
    
    errors = []
    for gs in grid_sizes:
        error = verify_solver_accuracy(grid_size=gs, verbose=False)
        errors.append(error)
        print(f"Grid {gs}: error = {error:.6e}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(grid_sizes, errors, 'bo-', markersize=8, linewidth=2, label='Numerical error')
    
    # Add reference line for O(h^2) convergence
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
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize generated data')
    parser.add_argument('--input', type=str, help='Path to NPZ file')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to plot')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--test', action='store_true', help='Run visualization tests')
    args = parser.parse_args()
    
    if args.test:
        print("Running visualization tests...")
        
        # Generate test data
        from generate_coefficients import generate_coefficient
        from generate_sources import generate_source
        from solve_poisson import solve_poisson
        from generate_sensors import sample_sensors
        
        config = DataGenConfig(grid_size=64)
        
        a = generate_coefficient(config, seed=42)
        f = generate_source(config, seed=42)
        u = solve_poisson(a, f, config)
        sensors = sample_sensors(u, config, seed=42)
        
        # Test single sample plot
        plot_sample(a, f, u, sensors=sensors, save_path='/tmp/test_sample.png', show=False)
        print("✓ Single sample plot saved")
        
        # Test grid plot
        a_batch = torch.stack([generate_coefficient(config, seed=i) for i in range(4)])
        f_batch = torch.stack([generate_source(config, seed=i) for i in range(4)])
        u_batch = torch.stack([solve_poisson(a_batch[i], f_batch[i], config) for i in range(4)])
        
        plot_samples_grid(a_batch, f_batch, u_batch, num_samples=4, 
                         save_path='/tmp/test_grid.png', show=False)
        print("✓ Grid plot saved")
        
        print("\n✓ All visualization tests passed")
    
    elif args.input:
        # Plot from file
        if args.output:
            Path(args.output).mkdir(parents=True, exist_ok=True)
        
        data = np.load(args.input)
        
        # Plot samples
        plot_samples_grid(
            data['a'], data['f'], data['u'],
            num_samples=args.num_samples,
            save_path=f"{args.output}/samples.png" if args.output else None
        )
        
        # Plot statistics
        plot_dataset_statistics(
            args.input,
            save_path=f"{args.output}/statistics.png" if args.output else None
        )
    
    else:
        print("Use --test to run tests or --input <file.npz> to visualize data")
