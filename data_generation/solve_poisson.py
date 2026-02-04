"""Finite difference solver for -∇·(a∇u) = f with Dirichlet BC."""

import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional

from config import DataGenConfig


def build_poisson_matrix(a: np.ndarray, h: float) -> sparse.csr_matrix:
    """Build sparse matrix using harmonic averaging for variable coefficient."""
    H, W = a.shape
    N_interior = (H - 2) * (W - 2)
    
    if N_interior <= 0:
        raise ValueError(f"Grid too small: {H}x{W}")
    
    def idx(i, j):
        return (i - 1) * (W - 2) + (j - 1)
    
    rows, cols, vals = [], [], []
    h2 = h * h
    
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            row = idx(i, j)
            
            # Harmonic-averaged coefficients at faces
            a_right = 2 * a[i, j] * a[i, j+1] / (a[i, j] + a[i, j+1] + 1e-10)
            a_left = 2 * a[i, j] * a[i, j-1] / (a[i, j] + a[i, j-1] + 1e-10)
            a_top = 2 * a[i, j] * a[i+1, j] / (a[i, j] + a[i+1, j] + 1e-10)
            a_bottom = 2 * a[i, j] * a[i-1, j] / (a[i, j] + a[i-1, j] + 1e-10)
            
            # Diagonal
            diag = (a_left + a_right + a_bottom + a_top) / h2
            rows.append(row)
            cols.append(row)
            vals.append(diag)
            
            # Off-diagonal
            if j > 1:
                rows.append(row); cols.append(idx(i, j-1)); vals.append(-a_left / h2)
            if j < W - 2:
                rows.append(row); cols.append(idx(i, j+1)); vals.append(-a_right / h2)
            if i > 1:
                rows.append(row); cols.append(idx(i-1, j)); vals.append(-a_bottom / h2)
            if i < H - 2:
                rows.append(row); cols.append(idx(i+1, j)); vals.append(-a_top / h2)
    
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_interior, N_interior))
    return A.tocsr()


def build_rhs(f: np.ndarray, a: np.ndarray, h: float, bc_value: float = 0.0) -> np.ndarray:
    """Build RHS vector including boundary contributions."""
    H, W = f.shape
    N_interior = (H - 2) * (W - 2)
    h2 = h * h
    
    rhs = np.zeros(N_interior)
    
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            row = (i - 1) * (W - 2) + (j - 1)
            rhs[row] = f[i, j]
            
            if bc_value != 0.0:
                if j == 1:
                    a_left = 2 * a[i, j] * a[i, 0] / (a[i, j] + a[i, 0] + 1e-10)
                    rhs[row] += a_left * bc_value / h2
                if j == W - 2:
                    a_right = 2 * a[i, j] * a[i, W-1] / (a[i, j] + a[i, W-1] + 1e-10)
                    rhs[row] += a_right * bc_value / h2
                if i == 1:
                    a_bottom = 2 * a[i, j] * a[0, j] / (a[i, j] + a[0, j] + 1e-10)
                    rhs[row] += a_bottom * bc_value / h2
                if i == H - 2:
                    a_top = 2 * a[i, j] * a[H-1, j] / (a[i, j] + a[H-1, j] + 1e-10)
                    rhs[row] += a_top * bc_value / h2
    
    return rhs


def solve_poisson(a: torch.Tensor, f: torch.Tensor, config: DataGenConfig, bc_value: float = 0.0) -> torch.Tensor:
    """Solve -∇·(a∇u) = f with Dirichlet BC."""
    a_np = a.detach().cpu().numpy().astype(np.float64)
    f_np = f.detach().cpu().numpy().astype(np.float64)
    
    H, W = a_np.shape
    h = config.get_grid_spacing()
    
    A = build_poisson_matrix(a_np, h)
    b = build_rhs(f_np, a_np, h, bc_value)
    
    try:
        u_interior = spsolve(A, b)
    except Exception as e:
        raise RuntimeError(f"Sparse solve failed: {e}")
    
    u_np = np.full((H, W), bc_value, dtype=np.float64)
    u_np[1:-1, 1:-1] = u_interior.reshape(H - 2, W - 2)
    
    return torch.from_numpy(u_np).float()


def solve_poisson_batch(a_batch: torch.Tensor, f_batch: torch.Tensor, config: DataGenConfig, bc_value: float = 0.0) -> torch.Tensor:
    """Solve Poisson for batch of samples."""
    solutions = []
    for i in range(a_batch.shape[0]):
        u = solve_poisson(a_batch[i], f_batch[i], config, bc_value)
        solutions.append(u)
    return torch.stack(solutions, dim=0)


def verify_solver_accuracy(grid_size: int = 64, verbose: bool = True) -> float:
    """Test against manufactured solution u = sin(πx)sin(πy)."""
    config = DataGenConfig(grid_size=grid_size)
    
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    a = torch.ones(grid_size, grid_size)
    u_exact = torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    f = 2 * np.pi**2 * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    
    u_numerical = solve_poisson(a, f, config, bc_value=0.0)
    
    interior = slice(1, -1)
    u_exact_int = u_exact[interior, interior]
    u_num_int = u_numerical[interior, interior]
    
    l2_error = torch.norm(u_num_int - u_exact_int) / torch.norm(u_exact_int)
    
    if verbose:
        print(f"Grid: {grid_size}x{grid_size}, L2 error: {l2_error:.6e}")
    
    return l2_error.item()


if __name__ == "__main__":
    print("Testing Poisson solver...")
    
    error = verify_solver_accuracy(grid_size=64)
    assert error < 0.01
    print("✓ Accuracy test passed")
    
    from generate_coefficients import generate_coefficient
    from generate_sources import generate_source
    
    config = DataGenConfig(grid_size=64)
    a = generate_coefficient(config, seed=42)
    f = generate_source(config, seed=42)
    u = solve_poisson(a, f, config)
    
    print(f"a: [{a.min():.3f}, {a.max():.3f}]")
    print(f"f: [{f.min():.3f}, {f.max():.3f}]")
    print(f"u: [{u.min():.3f}, {u.max():.3f}]")
    print("✓ Solver OK")
