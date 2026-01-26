"""
Finite difference solver for the heterogeneous 2D Poisson equation.

Solves: -∇·(a(x,y)∇u(x,y)) = f(x,y)
With Dirichlet boundary conditions: u = 0 on boundary.

Uses a 5-point stencil with harmonic averaging for the variable coefficient.
"""

import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple

from config import DataGenConfig


def build_poisson_matrix(
    a: np.ndarray,
    h: float
) -> sparse.csr_matrix:
    """
    Build the sparse coefficient matrix for the heterogeneous Poisson equation.
    
    Uses harmonic averaging for coefficients at cell faces:
    a_{i+1/2,j} = 2 * a_{i,j} * a_{i+1,j} / (a_{i,j} + a_{i+1,j})
    
    Args:
        a: Diffusion coefficient array of shape (H, W)
        h: Grid spacing
        
    Returns:
        Sparse CSR matrix of shape (N_interior, N_interior)
    """
    H, W = a.shape
    
    # Interior points only (excluding boundaries)
    N_interior = (H - 2) * (W - 2)
    
    if N_interior <= 0:
        raise ValueError(f"Grid too small: {H}x{W}, need at least 3x3")
    
    # Helper to convert (i, j) to linear index (interior points only)
    def idx(i, j):
        return (i - 1) * (W - 2) + (j - 1)
    
    # Build sparse matrix using COO format, then convert to CSR
    rows = []
    cols = []
    vals = []
    
    h2 = h * h
    
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            row = idx(i, j)
            
            # Compute harmonic-averaged coefficients at faces
            # Right face: (i, j+1/2)
            a_right = 2 * a[i, j] * a[i, j+1] / (a[i, j] + a[i, j+1] + 1e-10)
            # Left face: (i, j-1/2)
            a_left = 2 * a[i, j] * a[i, j-1] / (a[i, j] + a[i, j-1] + 1e-10)
            # Top face: (i+1/2, j)
            a_top = 2 * a[i, j] * a[i+1, j] / (a[i, j] + a[i+1, j] + 1e-10)
            # Bottom face: (i-1/2, j)
            a_bottom = 2 * a[i, j] * a[i-1, j] / (a[i, j] + a[i-1, j] + 1e-10)
            
            # Diagonal coefficient
            diag = (a_left + a_right + a_bottom + a_top) / h2
            rows.append(row)
            cols.append(row)
            vals.append(diag)
            
            # Off-diagonal entries (only for interior neighbors)
            # Left neighbor
            if j > 1:
                rows.append(row)
                cols.append(idx(i, j-1))
                vals.append(-a_left / h2)
            
            # Right neighbor
            if j < W - 2:
                rows.append(row)
                cols.append(idx(i, j+1))
                vals.append(-a_right / h2)
            
            # Bottom neighbor
            if i > 1:
                rows.append(row)
                cols.append(idx(i-1, j))
                vals.append(-a_bottom / h2)
            
            # Top neighbor
            if i < H - 2:
                rows.append(row)
                cols.append(idx(i+1, j))
                vals.append(-a_top / h2)
    
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_interior, N_interior))
    return A.tocsr()


def build_rhs(
    f: np.ndarray,
    a: np.ndarray,
    h: float,
    bc_value: float = 0.0
) -> np.ndarray:
    """
    Build the right-hand side vector for the linear system.
    
    Includes boundary condition contributions for points adjacent to boundary.
    
    Args:
        f: Source term array of shape (H, W)
        a: Diffusion coefficient array of shape (H, W)
        h: Grid spacing
        bc_value: Dirichlet boundary condition value
        
    Returns:
        RHS vector of shape (N_interior,)
    """
    H, W = f.shape
    N_interior = (H - 2) * (W - 2)
    
    h2 = h * h
    
    # Initialize with source term at interior points
    rhs = np.zeros(N_interior)
    
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            row = (i - 1) * (W - 2) + (j - 1)
            rhs[row] = f[i, j]
            
            # Add boundary contributions if adjacent to boundary
            if bc_value != 0.0:
                # Left boundary contribution
                if j == 1:
                    a_left = 2 * a[i, j] * a[i, 0] / (a[i, j] + a[i, 0] + 1e-10)
                    rhs[row] += a_left * bc_value / h2
                
                # Right boundary contribution
                if j == W - 2:
                    a_right = 2 * a[i, j] * a[i, W-1] / (a[i, j] + a[i, W-1] + 1e-10)
                    rhs[row] += a_right * bc_value / h2
                
                # Bottom boundary contribution
                if i == 1:
                    a_bottom = 2 * a[i, j] * a[0, j] / (a[i, j] + a[0, j] + 1e-10)
                    rhs[row] += a_bottom * bc_value / h2
                
                # Top boundary contribution
                if i == H - 2:
                    a_top = 2 * a[i, j] * a[H-1, j] / (a[i, j] + a[H-1, j] + 1e-10)
                    rhs[row] += a_top * bc_value / h2
    
    return rhs


def solve_poisson(
    a: torch.Tensor,
    f: torch.Tensor,
    config: DataGenConfig,
    bc_value: float = 0.0
) -> torch.Tensor:
    """
    Solve the heterogeneous Poisson equation -∇·(a∇u) = f.
    
    Uses finite differences with a 5-point stencil and Dirichlet BCs.
    
    Args:
        a: Diffusion coefficient field, shape (H, W)
        f: Source term field, shape (H, W)
        config: Configuration with grid parameters
        bc_value: Dirichlet boundary value (default: 0)
        
    Returns:
        torch.Tensor: Solution field u, shape (H, W)
    """
    # Convert to numpy for scipy solver
    a_np = a.detach().cpu().numpy().astype(np.float64)
    f_np = f.detach().cpu().numpy().astype(np.float64)
    
    H, W = a_np.shape
    h = config.get_grid_spacing()
    
    # Build sparse system
    A = build_poisson_matrix(a_np, h)
    b = build_rhs(f_np, a_np, h, bc_value)
    
    # Solve the linear system
    try:
        u_interior = spsolve(A, b)
    except Exception as e:
        raise RuntimeError(f"Sparse solve failed: {e}. Check coefficient 'a' for near-zero values.")
    
    # Reconstruct full solution with boundary conditions
    u_np = np.full((H, W), bc_value, dtype=np.float64)
    u_np[1:-1, 1:-1] = u_interior.reshape(H - 2, W - 2)
    
    # Convert back to torch tensor
    u = torch.from_numpy(u_np).float()
    
    return u


def solve_poisson_batch(
    a_batch: torch.Tensor,
    f_batch: torch.Tensor,
    config: DataGenConfig,
    bc_value: float = 0.0
) -> torch.Tensor:
    """
    Solve Poisson equation for a batch of samples.
    
    Args:
        a_batch: Batch of coefficients, shape (N, H, W)
        f_batch: Batch of source terms, shape (N, H, W)
        config: Configuration
        bc_value: Boundary value
        
    Returns:
        torch.Tensor: Batch of solutions, shape (N, H, W)
    """
    N = a_batch.shape[0]
    solutions = []
    
    for i in range(N):
        u = solve_poisson(a_batch[i], f_batch[i], config, bc_value)
        solutions.append(u)
    
    return torch.stack(solutions, dim=0)


def verify_solver_accuracy(
    grid_size: int = 64,
    verbose: bool = True
) -> float:
    """
    Verify solver accuracy against a known analytical solution.
    
    Uses the manufactured solution:
    u(x,y) = sin(pi*x) * sin(pi*y)
    
    With constant coefficient a = 1, this gives:
    f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
    
    Args:
        grid_size: Grid resolution
        verbose: Print results
        
    Returns:
        Relative L2 error
    """
    config = DataGenConfig(grid_size=grid_size)
    
    # Create coordinate grids
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Constant coefficient
    a = torch.ones(grid_size, grid_size)
    
    # Manufactured solution and corresponding source
    u_exact = torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    f = 2 * np.pi**2 * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    
    # Numerical solution
    u_numerical = solve_poisson(a, f, config, bc_value=0.0)
    
    # Compute relative L2 error (interior only, avoiding boundary effects)
    interior = slice(1, -1)
    u_exact_int = u_exact[interior, interior]
    u_num_int = u_numerical[interior, interior]
    
    l2_error = torch.norm(u_num_int - u_exact_int) / torch.norm(u_exact_int)
    
    if verbose:
        print(f"Grid size: {grid_size}x{grid_size}")
        print(f"Relative L2 error: {l2_error:.6e}")
        print(f"Max absolute error: {(u_num_int - u_exact_int).abs().max():.6e}")
    
    return l2_error.item()


if __name__ == "__main__":
    # Test solver
    print("Testing Poisson solver...")
    print("=" * 50)
    
    # Verify accuracy
    print("\n1. Accuracy verification with known solution:")
    error = verify_solver_accuracy(grid_size=64)
    assert error < 0.01, f"Solver error too large: {error}"
    print("   ✓ Passed (error < 1%)")
    
    # Test with random inputs
    print("\n2. Random input test:")
    from generate_coefficients import generate_coefficient
    from generate_sources import generate_source
    
    config = DataGenConfig(grid_size=64)
    a = generate_coefficient(config, seed=42)
    f = generate_source(config, seed=42)
    u = solve_poisson(a, f, config)
    
    print(f"   a range: [{a.min():.3f}, {a.max():.3f}]")
    print(f"   f range: [{f.min():.3f}, {f.max():.3f}]")
    print(f"   u range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"   u boundary max: {max(u[0,:].abs().max(), u[-1,:].abs().max(), u[:,0].abs().max(), u[:,-1].abs().max()):.6f}")
    print("   ✓ Solver completed successfully")
    
    print("\n✓ All solver tests passed")
