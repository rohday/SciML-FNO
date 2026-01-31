"""
Unit tests for data generation module.

Run with: python -m pytest tests/test_generation.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from config import DataGenConfig
from generate_coefficients import generate_coefficient, generate_coefficient_batch
from generate_sources import generate_source, generate_gaussian_bumps, generate_fourier_source
from solve_poisson import solve_poisson, verify_solver_accuracy
from utils import normalize_field, denormalize_field, save_dataset_npz, load_dataset_npz


@pytest.fixture
def config():
    """Default test configuration."""
    return DataGenConfig(grid_size=32, seed=42)


@pytest.fixture
def sample_data(config):
    """Generate sample (a, f, u) data for tests."""
    a = generate_coefficient(config, seed=42)
    f = generate_source(config, seed=42)
    u = solve_poisson(a, f, config)
    return a, f, u


class TestConfig:
    """Tests for configuration module."""
    
    def test_default_config(self):
        config = DataGenConfig()
        assert config.grid_size == 64
        assert config.a_min == 0.1
        assert config.a_max == 3.0
    
    def test_config_validation(self):
        # Valid config
        config = DataGenConfig(grid_size=32, a_min=0.1, a_max=2.0)
        assert config.grid_size == 32
        
        # Invalid: a_min <= 0
        with pytest.raises(AssertionError):
            DataGenConfig(a_min=0.0)
        
        # Invalid: a_max <= a_min
        with pytest.raises(AssertionError):
            DataGenConfig(a_min=2.0, a_max=1.0)
    
    def test_yaml_roundtrip(self, tmp_path):
        config = DataGenConfig(grid_size=48, a_min=0.2, seed=123)
        yaml_path = tmp_path / "config.yaml"
        
        config.to_yaml(str(yaml_path))
        loaded = DataGenConfig.from_yaml(str(yaml_path))
        
        assert loaded.grid_size == 48
        assert loaded.a_min == 0.2
        assert loaded.seed == 123


class TestCoefficientGeneration:
    """Tests for coefficient generation."""
    
    def test_coefficient_shape(self, config):
        a = generate_coefficient(config)
        assert a.shape == (config.grid_size, config.grid_size)
    
    def test_coefficient_range(self, config):
        a = generate_coefficient(config)
        assert a.min() >= config.a_min - 1e-6
        assert a.max() <= config.a_max + 1e-6
    
    def test_coefficient_dtype(self, config):
        a = generate_coefficient(config)
        assert a.dtype == torch.float32
    
    def test_coefficient_reproducibility(self, config):
        a1 = generate_coefficient(config, seed=123)
        a2 = generate_coefficient(config, seed=123)
        assert torch.allclose(a1, a2)
    
    def test_coefficient_variation(self, config):
        a1 = generate_coefficient(config, seed=1)
        a2 = generate_coefficient(config, seed=2)
        assert not torch.allclose(a1, a2)
    
    def test_coefficient_batch(self, config):
        batch = generate_coefficient_batch(config, batch_size=5, base_seed=0)
        assert batch.shape == (5, config.grid_size, config.grid_size)


class TestSourceGeneration:
    """Tests for source term generation."""
    
    def test_source_shape(self, config):
        f = generate_source(config)
        assert f.shape == (config.grid_size, config.grid_size)
    
    def test_source_nonzero(self, config):
        f = generate_source(config)
        assert f.abs().sum() > 0  # Source should not be identically zero
    
    def test_gaussian_bumps(self, config):
        f = generate_gaussian_bumps(config, seed=42)
        assert f.shape == (config.grid_size, config.grid_size)
        # Should have localized peaks
        assert f.max() > f.mean()
    
    def test_fourier_source(self, config):
        config_fourier = DataGenConfig(grid_size=32, source_method="fourier")
        f = generate_fourier_source(config_fourier, seed=42)
        assert f.shape == (32, 32)
    
    def test_source_reproducibility(self, config):
        f1 = generate_source(config, seed=42)
        f2 = generate_source(config, seed=42)
        assert torch.allclose(f1, f2)


class TestPoissonSolver:
    """Tests for the Poisson equation solver."""
    
    def test_solver_output_shape(self, config):
        a = generate_coefficient(config)
        f = generate_source(config)
        u = solve_poisson(a, f, config)
        
        assert u.shape == (config.grid_size, config.grid_size)
    
    def test_solver_boundary_conditions(self, config):
        """Test that u = 0 on boundaries."""
        a = generate_coefficient(config, seed=42)
        f = generate_source(config, seed=42)
        u = solve_poisson(a, f, config, bc_value=0.0)
        
        # Check all boundaries
        assert torch.allclose(u[0, :], torch.zeros(config.grid_size), atol=1e-10)
        assert torch.allclose(u[-1, :], torch.zeros(config.grid_size), atol=1e-10)
        assert torch.allclose(u[:, 0], torch.zeros(config.grid_size), atol=1e-10)
        assert torch.allclose(u[:, -1], torch.zeros(config.grid_size), atol=1e-10)
    
    def test_solver_known_solution(self):
        """Test against manufactured solution with constant coefficient."""
        error = verify_solver_accuracy(grid_size=32, verbose=False)
        assert error < 0.02  # Less than 2% error
    
    def test_solver_convergence(self):
        """Test that error decreases with grid refinement."""
        error_coarse = verify_solver_accuracy(grid_size=16, verbose=False)
        error_fine = verify_solver_accuracy(grid_size=32, verbose=False)
        
        # Error should decrease with refinement
        assert error_fine < error_coarse
    
    def test_solver_positive_source(self, config):
        """Test that positive source gives positive interior solution."""
        a = torch.ones(config.grid_size, config.grid_size)  # Constant coefficient
        f = torch.ones(config.grid_size, config.grid_size)  # Positive source
        u = solve_poisson(a, f, config)
        
        # Interior should be positive (maximum principle)
        interior = u[1:-1, 1:-1]
        assert interior.min() >= 0



class TestUtils:
    """Tests for utility functions."""
    
    def test_normalization_minmax(self):
        x = torch.randn(10, 32, 32) * 5 + 2
        
        x_norm, stats = normalize_field(x, method="minmax")
        
        assert x_norm.min() >= 0
        assert x_norm.max() <= 1
        assert 'min' in stats and 'max' in stats
    
    def test_normalization_roundtrip(self):
        x = torch.randn(10, 32, 32)
        
        x_norm, stats = normalize_field(x, method="minmax")
        x_denorm = denormalize_field(x_norm, "minmax", stats)
        
        assert torch.allclose(x, x_denorm, atol=1e-5)
    
    def test_npz_roundtrip(self, tmp_path, config):
        """Test saving and loading NPZ files."""
        a = np.random.randn(5, config.grid_size, config.grid_size).astype(np.float32)
        f = np.random.randn(5, config.grid_size, config.grid_size).astype(np.float32)
        u = np.random.randn(5, config.grid_size, config.grid_size).astype(np.float32)
        
        path = tmp_path / "test.npz"
        save_dataset_npz(str(path), a, f, u)
        
        loaded = load_dataset_npz(str(path))
        
        assert np.allclose(loaded['a'], a)
        assert np.allclose(loaded['f'], f)
        assert np.allclose(loaded['u'], u)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline(self, config):
        """Test complete generation pipeline."""
        # Generate coefficient
        a = generate_coefficient(config, seed=42)
        
        # Generate source
        f = generate_source(config, seed=42)
        
        # Solve PDE
        u = solve_poisson(a, f, config)
        
        # Verify shapes
        assert a.shape == (config.grid_size, config.grid_size)
        assert f.shape == (config.grid_size, config.grid_size)
        assert u.shape == (config.grid_size, config.grid_size)
        
        # Verify boundary conditions
        assert torch.allclose(u[0, :], torch.zeros(config.grid_size), atol=1e-10)
    
    def test_batch_generation(self, config):
        """Test batch generation consistency."""
        batch_size = 5
        
        a_batch = generate_coefficient_batch(config, batch_size, base_seed=0)
        
        # Each sample should be different
        for i in range(batch_size - 1):
            assert not torch.allclose(a_batch[i], a_batch[i + 1])
        
        # But reproducible
        a_batch_2 = generate_coefficient_batch(config, batch_size, base_seed=0)
        assert torch.allclose(a_batch, a_batch_2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
