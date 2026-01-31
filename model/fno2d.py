import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import FNOConfig
from .layers import SpectralConv2d

class FNO2d(nn.Module):
    """
    Fourier Neural Operator (2D) for Poisson Equation.
    
    Architecture:
    1. Lift: (a, f) -> Hidden
    2. Fourier Layers: Spectral Conv + Skip + Activation
    3. Project: Hidden -> u
    
    Physics-Aware Input:
    - a: Diffusion coefficient
    - f: Source term
    - Returns u: Potential/Temperature field
    """
    def __init__(self, config: FNOConfig):
        super().__init__()
        self.config = config
        
        # 1. Lifting Layer (P)
        # Projects input physics channels (2: a, f) to high-dim latent space
        self.p = nn.Conv2d(config.in_channels, config.width, 1)
        
        # 2. Fourier Layers (H)
        self.spectral_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for _ in range(config.depth):
            # Spectral Convolution: Global mixing in frequency domain
            self.spectral_layers.append(
                SpectralConv2d(config.width, config.width, config.modes_x, config.modes_y)
            )
            # Point-wise Convolution (W): Local mixing in spatial domain (kernel size 1)
            self.w_layers.append(nn.Conv2d(config.width, config.width, 1))
            
            # Normalization (optional, but helps stability)
            self.bn_layers.append(nn.GroupNorm(1, config.width) if False else nn.Identity())

        # 3. Projection Layer (Q)
        # Maps latent space back to single output channel (u)
        self.q = nn.Sequential(
            nn.Conv2d(config.width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, config.out_channels, 1)
        )

    def forward(self, a, f):
        """
        Forward pass solving -∇·(a∇u) = f
        
        Args:
            a: Variable coefficient (batch, H, W) or (batch, 1, H, W)
            f: Source term (batch, H, W) or (batch, 1, H, W)
        
        Returns:
            u: Solution field (batch, 1, H, W)
        """
        # Ensure inputs have channel dim
        if a.ndim == 3: a = a.unsqueeze(1)
        if f.ndim == 3: f = f.unsqueeze(1)
        
        # Concatenate inputs along channel dimension: [Batch, 2, H, W]
        # x represents the physical state at each point
        x = torch.cat([a, f], dim=1)
        
        # 1. Lift
        x = self.p(x)
        
        # 2. Iterate Fourier Layers
        for spectral, w, bn in zip(self.spectral_layers, self.w_layers, self.bn_layers):
            # Branch 1: Global spectral convolution
            x1 = spectral(x)
            
            # Branch 2: Local linear transform (skip-like)
            x2 = w(x)
            
            # Combine and Activate
            x = x1 + x2
            x = bn(x)
            x = F.gelu(x)
            
            # Optional dropout if defined
            if self.config.dropout > 0:
                x = F.dropout(x, p=self.config.dropout, training=self.training)

        # 3. Project
        u = self.q(x)
        
        return u
