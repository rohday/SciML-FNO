import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import FNOConfig
from .layers import SpectralConv2d, DepthwiseSpectralConv2d

class FNO2d(nn.Module):
    """FNO for 2D Poisson: (a, f) -> u via Lift -> Fourier Layers -> Project"""
    
    def __init__(self, config: FNOConfig):
        super().__init__()
        self.config = config
        
        # Lifting
        self.p = nn.Conv2d(config.in_channels, config.width, 1)
        
        # Fourier layers
        self.spectral_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for _ in range(config.depth):
            if config.depthwise_separable:
                # Depthwise: per-channel frequency filtering, ~32x fewer params
                self.spectral_layers.append(
                    DepthwiseSpectralConv2d(config.width, config.modes_x, config.modes_y)
                )
            else:
                # Full: cross-channel frequency mixing
                self.spectral_layers.append(
                    SpectralConv2d(config.width, config.width, config.modes_x, config.modes_y)
                )
            self.w_layers.append(nn.Conv2d(config.width, config.width, 1))
            self.bn_layers.append(nn.Identity())

        # Projection
        self.q = nn.Sequential(
            nn.Conv2d(config.width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, config.out_channels, 1)
        )

    def forward(self, a, f):
        if a.ndim == 3: a = a.unsqueeze(1)
        if f.ndim == 3: f = f.unsqueeze(1)
        
        x = torch.cat([a, f], dim=1)
        x = self.p(x)
        
        for spectral, w, bn in zip(self.spectral_layers, self.w_layers, self.bn_layers):
            x1 = spectral(x)
            x2 = w(x)
            x = bn(x1 + x2)
            x = F.gelu(x)
            
            if self.config.dropout > 0:
                x = F.dropout(x, p=self.config.dropout, training=self.training)

        u = self.q(x)
        return u
