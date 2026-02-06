from dataclasses import dataclass

@dataclass
class FNOConfig:
    """FNO model configuration."""
    in_channels: int = 2      # a(x,y), f(x,y)
    out_channels: int = 1     # u(x,y)
    modes_x: int = 16
    modes_y: int = 16
    width: int = 48
    depth: int = 6
    dropout: float = 0.0
    depthwise_separable: bool = True  # Use DepthwiseSpectralConv2d for ~26x fewer params
    
    def __post_init__(self):
        assert self.modes_x > 0
        assert self.modes_y > 0
        assert self.width > 0
