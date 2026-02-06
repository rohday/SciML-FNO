from dataclasses import dataclass

@dataclass
class FNOConfig:
    """FNO model configuration."""
    in_channels: int = 2      # a(x,y), f(x,y)
    out_channels: int = 1     # u(x,y)
    modes_x: int = 12
    modes_y: int = 12
    width: int = 32
    depth: int = 4
    dropout: float = 0.0
    
    def __post_init__(self):
        assert self.modes_x > 0
        assert self.modes_y > 0
        assert self.width > 0
