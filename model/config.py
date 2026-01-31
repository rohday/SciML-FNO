from dataclasses import dataclass

@dataclass
class FNOConfig:
    """
    Configuration for the Fourier Neural Operator.
    """
    # Physics/Data dimensions
    in_channels: int = 2   # a(x,y), f(x,y)
    out_channels: int = 1  # u(x,y)
    
    # Model architecture
    modes_x: int = 12      # Number of Fourier modes to keep in x (low-freq)
    modes_y: int = 12      # Number of Fourier modes to keep in y (low-freq)
    width: int = 32        # Hidden channel dimension (keep small for edge)
    depth: int = 4         # Number of Fourier Layers
    
    # Training
    dropout: float = 0.0
    
    def __post_init__(self):
        """Validation"""
        assert self.modes_x > 0
        assert self.modes_y > 0
        assert self.width > 0
