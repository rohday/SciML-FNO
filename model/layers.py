import torch
import torch.nn as nn
import torch.fft

class SpectralConv2d(nn.Module):
    """2D Fourier layer using rfft2. Keeps only low-frequency modes."""
    
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1 / (in_channels * out_channels)
        
        # Two weight sets for positive/negative x frequencies (rfft gives half-plane in y)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def complex_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        x_ft = torch.fft.rfft2(x, norm='ortho')

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 
                             dtype=torch.cfloat, device=x.device)
        
        # Low freq positive x
        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.complex_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y], self.weights1)

        # Low freq negative x (wraps around)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = \
            self.complex_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_y], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        
        return x
