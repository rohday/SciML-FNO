import torch
import torch.nn as nn
import torch.fft

class SpectralConv2d(nn.Module):
    """
    2D Fourier Layer using Real-to-Complex FFT (rfft2).
    
    This layer:
    1. Transforms input to Fourier domain (Real -> Complex)
    2. Multiplies low-frequency modes by learnable complex weights
    3. Transforms back to spatial domain (Complex -> Real)
    
    Physics Context:
    - We use rfft2 because inputs (a, f) and outputs (u) are strictly real physical quantities.
    - rfft2 saves ~50% memory/compute vs fft2 by exploiting conjugate symmetry.
    - We filter high frequencies (noise) and keep only 'modes' low frequencies.
    """
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Scale factor for initialization
        scale = (1 / (in_channels * out_channels))
        
        # Learnable Complex Weights
        # We need two sets of weights because rfft2 results in a half-plane in the last dimension (y)
        # But we still need to handle both positive and negative frequencies in x.
        #
        # rfft2 output shape: (batch, channels, x, y/2 + 1)
        # corner 1: modes_x * modes_y (Top-Left of frequency spectrum)
        # corner 2: modes_x * modes_y (Bottom-Left of frequency spectrum, aliased negative freqs)
        
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def complex_mul2d(self, input, weights):
        """
        Interaction (Complex Multiplication).
        Input: (batch, in_channel, x, y)
        Weights: (in_channel, out_channel, x, y)
        Output: (batch, out_channel, x, y)
        """
        # (b, i, x, y) * (i, o, x, y) -> (b, o, x, y) using Einstein summation
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. Real-to-Complex FFT
        # Input: (batch, in_channels, H, W)
        # Output: (batch, in_channels, H, W/2 + 1)
        # specific note: rfft2 returns the onesided FFT, saving memory.
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # 2. Multiply relevant Fourier modes
        # We initialize the result container with zeros (High frequencies = 0 ie filtered out)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 
                             dtype=torch.cfloat, device=x.device)
        
        # Upper block (Low freq positive x, Low freq positive y)
        # x_ft slice: [:, :, :modes_x, :modes_y]
        out_ft[:, :, :self.modes_x, :self.modes_y] = \
            self.complex_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y], self.weights1)

        # Lower block (Low freq negative x, Low freq positive y)
        # x_ft slice: [:, :, -modes_x:, :modes_y]
        # Note: In standard FFT, negative frequencies wrap around to the end.
        out_ft[:, :, -self.modes_x:, :self.modes_y] = \
            self.complex_mul2d(x_ft[:, :, -self.modes_x:, :self.modes_y], self.weights2)

        # 3. Complex-to-Real Inverse FFT
        # Output: (batch, out_channels, H, W)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        
        return x
