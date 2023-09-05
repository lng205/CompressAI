import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN

class Network(nn.Module):
    def __init__(self, N=128):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            nn.Conv2d(3, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2), 
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, padding=2, output_padding=1, stride=2)
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return x_hat, y_likelihoods
