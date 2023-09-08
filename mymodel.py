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


# import math
# import torch.nn as nn
# import torch.nn.functional as F

# x = torch.rand(1, 3, 64, 64)
# net = Network()
# x_hat, y_likelihoods = net(x)

# # bitrate of the quantized latent
# N, _, H, W = x.size()
# num_pixels = N * H * W
# bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)

# # mean square error
# mse_loss = F.mse_loss(x, x_hat)

# # final loss term
# loss = mse_loss + lmbda * bpp_loss


# aux_loss = net.entropy_bottleneck.loss()

from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv

class Network(CompressionModel):
    def __init__(self, N=128):
        super().__init__()
        self.encode = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )

        self.decode = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return x_hat, y_likelihoods