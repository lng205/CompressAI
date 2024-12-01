import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1 as GDN
from compressai.models.utils import (
    conv,
    deconv,
)


class Net(CompressionModel):
    def __init__(self, N=128):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )

        self.g_s = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, x, loss):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        y_rcv = self.transmission(y_hat, loss)
        x_hat = self.g_s(y_rcv)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def transmission(self, y, loss_ratio):
        """Randomly set some values to 0 in y."""
        rnd_y = torch.rand_like(y, device=y.device)
        return y * (rnd_y > loss_ratio)

    def encode(self, img):
        """
        Split the image's latent-space representation into subtensors,
        then do entorpy encode respectively.
        Index the packets for assembly.
        """
        pass

    def decode(self, y):
        """
        Decode and assemble the subtensors back to the tensor,
        and fill the lost subtensors with 0
        """
        pass