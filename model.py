import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models.utils import conv, deconv

class Net1(nn.Module):
    def __init__(self, N=128, loss=0.2):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
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
        self.loss = loss

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        y_rcv = self.transmission(y_hat, self.loss)
        x_hat = self.decode(y_rcv)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def aux_loss(self):
        return self.entropy_bottleneck.loss()

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net

    def transmission(self, y, loss_ratio):
        """Randomly set some values to 0 in y."""
        rnd_y = torch.rand_like(y, device=y.device)
        return y * (rnd_y > loss_ratio)