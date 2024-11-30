import torch
import torch.nn as nn
import math
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN1 as GDN
from compressai.models.utils import (
    conv,
    deconv,
    remap_old_keys,
    update_registered_buffers,
)


class Net(nn.Module):
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

    def aux_loss(self):
        return self.entropy_bottleneck.loss()

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )
                state_dict = remap_old_keys(name, state_dict)

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def update(self, scale_table=None, force=False, update_quantiles: bool = False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)
            update_quantiles (bool): fast update quantiles (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = False
        for _, module in self.named_modules():
            if isinstance(module, EntropyBottleneck):
                updated |= module.update(force=force, update_quantiles=update_quantiles)
            if isinstance(module, GaussianConditional):
                updated |= module.update_scale_table(scale_table, force=force)
        return updated


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
