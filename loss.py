import math
import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2, type="mse"):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == "mse":
            out["mse_loss"] = torch.mean((output["x_hat"] - target).pow(2))
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
            out["psnr"] = 10 * (torch.log(1.0 / out["mse_loss"]) / np.log(10))
        else:
            out["ms_ssim_loss"] = ms_ssim(output["x_hat"], target, data_range=1.0)
            out["loss"] = self.lmbda * (1 - out["ms_ssim_loss"]) + out["bpp_loss"]

        return out