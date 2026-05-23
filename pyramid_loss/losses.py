"""
Common pyramid-based losses for seismic inversion tasks, including Laplacian and Gaussian pyramids.

Credits
-------
Adapted from:
    gonglixue, "LaplacianLoss-pytorch"
    https://github.com/gonglixue/LaplacianLoss-pytorch/blob/master/losses.py

@author: Faxuan Wu (faxuanwu@126.com)
"""
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, List, Union, Tuple


def gaussian_kernel(size=5, device=torch.device('cpu'), channels=3, sigma=1, dtype=torch.float):
    """Generates a 2D Gaussian kernel and repeats it for each channel."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)

    kernel_tensor = torch.tensor(kernel, dtype=dtype).unsqueeze(0).unsqueeze(0)  # Shape: [1,1,H,W]
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)  # Shape: [C,1,H,W]
    return kernel_tensor.to(device)

def gaussian_conv2d(x, g_kernel):
    """Applies Gaussian convolution to each channel."""
    channels = x.shape[1]
    padding = g_kernel.shape[-1] // 2  # Assumes square kernel
    return F.conv2d(x, weight=g_kernel, padding=padding, groups=channels)

def downsample(x):
    """Downsamples spatial dimensions by factor of 2."""
    return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

def upsample(x, size):
    """Upsamples using bilinear interpolation followed by Gaussian blur."""
    x_up = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    return x_up

def create_laplacian_pyramid(x, kernel, levels):
    """Builds the Laplacian pyramid with Gaussian blur and upsampling."""
    pyramids = []
    current = x
    for _ in range(levels):
        filtered = gaussian_conv2d(current, kernel)
        down = downsample(filtered)
        up = upsample(down, size=current.shape[2:])  # Match original resolution
        lap = current - up
        pyramids.append(lap)
        current = down
    pyramids.append(current)  # Final low-pass
    return pyramids
    
def create_gaussian_pyramid(x: torch.Tensor, kernel: torch.Tensor, levels: int):
    """
    Build Gaussian pyramid:
      level0: original (optionally blurred before saving; here we save blurred->down pairs)
      For i in 1..levels: apply blur -> downsample -> save the downsampled lowpass
    Returns list of pyramid levels as tensors: [lvl0, lvl1, lvl2, ..., lvlN]
    where lvl0 = x (or blurred x if you prefer), lvl1 = down(blt(x)), etc.
    """
    pyramids = []
    current = x
    # Optionally: save the original as-is or blurred original (I keep original followed by blurred-downs)
    pyramids.append(current)
    for _ in range(levels):
        # blur to avoid aliasing
        filtered = gaussian_conv2d(current, kernel)
        down = downsample(filtered)
        pyramids.append(down)
        current = down
    return pyramids

class PyramidLoss(torch.nn.Module):
    """
    Unified pyramid loss.

    Args:
      pyramid: 'laplacian' or 'gaussian'
      loss: 'l1', 'l2' or a callable loss_fn(a, b) -> scalar Tensor
      levels: number of downsample steps (the pyramid will have levels+1 elements)
      channels: expected number of channels (used to build kernel)
      kernel_size, sigma: gaussian kernel params
      use_weights: if True apply per-level weights
      weights: optional list of length levels+1; if None and use_weights True, defaults to [2**i]
    """
    def __init__(self,
                 pyramid: str = 'laplacian',
                 loss: Union[str, Callable] = 'l1',
                 reduction: str = 'mean', ## or sum
                 levels: int = 3,
                 channels: int = 11,
                 kernel_size: int = 5,
                 sigma: float = 1.0,
                 device: torch.device = torch.device('cpu'),
                 dtype=torch.float,
                 use_weights: bool = False,
                 weights: Optional[List[float]] = None):
        super().__init__()
        pyramid = pyramid.lower()
        assert pyramid in ('laplacian', 'gaussian'), "pyramid must be 'laplacian' or 'gaussian'"
        self.pyramid = pyramid
        self.levels = levels
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.dtype = dtype

        # register kernel buffer so it moves with module, but keep flexible to change device in forward
        kernel = gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, device=device, dtype=dtype)
        self.register_buffer('_kernel', kernel, persistent=False)

        # loss selection
        if isinstance(loss, str):
            loss = loss.lower()
            if loss == 'l1':
                self.loss_fn = lambda a, b: F.l1_loss(a, b, reduction=reduction)
            elif loss in ('l2', 'mse'):
                self.loss_fn = lambda a, b: F.mse_loss(a, b, reduction=reduction)
            else:
                raise ValueError("loss must be 'l1', 'l2' or a callable")
        elif callable(loss):
            self.loss_fn = loss
        else:
            raise ValueError("loss must be 'l1', 'l2' or a callable")

        self.use_weights = bool(use_weights)
        if self.use_weights:
            if weights is None:
                # default weights: increasing power-of-two (coarse scales heavier). length = levels + 1
                self.weights = [2 ** i for i in range(levels + 1)]
            else:
                assert len(weights) == levels + 1, "weights length must be levels+1"
                self.weights = weights
            # convert to tensor for device movement convenience
            self.register_buffer('_weights_tensor', torch.tensor(self.weights, dtype=dtype), persistent=False)
        else:
            self._weights_tensor = None

    def _make_pyramids(self, x: torch.Tensor):
        kernel = self._kernel.to(device=x.device, dtype=x.dtype)
        if self.pyramid == 'laplacian':
            return create_laplacian_pyramid(x, kernel, self.levels)
        else:
            return create_gaussian_pyramid(x, kernel, self.levels)

    def forward(self,
                x: torch.Tensor,
                target: torch.Tensor,
                return_per_level: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert x.shape == target.shape, "Input and target must have the same shape"
        pyr_x = self._make_pyramids(x)
        pyr_t = self._make_pyramids(target)
        # compute per-level loss, taking care of shape mismatches by upsampling to max spatial size
        per_level_losses: List[torch.Tensor] = []
        total = 0.0
        for i, (px, pt) in enumerate(zip(pyr_x, pyr_t)):
            # if px.shape != pt.shape:
            #     # upsample smaller to larger
            #     target_h = max(px.shape[2], pt.shape[2])
            #     target_w = max(px.shape[3], pt.shape[3])
            #     px = upsample_to(px, (target_h, target_w)) if px.shape[2:] != (target_h, target_w) else px
            #     pt = upsample_to(pt, (target_h, target_w)) if pt.shape[2:] != (target_h, target_w) else pt

            lvl_loss = self.loss_fn(px, pt)
            per_level_losses.append(lvl_loss.detach().cpu() if lvl_loss.requires_grad else lvl_loss.cpu())
            if self.use_weights:
                w = self._weights_tensor[i].to(device=lvl_loss.device, dtype=lvl_loss.dtype)
                total = total + w * lvl_loss
            else:
                total = total + lvl_loss

        if return_per_level:
            # also return the raw per-level scalar tensors (not on GPU) for logging convenience
            return total, per_level_losses
        return total


