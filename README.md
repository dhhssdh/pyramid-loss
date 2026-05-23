# pyramid-loss

> PyTorch implementation of Laplacian and Gaussian pyramid for seismic data.

[![PyPI version](https://img.shields.io/pypi/v/pyramid-loss.svg)](https://pypi.org/project/pyramid-loss/)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x%2B-EE4C2C.svg)](https://pytorch.org/)

---

## Installation

```bash
pip install pyramid-loss
```

---

## Usage

### Build Pyramids

```python
import torch
from pyramid_loss import gaussian_kernel, create_laplacian_pyramid, create_gaussian_pyramid

ns = 3
nt = 100
nr = 300
x  = torch.rand(1, ns, nt, nr)

size     = 5   ### Kernel Size
channels = x.shape[1]
sigma    = 3   ### Standard Deviation
levels   = 5  ### number of level

kernel   = gaussian_kernel(size=size, channels=channels, sigma=sigma)
lap_pyr  = create_laplacian_pyramid(x, kernel, levels=levels)
gau_pyr  = create_gaussian_pyramid(x, kernel, levels=levels)

```

### Compute Loss

```python
from pyramid_loss import PyramidLoss

loss_fn = PyramidLoss(
    pyramid     = 'laplacian',   # 'laplacian' | 'gaussian'
    loss        = 'l1',          # 'l1'        | 'l2'
    levels      = levels,
    channels    = channels,
    use_weights = True,          # False → weight=1 for all levels
                                 # True  → weight=2^level per level (default)
    weights     = [1.0, 2.0, 4.0, 8.0, 16.0],  # or pass custom weights
)

y    = torch.rand(1, 3, 256, 256)
loss = loss_fn(x, y)
```

---

## References

- **[1]** gonglixue — [LaplacianLoss-pytorch](https://github.com/gonglixue/LaplacianLoss-pytorch/blob/master/losses.py)
- **[2]** Burt, P. J., & Adelson, E. H. (1987). *The Laplacian Pyramid as a Compact Image Code.* Readings in Computer Vision, pp. 671–679, Elsevier.

