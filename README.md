# pyramid-loss

PyTorch for Laplacian and Gaussian pyramid.

## Installation

```bash
pip install pyramid-loss
```

## Usage

### create pyramids

```python
import torch
from pyramid_loss import gaussian_kernel, create_laplacian_pyramid, create_gaussian_pyramid

x = torch.rand(1, 3, 256, 256)

size = 5
channels = x.shape[1]
sigma = 3
levels = 5
kernel = gaussian_kernel(size=size, channels=channels, sigma=sigma)

lap_pyr  = create_laplacian_pyramid(x, kernel, levels=levels)
gaus_pyr = create_gaussian_pyramid(x, kernel, levels=levels)
```

### loss

```python
loss_fn = PyramidLoss(
    pyramid='laplacian', # 'laplacian' or 'gaussian'
    loss='l1', # 'l1' or 'l2'
    levels=levels,
    channels=channels,
    use_weights=True,  # weight for each level, if use_weights = False default 1, for all level,
                       #                        if use_weights = True  default 2**(level) for each level
    weights=[1.0, 2.0, 4.0, 8.0],  ## self defined weight for each level
)
y = torch.rand(1, 3, 256, 256)
loss = loss_fn(x,y)
```
