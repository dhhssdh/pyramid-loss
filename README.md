# pyramid-loss

Laplacian and Gaussian pyramid loss functions for PyTorch.

## Installation

```bash
pip install pyramid-loss
```

## Usage

### PyramidLoss

```python
import torch
from pyramid_loss import PyramidLoss

loss_fn = PyramidLoss(
    pyramid='laplacian',   # 'laplacian' 或 'gaussian'
    loss='l1',             # 'l1', 'l2' 或自定义 callable
    levels=3,
    channels=3,
)

x      = torch.rand(2, 3, 256, 256)
target = torch.rand(2, 3, 256, 256)

# 只要总 loss
loss = loss_fn(x, target)

# 同时要每层 loss（方便 logging）
loss, per_level = loss_fn(x, target, return_per_level=True)
```

### create pyramids

```python
import torch
from pyramid_loss import gaussian_kernel, create_laplacian_pyramid, create_gaussian_pyramid

x = torch.rand(1, 3, 256, 256)
kernel = gaussian_kernel(size=5, channels=3, sigma=1.0)

lap_pyr  = create_laplacian_pyramid(x, kernel, levels=3)
gaus_pyr = create_gaussian_pyramid(x, kernel, levels=3)
```

### weight loss

```python
loss_fn = PyramidLoss(
    pyramid='laplacian',
    loss='l1',
    levels=3,
    channels=3,
    use_weights=True,
    weights=[1.0, 2.0, 4.0, 8.0],  # 长度 = levels + 1
)
```

## License

MIT
