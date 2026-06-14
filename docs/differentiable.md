# Differentiable Operations

## Differentiable Resampling

```python
result = tensor.geotensor.differentiable_resample(
    target_shape=(512, 512),  # (height, width)
    method='bilinear',        # bilinear or nearest
    align_corners=True,
)
```

Uses `torch.nn.functional.grid_sample` for gradient-aware interpolation.

## Gradient Escape Hatch

```python
# Detach from computation graph
detached = tensor.geotensor.detach()

# Convert to PyTorch with gradients
x = tensor.geotensor.to_pytorch(require_grad=True)
loss = x.sin().mean()
loss.backward()
```

## Limitations

- Reprojection (CRS change) is NOT differentiable
- Calling `.reproject()` inside a gradient tape raises `NonDifferentiableError`
- Use `.detach()` before non-differentiable operations

## Supported Operations

| Operation | Differentiable |
|-----------|---------------|
| Grid-aligned resampling | Yes |
| Coordinate transforms | Yes |
| Spectral indices | Yes |
| Reprojection | No |
| Masking | No |
