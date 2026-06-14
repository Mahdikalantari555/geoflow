# ML Interoperability

## Framework Conversions

```python
# NumPy
arr = tensor.geotensor.to_numpy()

# PyTorch
t = tensor.geotensor.to_pytorch(require_grad=True, device='cuda:0')

# TensorFlow
t = tensor.geotensor.to_tensorflow()

# JAX
t = tensor.geotensor.to_jax()

# Xarray (zero-copy)
da = tensor.geotensor.to_xarray()
```

## DataLoader Integration

```python
# PyTorch
loader = tensor.geotensor.to_dataloader(
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    batch_dim='time',  # which dim to split
)

# TensorFlow
dataset = tensor.geotensor.to_tf_dataset(
    batch_size=32,
    shuffle=True,
    prefetch=2,
)
```

## Normalization

```python
# Compute normalization parameters
norm_params = tensor.geotensor.compute_normalization(
    method='zscore',    # zscore, minmax, robust
    dim=['x', 'y'],
)

# Apply normalization
normalized = tensor.geotensor.normalize(norm_params)

# Reverse
original = normalized.geotensor.denormalize()
```

## Methods

- `zscore` — (x - mean) / std
- `minmax` — (x - min) / range
- `robust` — (x - median) / IQR
