# Mask Operations

GeoTensor supports multi-layer boolean masks for handling missing data.

## Mask Structure

```python
mask = tensor.geotensor.mask  # GeoTensorMask
```

Layers: `'nodata'`, `'cloud'`, `'shadow'`, `'saturation'`, `'temporal_gap'`

## Access Layers

```python
cloud_mask = mask.get_layer('cloud')
mask.set_layer('cloud', cloud_array)
mask.remove_layer('cloud')
```

## Combined Mask

```python
combined = mask.combined(mode='any')  # True if any layer True
combined = mask.combined(mode='all')  # True if all layers True
```

## Create Masks

```python
# From condition
masked = tensor.geotensor.where(tensor > 0, drop=False)

# Combine masks
combined = tensor.geotensor.combine(other_mask, method='or')
```

## Apply and Trim

```python
# Apply mask (set masked cells to NaN)
masked_data = tensor.geotensor.apply_mask()

# Remove fully-masked rows/cols
trimmed = tensor.geotensor.trim_masked(margin=10)
```

## Fill

```python
# Constant fill
filled = tensor.geotensor.fill(value=0)

# Spatial interpolation (nearest neighbor)
filled = tensor.geotensor.fill(method='nearest', dim='x', limit=100)

# Temporal interpolation
filled = tensor.geotensor.fill(method='time')

# Linear interpolation
filled = tensor.geotensor.fill(method='linear', dim='time')
```

## Propagation Rules

- Arithmetic: if any input pixel is masked, output pixel is masked
- Aggregations: masked pixels are ignored (skipna=True)
- No automatic filling - user must explicitly call `.fill()`
