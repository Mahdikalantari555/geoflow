# Patch Sampling

## Tile Sampling

```python
patches = tensor.geotensor.sample_tiles(
    size=256,                       # pixels or '256m'
    stride=128,                     # overlap control
    extent='full',                  # full extent or bbox
    mask_threshold=0.2,             # reject >20% masked
    random_shift=False,             # jitter positions
    format='dict',                  # dict or list
)
```

## Random Sampling

```python
patches = tensor.geotensor.sample_random(
    num_patches=1000,
    size=256,
    region=None,  # geometry or mask layer
)
```

## Grid Sampling

```python
patches = tensor.geotensor.sample_grid(
    nx=10, ny=10,
    size=256,
)
```

## Centroid Sampling

```python
patches = tensor.geotensor.sample_centroids(
    points=points_gdf,  # GeoDataFrame or list of (x, y)
    size=256,
)
```

## Patch Dataset

```python
dataset = tensor.geotensor.to_patch_dataset(
    patch_size=256,
    stride=128,
    mask_threshold=0.2,
    augment=True,
    shuffle=True,
)

loader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

## Features

- Lazy loading from Dask arrays
- On-the-fly augmentation (flips, rotations)
- Mask threshold filtering
- Custom transforms
