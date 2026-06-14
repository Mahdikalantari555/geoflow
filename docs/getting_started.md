# Getting Started

## Installation

```bash
pip install -r requirements.txt
```

Or add to `pyproject.toml`:

```toml
dependencies = [
    "numpy>=1.21",
    "pandas>=1.3",
    "xarray>=2022.6",
    "rasterio>=1.3",
    "rioxarray>=0.14",
    "pyproj>=3.4",
    "affine>=2.4",
    "pint>=0.20",
    "dask[array]>=2023.0",
    "torch>=2.0",
    "pystac>=1.9",
    "pystac-client>=0.6",
    "zarr>=2.16",
    "h5netcdf>=1.1",
]
```

## Basic Usage

```python
import geoflow as gt
import numpy as np

# Create a GeoTensor from scratch
data = np.random.rand(4, 100, 100).astype(np.float32)
gt_tensor = gt.GeoTensor(
    data,
    coords={
        'x': np.linspace(500000, 501000, 100),
        'y': np.linspace(4000000, 4001000, 100),
        'band': ['B02', 'B03', 'B04', 'B08'],
    },
    dims=['band', 'y', 'x'],
    crs='EPSG:32632',
)

# Inspect properties
print(gt_tensor.geotensor.crs)        # EPSG:32632
print(gt_tensor.geotensor.resolution) # (10.0, 10.0)
print(gt_tensor.geotensor.bounds)     # (500000.0, 4000000.0, 501000.0, 4001000.0)
```

## From a File

```python
# Open a Cloud Optimized GeoTIFF
s2 = gt.open_cog(
    's3://bucket/sentinel2_B04.tif',
    band_info={'B04': gt.BandInfo('B04', wavelength_nm=665.0)},
    chunks={'x': 256, 'y': 256},
)

# Open a Zarr store
data = gt.open_zarr('s3://bucket/data.zarr', group='/S2')

# Open from STAC
s2 = gt.open_stac(
    url='https://earth-search.aws.element84.com/v1',
    collection='sentinel-2-l2a',
    datetime='2023-06-01/2023-09-30',
    bbox=(-120, 35, -119, 36),
    bands=['B04', 'B08'],
)
```

## Spatial Operations

```python
# Select by geographic coordinates
pixel = gt_tensor.geotensor.latlon(lat=35.5, lon=-120.3)

# Slice by bounding box
subset = gt_tensor.geotensor.slice_bbox(
    west=-120.0, south=35.0,
    east=-119.0, north=36.0,
    crs='EPSG:4326',
)

# Resample to different resolution
coarse = gt_tensor.geotensor.resample(resolution=100, method='average')

# Reproject to different CRS
reprojected = gt_tensor.geotensor.reproject(
    target_crs='EPSG:4326',
    resolution=0.001,
    method='bilinear',
)
```

## Alignment

```python
# Align two sensors with different CRS and resolution
s1 = gt.open_cog('sentinel1.tif')
s2 = gt.open_cog('sentinel2.tif')

s1_aligned, s2_aligned = gt.align(
    s1, s2,
    spec=gt.AlignmentSpec(
        spatial={'crs': 'EPSG:32632', 'resolution': 10, 'extent': 'intersection'},
        temporal={'method': 'nearest', 'tolerance': '5D'},
    )
)

# Now safe to combine
combined = s1_aligned + s2_aligned
```

## ML Pipeline

```python
import torch

# Normalize
norm_params = s2.geotensor.compute_normalization(method='zscore')
s2_norm = s2.geotensor.normalize(norm_params)

# Create patch dataset
dataset = s2_norm.geotensor.to_patch_dataset(
    patch_size=256, stride=128,
    augment=True, shuffle=True,
)

# Train
loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
for batch in loader:
    # batch: torch.Tensor of shape (32, 4, 256, 256)
    pass
```

## Next Steps

- [Architecture](architecture.md) - System design and module overview
- [API Reference](api_reference.md) - Complete method documentation
- [Examples](../examples/) - Runnable scripts
