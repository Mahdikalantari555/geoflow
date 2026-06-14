# GeoFlow

**A standalone toolkit for machine learning in remote sensing.**

**Version:** 0.0.1 | **License:** MIT

GeoFlow provides geospatial tensor operations built on Xarray, designed for Earth observation and ML pipelines. It bridges the gap between geospatial data and machine learning by embedding CRS, geotransforms, band semantics, masking, and provenance directly into tensor operations.

## Quick Start

```python
import geoflow as gt

# Open multi-sensor data
s1 = gt.open_cog("sentinel1.tif")
s2 = gt.open_cog("sentinel2.tif")

# Align to common grid, time, and bands
s1_a, s2_a = gt.align(s1, s2)
combined = s1_a + s2_a  # safe — same grid

# Sample patches for ML training
dataset = combined.geotensor.to_patch_dataset(patch_size=256, stride=128)

# Create DataLoader
loader = dataset.geotensor.to_dataloader(batch_size=32)

# Normalize
params = combined.geotensor.compute_normalization(method='zscore')
normalized = combined.geotensor.normalize(params)
```

## Features

| Feature | Module | Description |
|---------|--------|-------------|
| **GeoTensor** | `core` | `xarray.DataArray` subclass with CRS, geotransform, band semantics, mask, provenance |
| **Auto-alignment** | `align` | Multi-tensor spatial/temporal/spectral alignment engine |
| **Unit safety** | `units` | Pint-based EO unit registry with `unit_safe()` context manager |
| **Mask hierarchy** | `mask` | Multi-layer boolean mask (nodata, cloud, shadow, saturation, temporal_gap) |
| **Patch sampling** | `sampling` | Sliding window, random, grid, and centroid-based patch extraction |
| **Provenance** | `provenance` | W3C PROV-O graph with JSON/JSON-LD/DOT export |
| **ML interop** | `ml` | PyTorch/TensorFlow/JAX conversion + DataLoaders + normalization |
| **Lazy execution** | `backends` | Dask-backed operations with Zarr/COG/NetCDF/STAC I/O |
| **Differentiable ops** | `differentiable` | Grid-aligned resampling via `torch.grid_sample` |
| **Temporal** | `temporal` | Resampling, interpolation, aggregation, event-based selection |
| **STAC streaming** | `streaming` | Cloud-native STAC integration for lazy remote loading |
| **Tiling** | `tiling` | Unified tiling engine for sampling and inference stitching |
| **Augmentation** | `augmentor` | Remote sensing-aware augmentation (geometric, radiometric, spectral, cloud simulation) |
| **Preprocessing** | `preprocessor` | Atmospheric correction, radiometric calibration, normalization pipeline |
| **Spatial bias** | `spatial_bias` | Diagnostics and mitigation for spatial bias in training datasets |

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `xarray>=2022.6`, `dask[array]>=2023.0`
- `rasterio>=1.3`, `rioxarray>=0.14`, `pyproj>=3.4`, `affine>=2.4`
- `torch>=2.0`, `numpy>=1.21`, `pandas>=1.3`
- `pint>=0.20` (unit registry)
- `zarr>=2.16`, `pystac>=1.9`

### Optional

- `tensorflow>=2.10` — TensorFlow interop
- `jax>=0.4` — JAX interop
- `kornia` — advanced augmentations (rotation, cloud simulation)
- `bottleneck` — faster interpolation

## Module Reference

```
geoflow/
├── core.py              # GeoTensor, BandInfo, GeoExtent, GeoGrid
├── accessor.py          # .geotensor xarray accessor
├── align.py             # AlignmentSpec, align(), align_to_reference()
├── temporal.py          # Resample, interpolate, aggregate, event selection
├── units.py             # GeoUnitRegistry, unit_safe(), convert_units()
├── mask.py              # GeoTensorMask, apply_mask(), fill()
├── sampling.py          # sample_tiles(), sample_random(), GeoTensorPatchDataset
├── ml.py                # to_pytorch(), normalize(), denormalize(), to_dataloader()
├── backends.py          # open_cog(), open_zarr(), to_zarr(), to_cog()
├── provenance.py        # ProvenanceGraph, freeze(), export_provenance()
├── differentiable.py    # differentiable_resample(), detach()
├── streaming.py         # STACStreamingEngine, open_stac()
├── tiling.py            # GeoTilingEngine, stitch_tiles()
├── augmentor.py         # GeoAugmentor pipeline
├── preprocessor.py      # GeoPreprocessor pipeline
├── spatial_bias.py      # SpatialBiasDiagnostics
├── integration.py       # GeoCollateUtils
├── visualizer.py        # GeoVisualizer
├── generator.py         # GeoSyntheticGenerator
├── zero_copy.py         # ZeroCopyUtils (pinned memory, async GPU transfer)
├── loader.py            # GeoDataset, GeoDataLoader
├── config.py            # GeoFlowConfig singleton
├── docs/                # Documentation
├── examples/            # Runnable examples
└── tests/
    ├── conftest.py      # Test configuration
    └── test_geoflow.py  # Test suite
```

## Usage Examples

### Open and Inspect Data

```python
import geoflow as gt

s2 = gt.open_cog("sentinel2_l2a.tif")
print(s2.shape)           # (4, 1000, 1000)
print(s2.geotensor.crs)   # EPSG:4326
print(s2.geotensor.bounds) # (xmin, ymin, xmax, ymax)
```

### Align Multiple Sources

```python
s1 = gt.open_cog("sentinel1.tif")
s2 = gt.open_cog("sentinel2.tif")

# Automatic alignment to finest resolution, common CRS, intersecting extent
s1_a, s2_a = gt.align(s1, s2)

# Or align to a specific reference
aligned = gt.align_to_reference(s1, s2, reference=s2)
```

### Resample Spatial Resolution

```python
# Resample to 100m resolution (same CRS)
coarse = s2.geotensor.resample(resolution=100, method='average')

# Reproject to different CRS
reprojected = s2.geotensor.reproject('EPSG:32632', resolution=10)
```

### Sample Patches for ML

```python
# Sliding window sampling
patches = s2.geotensor.sample_tiles(
    size=256, stride=128,
    mask_threshold=0.2,  # skip patches with >20% masked pixels
)

# Create a PyTorch-compatible dataset
dataset = s2.geotensor.to_patch_dataset(
    patch_size=256, stride=128,
    augment=True,
)
```

### Normalize and Convert

```python
import torch

# Compute normalization params from training data
params = s2.geotensor.compute_normalization(method='zscore')

# Apply normalization
normalized = s2.geotensor.normalize(params)

# Convert to PyTorch tensor
tensor = normalized.geotensor.to_pytorch()  # [C, H, W] float32

# Create DataLoader with configurable batch dimension
loader = normalized.geotensor.to_dataloader(batch_size=32, batch_dim='band')
```

### Unit Safety

```python
# Enable unit safety globally
with gt.unit_safe(True):
    # This will raise UnitMismatchError
    result = radiance_tensor + reflectance_tensor

# Disable for specific operations
with gt.unit_safe(False):
    normalized = (tensor - mean) / std  # no unit check
```

### Provenance Tracking

```python
# Export provenance
json_str = s2.geotensor.export_provenance(format='json')
jsonld_str = s2.geotensor.export_provenance(format='json-ld')

# Freeze tensor (read-only)
s2.geotensor.freeze()

# Reproduce from provenance
reproduced = gt.reproduce('provenance.json')
```

## Documentation

See the `docs/` folder for detailed documentation:

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md)
- [Examples](docs/examples.md)

## Testing

```bash
cd geoflow
pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.
