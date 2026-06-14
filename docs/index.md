# GeoFlow

**Version:** 2.0.0

GeoFlow is a geospatial tensor library built on Xarray, providing labelled multi-dimensional arrays with CRS, geotransform, band semantics, mask hierarchy, provenance tracking, and seamless ML integration.

```python
import geoflow as gt
```

## Quick Start

```python
import geoflow as gt

# Open a Cloud Optimized GeoTIFF
s2 = gt.open_cog("s2_B04.tif", band_info={"B04": gt.BandInfo("B04")})

# Align two sensors
s1, s2 = gt.align(sentinel1, sentinel2)

# Sample patches for ML
patches = s2.geotensor.sample_tiles(size=256)

# Convert to PyTorch
tensor = s2.geotensor.to_pytorch()
```

## Installation

```bash
pip install -r requirements.txt
```

## Features

- **GeoTensor**: xarray.DataArray subclass with CRS, geotransform, band semantics
- **Automatic Alignment**: Multi-source spatial, temporal, and spectral alignment
- **Mask Hierarchy**: Multi-layer boolean masks with propagation rules
- **Unit Safety**: Pint-based unit registry for Earth observation measurements
- **Patch Sampling**: Tile, random, grid, and centroid sampling for ML
- **Provenance**: W3C PROV-O based lineage tracking
- **ML Integration**: Zero-copy conversion to PyTorch, TensorFlow, JAX
- **Differentiable Operations**: Grid-aligned resampling with gradient support
- **I/O Backends**: COG, Zarr, NetCDF, STAC, VRT virtual mosaics

## Documentation

| Topic | Link |
|-------|------|
| Getting Started | [getting_started.md](getting_started.md) |
| Architecture | [architecture.md](architecture.md) |
| API Reference | [api_reference.md](api_reference.md) |
| Alignment | [alignment.md](alignment.md) |
| Masking | [masking.md](masking.md) |
| Temporal Operations | [temporal.md](temporal.md) |
| Unit Safety | [units.md](units.md) |
| Patch Sampling | [sampling.md](sampling.md) |
| Differentiable Ops | [differentiable.md](differentiable.md) |
| I/O Backends | [io_backends.md](io_backends.md) |
| Provenance | [provenance.md](provenance.md) |
| ML Interoperability | [ml_interop.md](ml_interop.md) |
| Configuration | [configuration.md](configuration.md) |

## Examples

See the `examples/` folder for runnable scripts:

- `01_basic_geotensor.py` - Create and inspect GeoTensors
- `02_open_and_write.py` - I/O operations
- `03_spatial_alignment.py` - Align multi-source data
- `04_temporal_operations.py` - Time handling
- `05_masking.py` - Mask operations
- `06_patch_sampling.py` - ML patch extraction
- `07_ml_pipeline.py` - Full ML pipeline
- `08_unit_safe_math.py` - Unit-aware arithmetic
- `09_provenance_tracking.py` - Lineage tracking
- `10_differentiable.py` - Gradient-aware operations
