# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     User API                            │
├─────────────────────────────────────────────────────────┤
│  GeoTensor (subclass of xarray.DataArray)               │
│    - .geotensor accessor provides all methods           │
├─────────────────────────────────────────────────────────┤
│  Alignment Engine  │  Mask Engine  │  Unit Registry     │
├─────────────────────────────────────────────────────────┤
│  Xarray + Dask (core labelled array)                    │
├─────────────────────────────────────────────────────────┤
│  rioxarray (CRS, reproject) │ rasterio (I/O, warping)   │
├─────────────────────────────────────────────────────────┤
│  Backends: Zarr, COG, NetCDF, STAC                      │
└─────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Geography is intrinsic, not metadata** – Every GeoTensor carries its spatial reference as a first-class citizen.
2. **Physical measurements remain interpretable** – Units, sensor calibration, and uncertainty are attached.
3. **Operations preserve spatial validity** – Invalid combinations raise explicit errors.
4. **Large-scale data remains computationally feasible** – Lazy evaluation, chunking, and cloud-native I/O are default.
5. **Machine learning interoperability is mandatory** – Convert to framework tensors with zero copy when possible.
6. **Invalid scientific operations fail explicitly** – No silent nodata filling, no automatic unit conversion without user intent.
7. **Build, don't rewrite** – Reuse Xarray/Dask/Rioxarray wherever possible.

## Module Reference

| Module | File | Purpose |
|--------|------|---------|
| `geoflow.core` | `core.py` | GeoTensor class, BandInfo, GeoExtent, GeoGrid, exceptions |
| `geoflow.accessor` | `accessor.py` | `.geotensor` accessor with all methods |
| `geoflow.temporal` | `temporal.py` | Time handling, resampling, aggregation |
| `geoflow.units` | `units.py` | Pint registry, unit conversion, unit safety |
| `geoflow.align` | `align.py` | AlignmentSpec, align(), align_to_reference() |
| `geoflow.mask` | `mask.py` | GeoTensorMask, layers, fill, trim |
| `geoflow.sampling` | `sampling.py` | Patch sampling, GeoTensorPatchDataset |
| `geoflow.differentiable` | `differentiable.py` | Differentiable resampling |
| `geoflow.backends` | `backends.py` | COG/Zarr/NetCDF I/O, VRT |
| `geoflow.streaming` | `streaming.py` | STAC streaming engine, open_stac() |
| `geoflow.provenance` | `provenance.py` | ProvenanceGraph, export, freeze |
| `geoflow.ml` | `ml.py` | Framework conversions, DataLoader, normalization |
| `geoflow.config` | `config.py` | Global configuration |
| `geoflow.preprocessor` | `preprocessor.py` | Atmospheric correction, spectral indices |
| `geoflow.augmentor` | `augmentor.py` | Geometric/radiometric/spectral augmentations |
| `geoflow.generator` | `generator.py` | Synthetic data generation |
| `geoflow.visualizer` | `visualizer.py` | False color composites, band combinations |
| `geoflow.loader` | `loader.py` | GeoDataset, GeoDataLoader |
| `geoflow.zero_copy` | `zero_copy.py` | Zero-copy GPU transfer utilities |
| `geoflow.integration` | `integration.py` | GeoCollateUtils for collation |
| `geoflow.spatial_bias` | `spatial_bias.py` | Spatial bias diagnostics |

## Data Flow

```
File (COG/Zarr/NetCDF/STAC)
    ↓ open_cog / open_zarr / open_stac
GeoTensor (xarray.DataArray + attrs)
    ↓ .geotensor accessor methods
Operations (reproject, resample, align, mask, fill)
    ↓ .geotensor.to_pytorch() / to_dataloader()
PyTorch / TensorFlow / JAX tensors
    ↓
ML Training Loop
```
