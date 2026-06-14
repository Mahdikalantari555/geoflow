# Changelog

## 2.0.0

- GeoTensor Spec v2.0 compliant
- Added `.geotensor.resample()` for same-CRS resampling
- Implemented `reproduce()` for provenance replay
- Added `batch_dim` parameter to `to_dataloader()`
- Added `datetime` alias for `open_stac()`
- Added `NonDifferentiableError` guard in `reproject()`
- Fixed `fill(method=...)` to properly use method parameter
- Added `DataNotAvailableError` exception

## 1.0.0

- Initial release
- Core GeoTensor class (xarray.DataArray subclass)
- Alignment engine (spatial, temporal, spectral)
- Mask hierarchy with layers
- Unit safety with Pint
- Patch sampling (tile, random, grid, centroid)
- Provenance tracking (W3C PROV-O)
- ML interoperability (PyTorch, TensorFlow, JAX)
- I/O backends (COG, Zarr, NetCDF, STAC, VRT)
- Differentiable resampling
