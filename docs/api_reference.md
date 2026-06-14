# API Reference

## Top-Level Functions

| Function | Description | Module |
|----------|-------------|--------|
| `gt.open_cog(path, ...)` | Open Cloud Optimized GeoTIFF | `backends.py` |
| `gt.open_zarr(store, ...)` | Open Zarr store | `backends.py` |
| `gt.open_netcdf(path, ...)` | Open NetCDF file | `backends.py` |
| `gt.open_stac(url, ...)` | Open STAC catalog | `streaming.py` |
| `gt.align(*tensors, spec=...)` | Multi-tensor alignment | `align.py` |
| `gt.align_to_reference(*tensors, reference=...)` | Align to reference | `align.py` |
| `gt.VRT(files, method=...)` | Virtual mosaic | `backends.py` |
| `gt.unit_safe(enabled)` | Context manager for unit checks | `units.py` |
| `gt.freeze(obj)` | Freeze GeoTensor (read-only) | `provenance.py` |
| `gt.reproduce(provenance_data, ...)` | Reproduce from provenance | `provenance.py` |

## GeoTensor Methods (via `.geotensor` accessor)

### Coordinate System

| Method | Returns | Description |
|--------|---------|-------------|
| `.crs` | `pyproj.CRS` | Coordinate Reference System |
| `.transform` | `affine.Affine` | Geotransform (pixel to world) |
| `.bounds` | `(xmin, ymin, xmax, ymax)` | Geographic bounds |
| `.resolution` | `(x_res, y_res)` | Spatial resolution |
| `.latlon(lat, lon)` | `GeoTensor` | Select by geographic coords |
| `.slice_bbox(west, south, east, north, crs=...)` | `GeoTensor` | Slice by bounding box |
| `.reproject(target_crs, resolution=..., method=...)` | `GeoTensor` | Reproject to new CRS |
| `.resample(resolution=..., method=...)` | `GeoTensor` | Resample within same CRS |
| `.crs_equals(other)` | `bool` | Compare CRS |

### Temporal

| Method | Returns | Description |
|--------|---------|-------------|
| `.time_interval` | `pd.Timedelta` | Median time gap |
| `.time_metadata` | `dict` | Start, end, acquisition_mode |
| `.resample_time(rule, method, fill_value)` | `GeoTensor` | Resample time axis |
| `.temporal_interpolate(method, limit)` | `GeoTensor` | Fill time gaps |
| `.temporal_window(start, end)` | `GeoTensor` | Temporal slice |
| `.temporal_aggregate(period, func)` | `GeoTensor` | Period aggregation |
| `.select_by_event(event_times, tolerance)` | `GeoTensor` | Event-based sampling |
| `.time_to_relative()` | `GeoTensor` | Convert to relative time |

### Units

| Method | Returns | Description |
|--------|---------|-------------|
| `.convert_units(to, from_unit=...)` | `GeoTensor` | Convert physical units |
| `.band_units()` | `dict[str, str]` | Get band units |

### Alignment

| Method | Returns | Description |
|--------|---------|-------------|
| `.align(*tensors, spec=...)` | `list[GeoTensor]` | Multi-tensor alignment |
| `.align_to_reference(*tensors, reference=...)` | `list[GeoTensor]` | Align to reference |

### Mask

| Method | Returns | Description |
|--------|---------|-------------|
| `.mask` | `GeoTensorMask` | Get mask object |
| `.where(condition, drop)` | `GeoTensor` | Create mask from condition |
| `.combine(other_mask, method)` | `GeoTensor` | Combine masks |
| `.apply_mask()` | `GeoTensor` | Apply mask (set to NaN) |
| `.trim_masked(margin)` | `GeoTensor` | Remove fully-masked rows/cols |
| `.fill(value=... or method=...)` | `GeoTensor` | Fill masked values |

### Sampling

| Method | Returns | Description |
|--------|---------|-------------|
| `.sample_tiles(size, stride, ...)` | `list[dict]` | Sliding window tiles |
| `.sample_random(num_patches, size, ...)` | `list[dict]` | Random patches |
| `.sample_grid(nx, ny, size, ...)` | `list[dict]` | Regular grid |
| `.sample_centroids(points, size, ...)` | `list[dict]` | Centroid-based patches |
| `.to_patch_dataset(patch_size, stride, ...)` | `GeoTensorPatchDataset` | ML dataset |

### Differentiable

| Method | Returns | Description |
|--------|---------|-------------|
| `.differentiable_resample(target_shape, ...)` | `GeoTensor` | Gradient-aware resampling |
| `.detach()` | `GeoTensor` | Detach from computation graph |

### I/O

| Method | Returns | Description |
|--------|---------|-------------|
| `.to_zarr(store, ...)` | `None` | Write to Zarr |
| `.to_cog(path, ...)` | `None` | Write to COG |
| `.compute()` | `np.ndarray` | Compute to NumPy |
| `.load()` | `GeoTensor` | Load to memory |
| `.persist()` | `GeoTensor` | Persist Dask array |
| `.chunk(chunks)` | `GeoTensor` | Rechunk |

### ML

| Method | Returns | Description |
|--------|---------|-------------|
| `.to_numpy()` | `np.ndarray` | Convert to NumPy |
| `.to_pytorch(require_grad, device)` | `torch.Tensor` | Convert to PyTorch |
| `.to_tensorflow()` | `tf.Tensor` | Convert to TensorFlow |
| `.to_jax()` | `jax.Array` | Convert to JAX |
| `.to_xarray()` | `xr.DataArray` | Convert to xarray |
| `.to_dataloader(batch_size, ...)` | `DataLoader` | PyTorch DataLoader |
| `.to_tf_dataset(batch_size, ...)` | `tf.data.Dataset` | TensorFlow Dataset |
| `.compute_normalization(method, dim)` | `dict` | Compute norm params |
| `.normalize(norm_params)` | `GeoTensor` | Apply normalization |
| `.denormalize()` | `GeoTensor` | Reverse normalization |

### Provenance

| Method | Returns | Description |
|--------|---------|-------------|
| `.provenance` | `ProvenanceGraph` | Get provenance graph |
| `.export_provenance(format)` | `str` | Export to JSON/JSON-LD/DOT |
| `.freeze()` | `GeoTensor` | Make read-only |
| `.unfreeze()` | `GeoTensor` | Remove read-only |

## Exception Classes

| Exception | Description |
|-----------|-------------|
| `GeoFlowError` | Base class for all GeoFlow errors |
| `CRSMismatchError` | Incompatible CRS |
| `ResolutionMismatchError` | Incompatible resolution |
| `UnitMismatchError` | Incompatible physical units |
| `ImmutableGeoTensorError` | Modified a frozen GeoTensor |
| `NonDifferentiableError` | Non-differentiable op inside grad tape |
| `DataNotAvailableError` | Source data not found during reproduction |

## Data Classes

### BandInfo

```python
gt.BandInfo(
    name='B04',                          # Band name
    wavelength_nm=665.0,                 # Central wavelength (nm)
    unit='reflectance',                  # Physical unit
    sensor='MSI',                        # Sensor name
    measurement_type='reflectance',      # Type of measurement
    valid_range=(0.0, 1.0),             # Valid physical range
    scale_factor=0.0001,                 # DN to physical scale
    add_offset=0.0,                      # Additive offset
    nodata=None,                         # No-data sentinel
    calibration_coeffs={'gain': 0.01},   # Sensor calibration
)
```

### GeoExtent

```python
gt.GeoExtent(
    xmin=500000.0, ymin=4000000.0,
    xmax=600000.0, ymax=4100000.0,
    crs='EPSG:32632'
)
```

### GeoGrid

```python
gt.GeoGrid(
    width=1000, height=1000,
    transform=affine_transform,
    crs='EPSG:32632'
)
```

### AlignmentSpec

```python
gt.AlignmentSpec(
    spatial=gt.SpatialAlignment(
        crs='EPSG:32632',
        resolution=10.0,
        method='bilinear',
        extent='intersection',
    ),
    temporal=gt.TemporalAlignment(
        method='intersection',
        interpolation='nearest',
        tolerance='1D',
    ),
    spectral=gt.SpectralAlignment(
        method='match_band',
        bandset='union',
        wavelength_tolerance_nm=10,
    ),
)
```
