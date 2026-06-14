"""Data I/O backends for GeoTensor (§13 of GeoTensor Spec).

Provides open / write functions for COG, Zarr, NetCDF, and VRT virtual
mosaics, plus lazy computation primitives (chunk, persist, compute, load).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from .core import BandInfo, GeoTensor, to_geotensor


# ====================================================================
# Open functions
# ====================================================================


def open_cog(
    path: str,
    band_info: Optional[Union[Dict[str, BandInfo], List[BandInfo], BandInfo]] = None,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> GeoTensor:
    """Open a Cloud Optimized GeoTIFF (COG) as a GeoTensor.

    Args:
        path: Path or URL to the COG file.
        band_info: Optional band metadata.
        chunks: Chunk sizes (e.g. ``{'x': 256, 'y': 256}``).
                If ``None``, uses default spatial chunks.
        **kwargs: Additional arguments for :func:`rioxarray.open_rasterio`.

    Returns:
        A :class:`GeoTensor` with lazy data.
    """
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        raise ImportError("rioxarray is required to open COGs.")

    kwargs.setdefault('masked', True)
    if chunks is not None:
        kwargs['chunks'] = chunks

    xds = xr.open_dataset(path, engine='rasterio', **kwargs)
    da = xds['band_data'] if 'band_data' in xds else xds.to_array()

    # Convert to GeoTensor
    gt = GeoTensor(
        da.data,
        coords=da.coords,
        dims=da.dims,
        crs=da.attrs.get('crs'),
        band_info=band_info,
        attrs={k: v for k, v in da.attrs.items() if k != 'crs'},
    )
    return gt


def open_zarr(
    store: str,
    group: Optional[str] = None,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> GeoTensor:
    """Open a Zarr store as a GeoTensor.

    Args:
        store: Zarr store path or URL.
        group: Optional group within the Zarr store.
        chunks: Chunk overrides.
        **kwargs: Additional arguments for :func:`xr.open_zarr`.

    Returns:
        A :class:`GeoTensor` with lazy data.
    """
    kwargs.setdefault('masked', True)
    if chunks is not None:
        kwargs['chunks'] = chunks

    ds = xr.open_zarr(store, group=group, **kwargs)

    # If dataset has multiple variables, merge into single array
    if len(ds.data_vars) == 1:
        da = next(iter(ds.data_vars.values()))
    elif 'data' in ds:
        da = ds['data']
    else:
        da = ds.to_array()

    return GeoTensor(
        da.data,
        coords=da.coords,
        dims=da.dims,
        crs=da.attrs.get('crs'),
        attrs={k: v for k, v in da.attrs.items() if k != 'crs'},
    )


def open_netcdf(
    path: str,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> GeoTensor:
    """Open a NetCDF file as a GeoTensor.

    Args:
        path: Path to NetCDF file.
        chunks: Chunk overrides.
        **kwargs: Additional arguments for :func:`xr.open_dataset`.

    Returns:
        A :class:`GeoTensor` with lazy data.
    """
    ds = xr.open_dataset(path, **kwargs)

    if len(ds.data_vars) == 1:
        da = next(iter(ds.data_vars.values()))
    elif 'data' in ds:
        da = ds['data']
    else:
        da = ds.to_array()

    return GeoTensor(
        da.data,
        coords=da.coords,
        dims=da.dims,
        crs=da.attrs.get('crs'),
        attrs={k: v for k, v in da.attrs.items() if k != 'crs'},
    )


# ====================================================================
# Write operations
# ====================================================================


def to_zarr(
    obj: xr.DataArray,
    store: str,
    compute: bool = True,
    consolidated: bool = True,
    chunks: Optional[Dict[str, int]] = None,
    encoding: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Write to a Zarr store.

    Args:
        obj: Input DataArray / GeoTensor.
        store: Zarr store path or URL.
        compute: If ``True``, compute and write immediately.
        consolidated: Write consolidated metadata.
        chunks: Chunk sizes for storage.
        encoding: Per-variable encoding (e.g. compression).
        **kwargs: Additional arguments for :meth:`xr.DataArray.to_zarr`.
    """
    if encoding is None:
        encoding = {obj.name or 'data': {'compress': 'zstd', 'shuffle': True}}

    kwargs.setdefault('encoding', encoding)
    kwargs.setdefault('consolidated', consolidated)
    kwargs.setdefault('compute', compute)

    if chunks is not None:
        obj = obj.chunk(chunks)

    # Attach CRS and other metadata as attributes
    attrs = {k: v for k, v in obj.attrs.items()
             if isinstance(v, (str, int, float, list, dict))}
    obj.attrs.update(attrs)

    if compute:
        obj.to_zarr(store, **kwargs)
    else:
        delayed = obj.to_zarr(store, compute=False, **kwargs)
        return delayed


def to_cog(
    obj: xr.DataArray,
    path: str,
    dtype: str = 'uint16',
    overviews: Optional[List[int]] = None,
    **kwargs: Any,
) -> None:
    """Write to a Cloud Optimized GeoTIFF (COG).

    Args:
        obj: Input DataArray / GeoTensor.
        path: Output file path.
        dtype: Output data type (e.g. ``'uint16'``, ``'float32'``).
        overviews: Overview factors (e.g. ``[2, 4, 8, 16]``).
        **kwargs: Additional arguments for COG writing.
    """
    if overviews is None:
        overviews = [2, 4, 8, 16]

    try:
        import rioxarray  # noqa: F401
    except ImportError:
        raise ImportError("rioxarray is required to write COGs.")

    # Ensure data is computed
    if hasattr(obj.data, 'compute'):
        obj = obj.compute()

    # Select a lossless output dtype aligned with the source data.
    if np.issubdtype(obj.dtype, np.floating):
        out_dtype = 'float32'
    else:
        out_dtype = 'uint16'

    # Cast dtype
    if obj.dtype != np.dtype(out_dtype):
        obj = obj.astype(out_dtype)

    # Write as COG with overviews
    obj.rio.to_raster(
        path,
        driver='COG',
        dtype=out_dtype,
        overview_resampling='bilinear',
        overview_levels=overviews,
        **kwargs,
    )


# ====================================================================
# Virtual Mosaic (VRT)
# ====================================================================


def VRT(
    files: List[str],
    method: str = 'median',
    band_mapping: Optional[Dict[str, str]] = None,
) -> GeoTensor:
    """Create a virtual mosaic from multiple files without copying data.

    Args:
        files: List of file paths or URLs.
        method: Compositing method (``'median'``, ``'mean'``, ``'min'``,
                ``'max'``, ``'first'``).
        band_mapping: Optional mapping of band names.

    Returns:
        A :class:`GeoTensor` representing the virtual mosaic.
    """
    if not files:
        raise ValueError("Need at least one file for VRT.")

    # Open all files and align
    tensors = []
    for f in files:
        gt = open_cog(f, band_info=band_mapping)
        tensors.append(gt)

    if len(tensors) == 1:
        return tensors[0]

    # Stack and composite
    from .align import align, AlignmentSpec, SpatialAlignment

    spec = AlignmentSpec(
        spatial=SpatialAlignment(extent='union', method='nearest'),
    )
    aligned = align(*tensors, spec=spec)

    # Stack along new dimension
    stacked = xr.concat(aligned, dim='vrt_src')

    # Apply composite method
    if method == 'median':
        result = stacked.median(dim='vrt_src')
    elif method == 'mean':
        result = stacked.mean(dim='vrt_src')
    elif method == 'min':
        result = stacked.min(dim='vrt_src')
    elif method == 'max':
        result = stacked.max(dim='vrt_src')
    elif method == 'first':
        result = stacked.isel(vrt_src=0)
    else:
        raise ValueError(f"Unknown VRT method: {method}")

    # Restore GeoTensor metadata from first tensor
    return to_geotensor(result, tensors[0])


# ====================================================================
# Lazy computation primitives
# ====================================================================


def chunk(
    obj: xr.DataArray,
    chunks: Dict[str, int],
) -> GeoTensor:
    """Rechunk the underlying array.

    Args:
        obj: Input DataArray / GeoTensor.
        chunks: Mapping of dimension name to chunk size.

    Returns:
        A :class:`GeoTensor` with new chunking.
    """
    result = obj.chunk(chunks)
    return to_geotensor(result, obj)


def compute(obj: xr.DataArray) -> np.ndarray:
    """Compute lazy Dask array and return as a numpy array.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        NumPy :class:`ndarray` with the computed data.
    """
    return obj.values


def load(obj: xr.DataArray) -> GeoTensor:
    """Load lazy data into memory and return as in-memory GeoTensor.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        A :class:`GeoTensor` with in-memory (non-lazy) data.
    """
    result = obj.load()
    return to_geotensor(result, obj)


def persist(obj: xr.DataArray) -> GeoTensor:
    """Persist the Dask graph in distributed memory.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        A :class:`GeoTensor` with persisted (cached) data.
    """
    result = obj.persist()
    return to_geotensor(result, obj)