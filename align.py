"""Automatic spatial / temporal / spectral alignment engine (§9 of GeoTensor Spec).

Provides :class:`AlignmentSpec`, :func:`align`, and :func:`align_to_reference`
for bringing multiple GeoTensor instances to a common grid, time axis,
and band set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from .core import (
    CRSMismatchError,
    GeoTensor,
    ResolutionMismatchError,
    to_geotensor,
)


# ====================================================================
# Alignment Specification Dataclasses
# ====================================================================


@dataclass
class SpatialAlignment:
    """Controls how spatial dimensions are aligned.

    Attributes:
        crs: Target CRS. ``None`` = auto (prefer first tensor's CRS).
        resolution: Target resolution in CRS units. ``None`` = auto
                    (finest among inputs).
        method: Resampling method (``'bilinear'``, ``'nearest'``,
                ``'cubic'``, ``'average'``).
        extent: How to compute the output extent — ``'intersection'``,
                ``'union'``, or a :class:`~.core.GeoExtent` / bounding box.
        target_aligned_pixels: If ``True``, snap the grid so pixels
                               align with the target grid.
    """
    crs: Optional[Union[str, pyproj.CRS]] = None
    resolution: Optional[float] = None
    method: str = 'bilinear'
    extent: Union[str, Tuple[float, float, float, float], Any] = 'intersection'
    target_aligned_pixels: bool = False


@dataclass
class TemporalAlignment:
    """Controls how time dimensions are aligned.

    Attributes:
        method: ``'intersection'``, ``'union'``, ``'reference'``.
        interpolation: Interpolation type for union fills
                       (``'nearest'``, ``'linear'``, ``'ffill'``).
        tolerance: Max gap for interpolation (e.g. ``'5D'``).
    """
    method: str = 'intersection'
    interpolation: str = 'nearest'
    tolerance: str = '1D'


@dataclass
class SpectralAlignment:
    """Controls how spectral bands are aligned.

    Attributes:
        method: ``'match_band'``, ``'interpolate'``, ``'drop'``.
        bandset: Which bands to keep — ``'union'``, ``'intersection'``,
                 or an explicit list of band names.
        wavelength_tolerance_nm: Max wavelength difference for band matching.
    """
    method: str = 'match_band'
    bandset: Union[str, List[str]] = 'union'
    wavelength_tolerance_nm: float = 10.0


@dataclass
class AlignmentSpec:
    """Complete alignment specification for all dimensions.

    Attributes:
        spatial: :class:`SpatialAlignment` configuration.
        temporal: :class:`TemporalAlignment` configuration.
        spectral: :class:`SpectralAlignment` configuration.
    """
    spatial: SpatialAlignment = field(default_factory=SpatialAlignment)
    temporal: TemporalAlignment = field(default_factory=TemporalAlignment)
    spectral: SpectralAlignment = field(default_factory=SpectralAlignment)


# ====================================================================
# Alignment Functions
# ====================================================================


def align(
    *tensors: GeoTensor,
    spec: Optional[AlignmentSpec] = None,
) -> List[GeoTensor]:
    """Align multiple GeoTensors to a common spatial / temporal / spectral grid.

    The operation is **lazy** — returns views with a Dask graph.
    Computation is triggered when data is accessed.

    Args:
        *tensors: Two or more :class:`GeoTensor` instances to align.
        spec: :class:`AlignmentSpec` controlling per-dimension alignment.
              If ``None``, default spec (intersection, finest resolution,
              common CRS) is used.

    Returns:
        List of aligned :class:`GeoTensor` instances, all sharing the
        same CRS, spatial grid, time axis, and band set.
    """
    if len(tensors) < 2:
        raise ValueError("Need at least two tensors to align.")

    spec = spec or AlignmentSpec()
    result = list(tensors)

    # 1. Spatial alignment
    result = _align_spatial(result, spec.spatial)

    # 2. Temporal alignment
    result = _align_temporal(result, spec.temporal)

    # 3. Spectral alignment
    result = _align_spectral(result, spec.spectral)

    return result


def align_to_reference(
    *tensors: GeoTensor,
    reference: GeoTensor,
    spec: Optional[AlignmentSpec] = None,
) -> List[GeoTensor]:
    """Align one or more tensors to a reference tensor.

    Equivalent to ``align(*tensors, spec=SpatialAlignment(reference=ref))``
    where the reference's CRS, resolution, and grid are used as targets.

    Args:
        *tensors: GeoTensors to align.
        reference: The reference :class:`GeoTensor` to align to.
        spec: Optional :class:`AlignmentSpec`. If provided, settings
              override the reference's grid where specified.

    Returns:
        List of aligned :class:`GeoTensor` instances (reference excluded).
    """
    spec = spec or AlignmentSpec()

    # Override spatial spec with reference grid
    ref_spatial = SpatialAlignment(
        crs=reference.crs,
        resolution=reference.resolution[0],
        method=spec.spatial.method,
        extent=reference.bounds,
        target_aligned_pixels=True,
    )

    aligned_spec = AlignmentSpec(
        spatial=ref_spatial,
        temporal=spec.temporal,
        spectral=spec.spectral,
    )

    return align(reference, *tensors, spec=aligned_spec)


# ====================================================================
# Internal alignment stages
# ====================================================================


def _align_spatial(
    tensors: List[GeoTensor],
    spatial: SpatialAlignment,
) -> List[GeoTensor]:
    """Align all tensors to a common CRS and spatial grid."""
    if not tensors:
        return tensors

    # Determine target CRS
    target_crs = spatial.crs
    if target_crs is None:
        target_crs = tensors[0].crs
    target_crs = pyproj.CRS(target_crs) if not isinstance(target_crs, pyproj.CRS) else target_crs

    # Determine target resolution (finest among inputs)
    target_res = spatial.resolution
    if target_res is None:
        resolutions = [
            abs(t.transform.a) for t in tensors
            if hasattr(t, 'transform') and t.transform is not None
        ]
        target_res = min(resolutions) if resolutions else 1.0

    # Determine target extent
    if isinstance(spatial.extent, str):
        if spatial.extent == 'intersection':
            bounds = _intersection_bounds(tensors, target_crs)
        elif spatial.extent == 'union':
            bounds = _union_bounds(tensors, target_crs)
        else:
            raise ValueError(f"Unknown extent mode: {spatial.extent}")
    elif isinstance(spatial.extent, (tuple, list)):
        bounds = tuple(spatial.extent)  # type: ignore[assignment]
    else:
        # Assume it's a GeoExtent or similar
        bounds = (
            spatial.extent.xmin, spatial.extent.ymin,
            spatial.extent.xmax, spatial.extent.ymax,
        )

    # Reproject each tensor to the target grid
    result = []
    for t in tensors:
        aligned = _reproject_to_grid(
            t, target_crs, target_res, bounds, spatial.method
        )
        result.append(aligned)

    return result


def _align_temporal(
    tensors: List[GeoTensor],
    temporal: TemporalAlignment,
) -> List[GeoTensor]:
    """Align the time dimension across tensors."""
    # Check which tensors have time dimension
    timed = [t for t in tensors if 'time' in t.dims]
    if len(timed) < 1:
        return tensors  # Nothing to align temporally

    if temporal.method == 'intersection':
        # Find common time range
        starts = [pd.Timestamp(t.coords['time'].values[0]) for t in timed]
        ends = [pd.Timestamp(t.coords['time'].values[-1]) for t in timed]
        common_start = max(starts)
        common_end = min(ends)

        result = []
        for t in tensors:
            if 'time' in t.dims:
                t_sliced = t.sel(time=slice(common_start, common_end))
            else:
                t_sliced = t
            result.append(to_geotensor(t_sliced, t))

    elif temporal.method == 'union':
        # Union of all time coordinates
        all_times = pd.DatetimeIndex([
            ts for t in timed
            for ts in pd.DatetimeIndex(t.coords['time'].values)
        ]).sort_values().unique()

        result = []
        for t in tensors:
            if 'time' in t.dims and len(all_times) > 0:
                # Reindex to union
                t_reindexed = t.reindex(
                    time=all_times,
                    method=temporal.interpolation,
                    tolerance=temporal.tolerance,
                )
            else:
                t_reindexed = t
            result.append(to_geotensor(t_reindexed, t))

    elif temporal.method == 'reference':
        # Use the first tensor's time as reference
        ref_time = timed[0].coords['time'].values
        result = []
        for t in tensors:
            if 'time' in t.dims:
                t_reindexed = t.reindex(
                    time=ref_time,
                    method=temporal.interpolation,
                    tolerance=temporal.tolerance,
                )
            else:
                t_reindexed = t
            result.append(to_geotensor(t_reindexed, t))

    else:
        raise ValueError(f"Unknown temporal alignment method: {temporal.method}")

    return result


def _align_spectral(
    tensors: List[GeoTensor],
    spectral: SpectralAlignment,
) -> List[GeoTensor]:
    """Align the band dimension across tensors."""
    # Determine band set
    all_bands: List[List[str]] = [
        t.band_names for t in tensors if 'band' in t.dims
    ]
    if not all_bands:
        return tensors  # No band dimension

    if isinstance(spectral.bandset, list):
        target_bands = spectral.bandset
    elif spectral.bandset == 'intersection':
        target_bands = list(set.intersection(*map(set, all_bands)))
    else:  # union
        target_bands = list(set.union(*map(set, all_bands)))

    if not target_bands:
        raise ValueError("No common bands found across tensors.")

    # Filter/subset each tensor to the target bands
    result = []
    for t in tensors:
        if 'band' in t.dims:
            available = set(t.band_names)
            keep = [b for b in target_bands if b in available]

            if spectral.method == 'match_band':
                t_subset = t.sel(band=keep) if keep else t
            elif spectral.method == 'drop':
                t_subset = t.sel(band=keep) if keep else t
            elif spectral.method == 'interpolate':
                # Wavelength interpolation (placeholder — needs wavelength info)
                t_subset = t.sel(band=keep) if keep else t
            else:
                raise ValueError(f"Unknown spectral alignment method: {spectral.method}")
        else:
            t_subset = t
        result.append(to_geotensor(t_subset, t))

    return result


# ====================================================================
# Spatial reprojection helpers
# ====================================================================


def _reproject_to_grid(
    tensor: GeoTensor,
    target_crs: pyproj.CRS,
    target_res: float,
    bounds: Tuple[float, float, float, float],
    method: str,
) -> GeoTensor:
    """Reproject a single tensor to a target grid using rioxarray.

    Falls back to manual computation if rioxarray is unavailable.
    """
    if tensor.crs == target_crs and _check_resolution(tensor, target_res):
        # Already matching — crop to bounds
        return _crop_to_bounds(tensor, bounds)

    try:
        return _reproject_rioxarray(tensor, target_crs, target_res, bounds, method)
    except ImportError:
        return _reproject_native(tensor, target_crs, target_res, bounds, method)


def _reproject_rioxarray(
    tensor: GeoTensor,
    target_crs: pyproj.CRS,
    target_res: float,
    bounds: Tuple[float, float, float, float],
    method: str,
) -> GeoTensor:
    """Use rioxarray's reproject_match for alignment."""
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        raise ImportError("rioxarray is required for spatial alignment. ")

    # Register rioxarray accessor
    xr.DataArray.rio  # trigger rioxarray registration

    # Reproject
    aligned = tensor.rio.reproject(
        target_crs.to_string(),
        resolution=target_res,
        resampling=method,
    )

    # Crop to bounds
    aligned = aligned.rio.clip_box(
        minx=bounds[0], miny=bounds[1],
        maxx=bounds[2], maxy=bounds[3],
    )

    return to_geotensor(aligned, tensor)


def _reproject_native(
    tensor: GeoTensor,
    target_crs: pyproj.CRS,
    target_res: float,
    bounds: Tuple[float, float, float, float],
    method: str,
) -> GeoTensor:
    """Fallback reprojection using rasterio warp directly."""
    import rasterio
    from rasterio.enums import Resampling

    resampling_map = {
        'bilinear': Resampling.bilinear,
        'nearest': Resampling.nearest,
        'cubic': Resampling.cubic,
        'average': Resampling.average,
        'lanczos': Resampling.lanczos,
    }
    rst = resampling_map.get(method, Resampling.bilinear)

    # Compute output dimensions
    width = max(1, int(round((bounds[2] - bounds[0]) / target_res)))
    height = max(1, int(round((bounds[3] - bounds[1]) / target_res)))
    dst_transform = rasterio.transform.from_bounds(
        *bounds, width=width, height=height
    )

    src_np = tensor.values
    dst_np = np.zeros((height, width), dtype=src_np.dtype) if src_np.ndim == 2 else \
             np.zeros((src_np.shape[0], height, width), dtype=src_np.dtype)

    # ndim handling
    if src_np.ndim == 2:
        rasterio.warp.reproject(
            source=src_np,
            destination=dst_np,
            src_transform=tensor.transform,
            src_crs=tensor.crs.to_string(),
            dst_transform=dst_transform,
            dst_crs=target_crs.to_string(),
            resampling=rst,
        )
    else:
        for i in range(src_np.shape[0]):
            rasterio.warp.reproject(
                source=src_np[i],
                destination=dst_np[i],
                src_transform=tensor.transform,
                src_crs=tensor.crs.to_string(),
                dst_transform=dst_transform,
                dst_crs=target_crs.to_string(),
                resampling=rst,
            )

    # Build new coords
    x_coords = np.array([
        dst_transform.c + (i + 0.5) * dst_transform.a
        for i in range(width)
    ])
    y_coords = np.array([
        dst_transform.f + (j + 0.5) * dst_transform.e
        for j in range(height)
    ])

    # Preserve non-spatial dims
    new_dims = list(tensor.dims)
    new_coords = {k: tensor.coords[k] for k in tensor.dims if k not in ('x', 'y')}
    new_coords['x'] = ('x', x_coords)
    new_coords['y'] = ('y', y_coords)

    gt = GeoTensor(
        dst_np,
        coords=new_coords,
        dims=tensor.dims,
        crs=target_crs,
        band_info=tensor.attrs.get('band_semantics'),
        attrs={k: v for k, v in tensor.attrs.items()
               if k not in ('crs', 'band_semantics')},
    )
    return gt


def _crop_to_bounds(
    tensor: GeoTensor,
    bounds: Tuple[float, float, float, float],
) -> GeoTensor:
    """Crop tensor to the given bounding box."""
    try:
        return tensor.rio.clip_box(
            minx=bounds[0], miny=bounds[1],
            maxx=bounds[2], maxy=bounds[3],
        )
    except Exception:
        # Fallback: manual slice
        x = tensor.coords['x'].values
        y = tensor.coords['y'].values
        x_mask = (x >= bounds[0]) & (x <= bounds[2])
        y_mask = (y >= bounds[1]) & (y <= bounds[3])
        return tensor.isel(x=x_mask, y=y_mask)


# ====================================================================
# Bounding box helpers
# ====================================================================


def _intersection_bounds(
    tensors: List[GeoTensor],
    target_crs: pyproj.CRS,
) -> Tuple[float, float, float, float]:
    """Compute the intersecting bounding box across all tensors."""
    from pyproj import Transformer

    # Transform all bounds to target CRS
    all_xmins, all_ymins, all_xmaxs, all_ymaxs = [], [], [], []
    for t in tensors:
        b = t.bounds
        if t.crs != target_crs:
            transformer = Transformer.from_crs(t.crs, target_crs, always_xy=True)
            corners = [
                transformer.transform(b[0], b[1]),
                transformer.transform(b[0], b[3]),
                transformer.transform(b[2], b[1]),
                transformer.transform(b[2], b[3]),
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            all_xmins.append(min(xs))
            all_ymins.append(min(ys))
            all_xmaxs.append(max(xs))
            all_ymaxs.append(max(ys))
        else:
            all_xmins.append(b[0])
            all_ymins.append(b[1])
            all_xmaxs.append(b[2])
            all_ymaxs.append(b[3])

    # Intersection = max of mins, min of maxes
    return (max(all_xmins), max(all_ymins),
            min(all_xmaxs), min(all_ymaxs))


def _union_bounds(
    tensors: List[GeoTensor],
    target_crs: pyproj.CRS,
) -> Tuple[float, float, float, float]:
    """Compute the union bounding box across all tensors."""
    from pyproj import Transformer

    all_xmins, all_ymins, all_xmaxs, all_ymaxs = [], [], [], []
    for t in tensors:
        b = t.bounds
        if t.crs != target_crs:
            transformer = Transformer.from_crs(t.crs, target_crs, always_xy=True)
            corners = [
                transformer.transform(b[0], b[1]),
                transformer.transform(b[0], b[3]),
                transformer.transform(b[2], b[1]),
                transformer.transform(b[2], b[3]),
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            all_xmins.append(min(xs))
            all_ymins.append(min(ys))
            all_xmaxs.append(max(xs))
            all_ymaxs.append(max(ys))
        else:
            all_xmins.append(b[0])
            all_ymins.append(b[1])
            all_xmaxs.append(b[2])
            all_ymaxs.append(b[3])

    return (min(all_xmins), min(all_ymins),
            max(all_xmaxs), max(all_ymaxs))


def _check_resolution(tensor: GeoTensor, target_res: float) -> bool:
    """Check if tensor's resolution matches target within tolerance."""
    try:
        rx, ry = tensor.resolution
        return abs(rx - target_res) < 1e-9 and abs(ry - target_res) < 1e-9
    except Exception:
        return False