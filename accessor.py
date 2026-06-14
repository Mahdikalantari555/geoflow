"""GeoTensor accessor registered on :class:`xr.DataArray`.

Provides the ``.geotensor`` accessor on any xarray DataArray, enabling
geospatial, temporal, spectral, and ML operations per the GeoTensor spec.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from rasterio.warp import Resampling

from .core import (
    CRSMismatchError,
    GeoTensor,
    GeoExtent,
    NonDifferentiableError,
    ResolutionMismatchError,
    assert_compatible,
    to_geotensor,
)

# Module implementations (filled in as phases progress)
from . import temporal as _temporal
from . import units as _units
from . import mask as _mask
from . import sampling as _sampling
from . import differentiable as _differentiable
from . import backends as _backends
from . import provenance as _provenance
from . import ml as _ml
from .align import align as _align_fn
from .align import align_to_reference as _align_to_ref_fn


# ====================================================================
# GeoTensor Accessor
# ====================================================================


@xr.register_dataarray_accessor('geotensor')
class GeoTensorAccessor:
    """Accessor providing GeoTensor methods on any :class:`xr.DataArray`.

    If the underlying object is a :class:`GeoTensor`, the accessor
    delegates directly to it.  Otherwise, it provides basic geo-aware
    properties derived from the DataArray's coordinates and attrs.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    @property
    def _gt(self) -> Optional[GeoTensor]:
        """The underlying :class:`GeoTensor` if the object is one."""
        return self._obj if isinstance(self._obj, GeoTensor) else None

    # ================================================================
    # Coordinate System (§5.3)
    # ================================================================

    @property
    def crs(self):
        """Return the CRS from ``attrs['crs']`` or attempt to infer."""
        gt = self._gt
        if gt is not None:
            return gt.crs
        crs_val = self._obj.attrs.get('crs')
        if crs_val is not None:
            import pyproj
            return pyproj.CRS(crs_val) if not isinstance(crs_val, pyproj.CRS) else crs_val
        raise AttributeError(
            "No CRS found in attrs. Set ``obj.attrs['crs'] = pyproj.CRS(...)``."
        )

    @property
    def transform(self):
        """Derive an affine geotransform from ``x`` / ``y`` coordinates."""
        gt = self._gt
        if gt is not None:
            return gt.transform
        import affine
        x = self._obj.coords['x'].values
        y = self._obj.coords['y'].values
        if len(x) < 2 or len(y) < 2:
            return affine.Affine(1.0, 0.0, float(x[0]),
                                 0.0, -1.0, float(y[0]))
        x_res = float(x[1] - x[0])
        y_res = float(y[1] - y[0])
        x_min = float(x[0]) - x_res / 2.0
        y_max = float(y[0]) - y_res / 2.0
        return affine.Affine(x_res, 0.0, x_min,
                             0.0, y_res, y_max)

    @property
    def bounds(self):
        """Return ``(xmin, ymin, xmax, ymax)`` in CRS units."""
        gt = self._gt
        if gt is not None:
            return gt.bounds
        t = self.transform
        h = len(self._obj.coords['y'])
        w = len(self._obj.coords['x'])
        xmin, ymax = t * (0, 0)
        xmax, ymin = t * (w, h)
        return (min(xmin, xmax), min(ymin, ymax),
                max(xmin, xmax), max(ymin, ymax))

    @property
    def resolution(self):
        """Return ``(x_res, y_res)`` in CRS units."""
        gt = self._gt
        if gt is not None:
            return gt.resolution
        t = self.transform
        return (abs(t.a), abs(t.e))

    def latlon(self, lat: float, lon: float) -> GeoTensor:
        """Index by geographic coordinates (lat, lon).

        Transforms ``(lat, lon)`` to the GeoTensor's CRS and selects
        the nearest pixel.
        """
        import pyproj
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        result = self._obj.sel(x=x, y=y, method='nearest')
        return to_geotensor(result, self._obj)

    def slice_bbox(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        crs: str = 'EPSG:4326',
    ) -> GeoTensor:
        """Slice by a bounding box in geographic coordinates."""
        import pyproj
        from pyproj import Transformer
        if crs != 'EPSG:4326':
            from pyproj import Transformer as _Transformer
            _transformer = _Transformer.from_crs('EPSG:4326', crs, always_xy=True)
            west, south = _transformer.transform(west, south)
            east, north = _transformer.transform(east, north)

        try:
            result = self._obj.rio.clip_box(
                minx=west, miny=south,
                maxx=east, maxy=north,
            )
        except Exception:
            x = self._obj.coords['x'].values
            y = self._obj.coords['y'].values
            x_mask = (x >= west) & (x <= east)
            y_mask = (y >= south) & (y <= north)
            result = self._obj.isel(x=x_mask, y=y_mask)

        return to_geotensor(result, self._obj)

    def reproject(
        self,
        target_crs: Any,
        resolution: Optional[float] = None,
        method: str = 'bilinear',
    ) -> GeoTensor:
        """Reproject to a target CRS using rioxarray.

        Raises:
            NonDifferentiableError: If called inside a gradient tape
                (torch.is_grad_enabled) since reprojection is not
                differentiable.
        """
        try:
            import torch
            if torch.is_grad_enabled():
                raise NonDifferentiableError(
                    "Reprojection is not differentiable and cannot be "
                    "called inside a gradient tape. Use .geotensor.detach() "
                    "first to break the computation graph."
                )
        except ImportError:
            pass

        try:
            import rioxarray  # noqa: F401
        except ImportError:
            raise ImportError("rioxarray is required for reprojection.")

        method_value = method
        if isinstance(method, str):
            method_value = Resampling[method.lower()]

        aligned = self._obj.rio.reproject(
            target_crs if isinstance(target_crs, str) else target_crs.to_string(),
            resolution=resolution,
            resampling=method_value,
        )
        return to_geotensor(aligned, self._obj)

    def crs_equals(self, other: Any) -> bool:
        """Check if this CRS equals another."""
        import pyproj
        try:
            other_crs = pyproj.CRS(other) if not isinstance(other, pyproj.CRS) else other
            return self.crs == other_crs
        except Exception:
            return str(self.crs) == str(other)

    def resample(
        self,
        resolution: Optional[float] = None,
        method: str = 'bilinear',
    ) -> GeoTensor:
        """Resample to a different spatial resolution within the same CRS.

        Args:
            resolution: Target resolution in CRS units (metres or degrees).
                        If ``None``, returns a copy without resampling.
            method: Resampling method (``'bilinear'``, ``'nearest'``,
                    ``'cubic'``, ``'average'``).

        Returns:
            A new :class:`GeoTensor` at the target resolution.
        """
        if resolution is None:
            return to_geotensor(self._obj.copy(deep=False), self._obj)

        try:
            import rioxarray  # noqa: F401
        except ImportError:
            raise ImportError("rioxarray is required for resampling.")

        method_value = method
        if isinstance(method, str):
            method_value = Resampling[method.lower()]

        current_res = self.resolution
        x_res, y_res = current_res

        if abs(x_res - resolution) < 1e-10 and abs(y_res - resolution) < 1e-10:
            return to_geotensor(self._obj.copy(deep=False), self._obj)

        reprojected = self._obj.rio.reproject(
            self.crs,
            resolution=resolution,
            resampling=method_value,
        )
        return to_geotensor(reprojected, self._obj)

    # ================================================================
    # Properties that delegate to submodules
    # ================================================================

    @property
    def mask(self) -> _mask.GeoTensorMask:
        """Return the mask associated with this GeoTensor."""
        return _mask.get_mask(self._obj)

    @property
    def provenance(self) -> _provenance.ProvenanceGraph:
        """Return the provenance graph associated with this GeoTensor."""
        return _provenance.get_provenance(self._obj)

    # ================================================================
    # Temporal Operations (§5.4, §7)
    # ================================================================

    @property
    def time_interval(self) -> Optional[pd.Timedelta]:
        """Nominal time spacing (median gap)."""
        return _temporal.time_interval(self._obj)

    @property
    def time_metadata(self) -> Dict[str, Any]:
        """Dict with start, end, acquisition_mode."""
        return _temporal.time_metadata(self._obj)

    def resample_time(
        self,
        rule: str = '1D',
        method: str = 'nearest',
        fill_value: float = np.nan,
    ) -> GeoTensor:
        return _temporal.resample_time(self._obj, rule=rule, method=method, fill_value=fill_value)

    def temporal_interpolate(
        self,
        method: str = 'linear',
        limit: int = 2,
    ) -> GeoTensor:
        return _temporal.temporal_interpolate(self._obj, method=method, limit=limit)

    def temporal_window(
        self,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
    ) -> GeoTensor:
        return _temporal.temporal_window(self._obj, start=start, end=end)

    def temporal_aggregate(
        self,
        period: str = 'M',
        func: str = 'mean',
    ) -> GeoTensor:
        return _temporal.temporal_aggregate(self._obj, period=period, func=func)

    def select_by_event(
        self,
        event_times: Sequence[Union[str, pd.Timestamp]],
        tolerance: str = '1D',
    ) -> GeoTensor:
        return _temporal.select_by_event(self._obj, event_times=event_times, tolerance=tolerance)

    def time_to_relative(self) -> GeoTensor:
        return _temporal.time_to_relative(self._obj)

    # ================================================================
    # Unit Operations (§8)
    # ================================================================

    def convert_units(self, **kwargs: Any) -> GeoTensor:
        return _units.convert_units(self._obj, **kwargs)

    def band_units(self) -> Dict[str, str]:
        return _units.band_units(self._obj)

    # ================================================================
    # Alignment (§9)
    # ================================================================

    def align(
        self,
        *tensors: GeoTensor,
        spec: Optional[Any] = None,
    ) -> List[GeoTensor]:
        return _align_fn(*((self._obj,) + tensors), spec=spec)

    def align_to_reference(
        self,
        *tensors: GeoTensor,
        reference: Optional[GeoTensor] = None,
    ) -> List[GeoTensor]:
        if reference is None:
            reference = self._obj if isinstance(self._obj, GeoTensor) else None
        return _align_to_ref_fn(*tensors, reference=reference)

    # ================================================================
    # Mask Operations (§10)
    # ================================================================

    def where(
        self,
        condition: Any,
        drop: bool = False,
    ) -> GeoTensor:
        return _mask.where(self._obj, condition, drop=drop)

    def combine(
        self,
        other_mask: Any,
        method: str = 'or',
    ) -> GeoTensor:
        return _mask.combine(self._obj, other_mask, method=method)

    def apply_mask(self) -> GeoTensor:
        return _mask.apply_mask(self._obj)

    def trim_masked(self, margin: int = 10) -> GeoTensor:
        return _mask.trim_masked(self._obj, margin=margin)

    def fill(self, **kwargs: Any) -> GeoTensor:
        return _mask.fill(self._obj, **kwargs)

    # ================================================================
    # Sampling Operations (§11)
    # ================================================================

    def sample_tiles(self, **kwargs: Any) -> List[Any]:
        return _sampling.sample_tiles(self._obj, **kwargs)

    def sample_random(self, **kwargs: Any) -> List[Any]:
        return _sampling.sample_random(self._obj, **kwargs)

    def sample_grid(self, **kwargs: Any) -> List[Any]:
        return _sampling.sample_grid(self._obj, **kwargs)

    def sample_centroids(self, **kwargs: Any) -> List[Any]:
        return _sampling.sample_centroids(self._obj, **kwargs)

    def to_patch_dataset(self, **kwargs: Any) -> _sampling.GeoTensorPatchDataset:
        return _sampling.to_patch_dataset(self._obj, **kwargs)

    # ================================================================
    # Differentiable Operations (§12)
    # ================================================================

    def differentiable_resample(
        self,
        target_shape: Tuple[int, int],
        method: str = 'bilinear',
        align_corners: bool = True,
    ) -> GeoTensor:
        return _differentiable.differentiable_resample(
            self._obj, target_shape=target_shape,
            method=method, align_corners=align_corners,
        )

    def detach(self) -> GeoTensor:
        return _differentiable.detach(self._obj)

    # ================================================================
    # I/O Operations (§13)
    # ================================================================

    def to_zarr(self, store: Any, **kwargs: Any) -> None:
        _backends.to_zarr(self._obj, store, **kwargs)

    def to_cog(self, path: str, **kwargs: Any) -> None:
        _backends.to_cog(self._obj, path, **kwargs)

    def compute(self) -> np.ndarray:
        return _backends.compute(self._obj)

    def load(self) -> GeoTensor:
        return _backends.load(self._obj)

    def persist(self) -> GeoTensor:
        return _backends.persist(self._obj)

    def chunk(self, chunks: Dict[str, int]) -> GeoTensor:
        return _backends.chunk(self._obj, chunks)

    # ================================================================
    # ML Operations (§15)
    # ================================================================

    def to_numpy(self) -> np.ndarray:
        return _ml.to_numpy(self._obj)

    def to_pytorch(self, **kwargs: Any) -> Any:
        return _ml.to_pytorch(self._obj, **kwargs)

    def to_tensorflow(self) -> Any:
        return _ml.to_tensorflow(self._obj)

    def to_jax(self) -> Any:
        return _ml.to_jax(self._obj)

    def to_xarray(self) -> xr.DataArray:
        return _ml.to_xarray(self._obj)

    def to_dataloader(self, **kwargs: Any) -> Any:
        return _ml.to_dataloader(self._obj, **kwargs)

    def to_tf_dataset(self, **kwargs: Any) -> Any:
        return _ml.to_tf_dataset(self._obj, **kwargs)

    def compute_normalization(self, **kwargs: Any) -> Dict[str, Any]:
        return _ml.compute_normalization(self._obj, **kwargs)

    def normalize(self, norm_params: Dict[str, Any]) -> GeoTensor:
        return _ml.normalize(self._obj, norm_params)

    def denormalize(self) -> GeoTensor:
        return _ml.denormalize(self._obj)

    # ================================================================
    # Provenance (§14)
    # ================================================================

    def export_provenance(self, format: str = 'json') -> str:
        return _provenance.export_provenance(self._obj, format=format)

    def freeze(self) -> GeoTensor:
        return _provenance.freeze(self._obj)

    def unfreeze(self) -> GeoTensor:
        return _provenance.unfreeze(self._obj)