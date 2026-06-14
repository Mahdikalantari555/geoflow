"""Core geospatial tensor abstractions for GeoFlow.

Phase 1 of the GeoFlow implementation plan.
GeoTensor subclasses ``xarray.DataArray`` and provides geospatial metadata
management, CRS handling, band semantics, and spatial coordinate derivation.

Canonical dimension order (per spec §5.2): ``(time, y, x, band)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import affine
import numpy as np
import pandas as pd
import pyproj
import torch
import xarray as xr


# ====================================================================
# Exception Classes
# ====================================================================


class GeoFlowError(Exception):
    """Base exception for all GeoFlow errors."""
    pass


class ResolutionMismatchError(GeoFlowError, ValueError):
    """Raised when two GeoTensors have incompatible spatial resolution."""
    pass


class CRSMismatchError(GeoFlowError, ValueError):
    """Raised when two GeoTensors have incompatible CRS."""
    pass


class UnitMismatchError(GeoFlowError, ValueError):
    """Raised when two GeoTensors have incompatible physical units."""
    pass


class ImmutableGeoTensorError(GeoFlowError, RuntimeError):
    """Raised when attempting to modify a frozen GeoTensor."""
    pass


class NonDifferentiableError(GeoFlowError, RuntimeError):
    """Raised when a non-differentiable operation is attempted
    inside a gradient-requiring context."""
    pass


class DataNotAvailableError(GeoFlowError, FileNotFoundError):
    """Raised when original data sources are not accessible
    during provenance reproduction."""
    pass


# ====================================================================
# Dataclasses
# ====================================================================


@dataclass
class BandInfo:
    """Semantic metadata for a single spectral band.

    Attributes:
        name: Human-readable band name (e.g., ``'B04'``, ``'Red'``).
        wavelength_nm: Central wavelength of the band in nanometres.
        unit: Physical unit of the band values (Pint-compatible string,
              e.g. ``'dimensionless'``, ``'W / m**2 / sr / um'``).
        sensor: Name of the sensor that acquired this band
                (e.g. ``'MSI'``, ``'OLI'``).
        calibration_coeffs: Sensor-specific calibration coefficients
                            (e.g. ``{'gain': 0.01, 'offset': 0.0}``).
        measurement_type: Type of measurement (``'reflectance'``,
                          ``'radiance'``, ``'brightness_temperature'``,
                          ``'backscatter'``, ``'DN'``).
        valid_range: Minimum and maximum valid physical values (inclusive)
                     as ``(min, max)``.
        scale_factor: Multiplicative factor to convert raw DNs to
                      physical units.
        add_offset: Additive offset applied after scaling.
        nodata: Sentinel value representing invalid / missing data.
    """
    name: str
    wavelength_nm: Optional[float] = None
    unit: str = 'dimensionless'
    sensor: Optional[str] = None
    calibration_coeffs: Optional[Dict[str, float]] = None
    measurement_type: str = 'reflectance'
    valid_range: Optional[Tuple[float, float]] = None
    scale_factor: float = 1.0
    add_offset: float = 0.0
    nodata: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate constraints after initialisation."""
        if self.valid_range is not None:
            if self.valid_range[0] >= self.valid_range[1]:
                raise ValueError(
                    f"valid_range must be (min, max) with min < max, "
                    f"got {self.valid_range}"
                )
        if self.scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be positive, got {self.scale_factor}"
            )


@dataclass
class GeoExtent:
    """Rectangular geographic extent with CRS.

    Attributes:
        xmin: Minimum x (west) coordinate in CRS units.
        ymin: Minimum y (south) coordinate in CRS units.
        xmax: Maximum x (east) coordinate in CRS units.
        ymax: Maximum y (north) coordinate in CRS units.
        crs: Coordinate reference system (string or ``pyproj.CRS``).
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs: Union[str, pyproj.CRS] = 'EPSG:4326'

    def __post_init__(self) -> None:
        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

    @property
    def width(self) -> float:
        """East-west extent in CRS units."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """North-south extent in CRS units."""
        return self.ymax - self.ymin

    @property
    def centroid(self) -> Tuple[float, float]:
        """Geographic centroid ``(x, y)``."""
        return (self.xmin + self.width / 2, self.ymin + self.height / 2)

    def contains(self, other: GeoExtent) -> bool:
        """Check whether *other* lies completely inside this extent."""
        return (
            other.xmin >= self.xmin
            and other.xmax <= self.xmax
            and other.ymin >= self.ymin
            and other.ymax <= self.ymax
        )

    def intersects(self, other: GeoExtent) -> bool:
        """Check whether *other* overlaps this extent."""
        return (
            other.xmin < self.xmax
            and other.xmax > self.xmin
            and other.ymin < self.ymax
            and other.ymax > self.ymin
        )


@dataclass
class GeoGrid:
    """Regular grid defining a rectangular raster.

    Attributes:
        width: Number of columns (pixels).
        height: Number of rows (pixels).
        transform: Affine transform mapping pixel corners to world
                   coordinates.
        crs: Coordinate reference system (string or ``pyproj.CRS``).
    """
    width: int
    height: int
    transform: affine.Affine
    crs: Union[str, pyproj.CRS] = 'EPSG:4326'

    def __post_init__(self) -> None:
        if isinstance(self.crs, str):
            self.crs = pyproj.CRS(self.crs)

    @property
    def extent(self) -> GeoExtent:
        """Geographic extent covered by this grid."""
        xmin, ymax = self.transform * (0, 0)
        xmax, ymin = self.transform * (self.width, self.height)
        return GeoExtent(
            xmin=min(xmin, xmax),
            ymin=min(ymin, ymax),
            xmax=max(xmin, xmax),
            ymax=max(ymin, ymax),
            crs=self.crs,
        )

    def align_to(self, target: GeoGrid) -> GeoGrid:
        """Align this grid to match a target grid.

        Returns a new :class:`GeoGrid` whose extent is snapped to the
        target grid such that pixels align perfectly.

        .. todo::

            Implement full pixel-aligned snapping logic using GCD
            of resolutions and common origin.
        """
        raise NotImplementedError("TODO: Phase 4 - align.py")


# ====================================================================
# Validation Helpers
# ====================================================================


def assert_compatible(*tensors: GeoTensor,
                      check_crs: bool = True,
                      check_resolution: bool = True,
                      check_extent: bool = True) -> None:
    """Verify that all GeoTensors have compatible CRS, resolution, and extent.

    Args:
        *tensors: Two or more GeoTensor instances to compare.
        check_crs: If ``True``, verify CRS equality.
        check_resolution: If ``True``, verify resolution equality.
        check_extent: If ``True``, verify that extents overlap.

    Raises:
        CRSMismatchError: If CRS values differ.
        ResolutionMismatchError: If resolutions differ.
        ValueError: If extents do not overlap.
    """
    from itertools import combinations

    for a, b in combinations(tensors, 2):
        if check_crs and a.crs != b.crs:
            raise CRSMismatchError(
                f"CRS mismatch: '{a.crs}' vs '{b.crs}'"
            )
        if check_resolution:
            rx_a, ry_a = a.resolution
            rx_b, ry_b = b.resolution
            if abs(rx_a - rx_b) > 1e-9 or abs(ry_a - ry_b) > 1e-9:
                raise ResolutionMismatchError(
                    f"Resolution mismatch: ({rx_a}, {ry_a}) vs "
                    f"({rx_b}, {ry_b})"
                )
        if check_extent:
            ba = a.bounds
            bb = b.bounds
            x_overlap = ba[0] < bb[2] and bb[0] < ba[2]
            y_overlap = ba[1] < bb[3] and bb[1] < ba[3]
            if not (x_overlap and y_overlap):
                raise ValueError(
                    f"Extents do not overlap: {ba} vs {bb}"
                )


# ====================================================================
# GeoTensor — Core Array Type
# ====================================================================


class GeoTensor(xr.DataArray):
    """GeoFlow's core array type, subclassing :class:`xarray.DataArray`.

    GeoTensor extends xarray's labelled multi-dimensional arrays with
    geospatial metadata: CRS, affine geotransform, band semantics,
    mask, and provenance tracking.

    **Canonical dimension order** (per spec §5.2): ``(time, y, x, band)``.
    At minimum, ``y`` and ``x`` dimensions must be present for spatial
    operations.

    **Metadata storage** (in ``self.attrs``):

    * ``'crs'`` — :class:`pyproj.CRS`
    * ``'band_semantics'`` — ``dict[str, BandInfo]``
    * ``'mask'`` — optional :class:`xr.DataArray`
    * ``'provenance'`` — provenance graph (Phase 9)
    * ``'geotensor_version'`` — version string

    Args:
        data: Input array (numpy, dask, or array-like).
        coords: Coordinate pairs (see :class:`xr.DataArray`).
        dims: Dimension names.
        crs: Coordinate reference system (EPSG string, WKT, or
             :class:`pyproj.CRS`). Stored in ``attrs['crs']``.
        band_info: Band metadata. Accepts a ``dict[str, BandInfo]``,
                   a ``list[BandInfo]``, or a single ``BandInfo``.
                   Stored in ``attrs['band_semantics']``.
        mask: Boolean mask array as an ``xr.DataArray``.
              Stored in ``attrs['mask']``.
        provenance: Provenance reference.
                    Stored in ``attrs['provenance']``.
        name: DataArray name.
        attrs: Additional attributes merged into ``self.attrs``.
        **kwargs: Additional arguments passed to
                  :class:`xr.DataArray.__init__`.

    Note:
        All geospatial metadata is stored in ``self.attrs`` so that
        xarray's internal copy/merge mechanisms preserve it
        automatically.
    """
    __slots__ = ()

    # Metadata is stored in self.attrs so operations preserve it.

    def __init__(
        self,
        data: Any,
        coords: Optional[Union[Dict, Sequence]] = None,
        dims: Optional[Union[str, Sequence[str]]] = None,
        crs: Optional[Union[str, pyproj.CRS]] = None,
        band_info: Optional[Union[Dict[str, 'BandInfo'],
                                  List['BandInfo'],
                                  'BandInfo']] = None,
        mask: Optional[xr.DataArray] = None,
        provenance: Optional[Any] = None,
        name: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # When xarray calls us via _replace(fastpath=True) we must not
        # re-process attrs or inject geo-metadata — everything is
        # already preserved in self.attrs from the previous instance.
        if kwargs.pop('fastpath', False):
            super().__init__(data, coords=coords, dims=dims,
                             name=name, attrs=attrs,
                             fastpath=True, **kwargs)
            return

        # Build merged attrs dict
        _attrs = dict(attrs) if attrs else {}

        if crs is not None:
            _attrs['crs'] = (
                pyproj.CRS(crs) if not isinstance(crs, pyproj.CRS) else crs
            )
        if band_info is not None:
            _attrs['band_semantics'] = _normalise_band_info(band_info)
        if mask is not None:
            _attrs['mask'] = mask
        if provenance is not None:
            _attrs['provenance'] = provenance

        _attrs.setdefault('geotensor_version', '0.0.1')

        super().__init__(
            data,
            coords=coords,
            dims=dims,
            name=name,
            attrs=_attrs,
            **kwargs,
        )

    # ---- Properties ---------------------------------------------------

    @property
    def crs(self) -> pyproj.CRS:
        """Coordinate Reference System.

        Returns:
            :class:`pyproj.CRS` stored in ``attrs['crs']``.

        Raises:
            AttributeError: If no CRS has been set.
        """
        crs_val = self.attrs.get('crs')
        if crs_val is None:
            raise AttributeError(
                "GeoTensor has no CRS set. Assign one via "
                "``gt.attrs['crs'] = pyproj.CRS(...)``."
            )
        return pyproj.CRS(crs_val) if not isinstance(crs_val, pyproj.CRS) else crs_val

    @crs.setter
    def crs(self, value: Union[str, pyproj.CRS]) -> None:
        self.attrs['crs'] = pyproj.CRS(value) if not isinstance(value, pyproj.CRS) else value

    @property
    def transform(self) -> affine.Affine:
        """Affine geotransform derived from ``x`` and ``y`` coordinates.

        Computes the standard rasterio-style affine transform from
        1-D pixel-centre coordinate arrays. The transform maps
        ``(col, row)`` pixel corners to ``(x, y)`` world coordinates.

        Returns:
            :class:`affine.Affine` with parameters
            ``(x_res, 0.0, x_min, 0.0, y_res, y_max)``.
        """
        x = self.coords['x'].values
        y = self.coords['y'].values

        if len(x) < 2 or len(y) < 2:
            return affine.Affine(1.0, 0.0, float(x[0]),
                                 0.0, -1.0, float(y[0]))

        x_res = float(x[1] - x[0])
        y_res = float(y[1] - y[0])

        # Pixel-centre coordinates -> corner-based affine
        x_min = float(x[0]) - x_res / 2.0
        y_max = float(y[0]) - y_res / 2.0  # works for both north-up and south-up

        return affine.Affine(x_res, 0.0, x_min,
                             0.0, y_res, y_max)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Geographic bounds as ``(xmin, ymin, xmax, ymax)``.

        Derived from the ``x`` and ``y`` coordinate arrays and the
        affine transform.
        """
        t = self.transform
        h, w = len(self.coords['y']), len(self.coords['x'])
        xmin, ymax = t * (0, 0)
        xmax, ymin = t * (w, h)
        return (min(xmin, xmax), min(ymin, ymax),
                max(xmin, xmax), max(ymin, ymax))

    @property
    def resolution(self) -> Tuple[float, float]:
        """Spatial resolution as ``(x_res, y_res)`` in CRS units.

        The absolute value is returned so resolution is always positive.
        """
        t = self.transform
        return (abs(t.a), abs(t.e))

    @property
    def band_names(self) -> List[str]:
        """List of band names from the ``band`` coordinate."""
        if 'band' in self.coords:
            return [str(b) for b in self.coords['band'].values]
        return []

    @property
    def time_coords(self) -> Optional[pd.DatetimeIndex]:
        """Time coordinates as a :class:`pd.DatetimeIndex`, or ``None``."""
        if 'time' in self.coords:
            vals = self.coords['time'].values
            return pd.DatetimeIndex(vals)
        return None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Full shape tuple of the underlying array."""
        return super().shape

    @property
    def geotensor_version(self) -> str:
        """Version string of the GeoTensor implementation."""
        return str(self.attrs.get('geotensor_version', '2.0.0'))

    # ---- Constructors -------------------------------------------------

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        transform: affine.Affine,
        crs: Union[str, pyproj.CRS],
        dims: Optional[Tuple[str, ...]] = None,
        band_names: Optional[List[str]] = None,
        time_coords: Optional[Union[pd.DatetimeIndex,
                                    List[str]]] = None,
        band_info: Optional[Union[Dict[str, 'BandInfo'],
                                  List['BandInfo'],
                                  'BandInfo']] = None,
        mask: Optional[xr.DataArray] = None,
        provenance: Optional[Any] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> 'GeoTensor':
        """Construct a :class:`GeoTensor` from a numpy array and
        geotransform.

        Args:
            data: Numpy array of shape compatible with *dims*.
            transform: :class:`affine.Affine` geotransform.
            crs: Coordinate Reference System.
            dims: Dimension names. Defaults are inferred from
                  ``data.ndim``:
                  - 2-D → ``('y', 'x')``
                  - 3-D → ``('band', 'y', 'x')``
                  - 4-D → ``('time', 'y', 'x', 'band')``
            band_names: Optional list of names for the band dimension.
            time_coords: Optional time coordinates for the time dimension.
            band_info: Optional band semantic metadata.
            mask: Optional boolean mask.
            provenance: Optional provenance reference.
            name: Optional DataArray name.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A new :class:`GeoTensor`.
        """
        data = np.asarray(data)
        ndim = data.ndim

        # Infer dims if not provided
        if dims is None:
            if ndim == 2:
                dims = ('y', 'x')
            elif ndim == 3:
                dims = ('band', 'y', 'x')
            elif ndim == 4:
                dims = ('time', 'y', 'x', 'band')
            else:
                raise ValueError(
                    f"Cannot infer dims for {ndim}-D array. "
                    f"Please provide explicit dims."
                )

        if 'y' not in dims or 'x' not in dims:
            raise ValueError("dims must contain 'y' and 'x'.")

        # Build spatial coordinates from transform
        y_idx = dims.index('y')
        x_idx = dims.index('x')
        height = data.shape[y_idx]
        width = data.shape[x_idx]

        # Pixel centre coordinates
        x_coords = np.array([
            transform.c + (i + 0.5) * transform.a
            for i in range(width)
        ], dtype=np.float64)
        y_coords = np.array([
            transform.f + (j + 0.5) * transform.e
            for j in range(height)
        ], dtype=np.float64)

        coords: Dict[str, Any] = {
            'x': ('x', x_coords),
            'y': ('y', y_coords),
        }

        # Handle band dimension
        if 'band' in dims:
            band_idx = dims.index('band')
            nbands = data.shape[band_idx]
            if band_names is not None:
                coords['band'] = ('band', list(band_names))
            else:
                coords['band'] = ('band', [f'B{i+1:02d}' for i in range(nbands)])

        # Handle time dimension
        if 'time' in dims:
            if time_coords is not None:
                coords['time'] = ('time', pd.DatetimeIndex(time_coords))
            else:
                ntimes = data.shape[dims.index('time')]
                coords['time'] = (
                    'time',
                    pd.date_range('2000-01-01', periods=ntimes),
                )

        return cls(
            data,
            coords=coords,
            dims=dims,
            crs=crs,
            band_info=band_info,
            mask=mask,
            provenance=provenance,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_torch(
        cls,
        tensor: torch.Tensor,
        transform: affine.Affine,
        crs: Union[str, pyproj.CRS],
        dims: Optional[Tuple[str, ...]] = None,
        band_names: Optional[List[str]] = None,
        time_coords: Optional[Union[pd.DatetimeIndex,
                                    List[str]]] = None,
        band_info: Optional[Union[Dict[str, 'BandInfo'],
                                  List['BandInfo'],
                                  'BandInfo']] = None,
        mask: Optional[xr.DataArray] = None,
        provenance: Optional[Any] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> 'GeoTensor':
        """Construct a :class:`GeoTensor` from a :class:`torch.Tensor`
        and geotransform.

        The torch tensor is detached and moved to CPU before conversion.

        Args:
            tensor: PyTorch tensor.
            transform: :class:`affine.Affine` geotransform.
            crs: Coordinate Reference System.
            dims: Dimension names (see :meth:`from_array`).
            band_names: Optional band names.
            time_coords: Optional time coordinates.
            band_info: Optional band semantic metadata.
            mask: Optional boolean mask.
            provenance: Optional provenance reference.
            name: Optional DataArray name.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            A new :class:`GeoTensor`.
        """
        # Detach and move to CPU numpy
        data = tensor.detach().cpu().numpy()
        return cls.from_array(
            data,
            transform=transform,
            crs=crs,
            dims=dims,
            band_names=band_names,
            time_coords=time_coords,
            band_info=band_info,
            mask=mask,
            provenance=provenance,
            name=name,
            **kwargs,
        )

    # ---- Display ------------------------------------------------------

    def __repr__(self) -> str:
        try:
            crs_str = str(self.crs.to_epsg() or self.crs.srs)
        except Exception:
            crs_str = 'unknown'
        try:
            b = self.bounds
            bounds_str = f"({b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f})"
        except Exception:
            bounds_str = 'N/A'
        return (
            f'<{type(self).__name__} '
            f'shape={self.shape} '
            f'dims={self.dims} '
            f'crs={crs_str} '
            f'bounds={bounds_str}>'
        )


# ====================================================================
# Internal Helpers
# ====================================================================


def _normalise_band_info(
    band_info: Union[Dict[str, BandInfo], List[BandInfo], BandInfo],
) -> Dict[str, BandInfo]:
    """Convert various ``band_info`` inputs to a canonical
    ``dict[str, BandInfo]``."""
    if isinstance(band_info, BandInfo):
        return {band_info.name: band_info}
    if isinstance(band_info, (list, tuple)):
        return {b.name: b for b in band_info}
    if isinstance(band_info, dict):
        # Ensure all values are BandInfo
        result: Dict[str, BandInfo] = {}
        for k, v in band_info.items():
            if isinstance(v, BandInfo):
                result[k] = v
            else:
                result[k] = BandInfo(name=str(k))
        return result
    raise TypeError(
        f"Expected dict, list, or BandInfo, got {type(band_info)}"
    )


def to_geotensor(
    result: xr.DataArray,
    original: xr.DataArray,
) -> GeoTensor:
    """Convert an xarray result to GeoTensor preserving attrs."""
    preserved = {
        k: original.attrs.get(k)
        for k in ['crs', 'band_semantics', 'mask', 'provenance', 'geotensor_version']
        if k in original.attrs
    }
    merged = {**result.attrs, **preserved}
    if isinstance(result, GeoTensor):
        return result
    return GeoTensor(
        result.data,
        coords=result.coords,
        dims=result.dims,
        attrs=merged,
        name=result.name,
    )
