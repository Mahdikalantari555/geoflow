"""Mask hierarchy for GeoTensor (§10 of GeoTensor Spec).

Provides :class:`GeoTensorMask` with multi-layer masking, propagation
rules, fill/interpolation, and trim operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .core import GeoTensor, to_geotensor


# ====================================================================
# GeoTensorMask
# ====================================================================


class GeoTensorMask:
    """Multi-layer mask for a GeoTensor.

    Each layer is a boolean :class:`xr.DataArray` of the same shape
    (``True`` = invalid / missing).

    Built-in layer keys: ``'nodata'``, ``'cloud'``, ``'shadow'``,
    ``'saturation'``, ``'temporal_gap'``.

    Args:
        shape: Shape tuple used if creating empty masks.
        dims: Dimension names.
        coords: Coordinate dict.
    """

    _KNOWN_LAYERS = ['nodata', 'cloud', 'shadow', 'saturation', 'temporal_gap']

    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        dims: Optional[Tuple[str, ...]] = None,
        coords: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._layers: Dict[str, xr.DataArray] = {}

    def get_layer(self, name: str) -> Optional[xr.DataArray]:
        """Return a specific mask layer.

        Args:
            name: Layer name (e.g. ``'cloud'``).

        Returns:
            Boolean DataArray, or ``None`` if the layer doesn't exist.
        """
        return self._layers.get(name)

    def set_layer(self, name: str, mask_array: xr.DataArray) -> None:
        """Set a mask layer.

        Args:
            name: Layer name.
            mask_array: Boolean DataArray. Must be compatible shape.
        """
        self._layers[name] = mask_array.bool() if mask_array.dtype != bool else mask_array

    def remove_layer(self, name: str) -> None:
        """Remove a mask layer if it exists."""
        self._layers.pop(name, None)

    @property
    def layers(self) -> Dict[str, xr.DataArray]:
        """All mask layers as ``dict[str, xr.DataArray]``."""
        return dict(self._layers)

    @property
    def layer_names(self) -> List[str]:
        """List of layer names."""
        return list(self._layers.keys())

    def combined(self, mode: str = 'any') -> Optional[xr.DataArray]:
        """Combine all layers using the specified logic.

        Args:
            mode: ``'any'`` (logical OR), ``'all'`` (logical AND).

        Returns:
            Combined boolean mask, or ``None`` if no layers exist.
        """
        if not self._layers:
            return None

        masks = list(self._layers.values())

        if mode == 'any':
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m
        elif mode == 'all':
            combined = masks[0]
            for m in masks[1:]:
                combined = combined & m
        else:
            raise ValueError(f"Unknown combine mode: '{mode}'. Use 'any' or 'all'.")

        return combined

    def __repr__(self) -> str:
        layers_repr = ', '.join(
            f"{name}: {arr.dims}" for name, arr in self._layers.items()
        )
        return f'<GeoTensorMask layers=[{layers_repr}]>'


# ====================================================================
# Mask operations — called from GeoTensorAccessor
# ====================================================================


def get_mask(obj: xr.DataArray) -> GeoTensorMask:
    """Retrieve or create a :class:`GeoTensorMask` from attrs.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        The :class:`GeoTensorMask` instance.
    """
    mask = obj.attrs.get('mask')
    if isinstance(mask, GeoTensorMask):
        return mask

    # Create new mask
    new_mask = GeoTensorMask()
    obj.attrs['mask'] = new_mask
    return new_mask


def where(
    obj: xr.DataArray,
    condition: Any,
    drop: bool = False,
) -> GeoTensor:
    """Create a mask from a condition or apply a condition.

    Args:
        obj: Input DataArray / GeoTensor.
        condition: Boolean condition. Can be a callable, DataArray,
                   or numpy array.
        drop: If ``True``, drop fully-masked rows/columns.

    Returns:
        A :class:`GeoTensor` with mask applied.
    """
    if callable(condition):
        mask_arr = condition(obj)
    else:
        mask_arr = xr.DataArray(condition)

    # Convert to boolean
    mask_bool = mask_arr.bool() if mask_arr.dtype != bool else mask_arr

    # Store a layer
    mask_obj = get_mask(obj)
    mask_obj.set_layer('condition', mask_bool)
    obj.attrs['mask'] = mask_obj

    if drop:
        # Drop spatial dims where all True
        result = _drop_fully_masked(obj, mask_bool)
    else:
        result = obj.copy(deep=False)

    return to_geotensor(result, obj)


def combine(
    obj: xr.DataArray,
    other_mask: Any,
    method: str = 'or',
) -> GeoTensor:
    """Combine this mask with another.

    Args:
        obj: Input DataArray / GeoTensor (source of mask).
        other_mask: Another mask (DataArray, GeoTensorMask, or GeoTensor).
        method: Combination logic (``'or'``, ``'and'``, ``'xor'``).

    Returns:
        A :class:`GeoTensor` with the combined mask.
    """
    mask_obj = get_mask(obj)

    # Get the other mask as a boolean DataArray
    if isinstance(other_mask, GeoTensorMask):
        other_combined = other_mask.combined(mode='any')
    elif isinstance(other_mask, xr.DataArray):
        other_combined = other_mask.bool() if other_mask.dtype != bool else other_mask
    else:
        other_combined = None

    if other_combined is None:
        return to_geotensor(obj, obj)

    # Create a 'combined' layer
    current = mask_obj.combined(mode='any')
    if current is None:
        mask_obj.set_layer('combined', other_combined)
    else:
        if method == 'or':
            new_mask = current | other_combined
        elif method == 'and':
            new_mask = current & other_combined
        elif method == 'xor':
            new_mask = current ^ other_combined
        else:
            raise ValueError(f"Unknown method: '{method}'. Use 'or', 'and', 'xor'.")
        mask_obj.set_layer('combined', new_mask)

    obj.attrs['mask'] = mask_obj
    return to_geotensor(obj, obj)


def apply_mask(obj: xr.DataArray) -> GeoTensor:
    """Apply the mask, setting masked cells to NaN.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        A :class:`GeoTensor` with NaN where mask is ``True``.
    """
    mask_obj = get_mask(obj)
    combined = mask_obj.combined(mode='any')
    if combined is None:
        return to_geotensor(obj, obj)

    result = obj.where(~combined)
    return to_geotensor(result, obj)


def trim_masked(obj: xr.DataArray, margin: int = 10) -> GeoTensor:
    """Remove fully-masked rows and columns along spatial dims.

    Args:
        obj: Input DataArray / GeoTensor.
        margin: Number of extra rows/cols to keep around the
                valid-data bounding box.

    Returns:
        A trimmed :class:`GeoTensor`.
    """
    mask_obj = get_mask(obj)
    combined = mask_obj.combined(mode='any')
    if combined is None:
        return to_geotensor(obj, obj)

    # Find valid pixels (not masked)
    valid = ~combined

    # Collapse all non-spatial dims
    for d in valid.dims:
        if d not in ('y', 'x'):
            valid = valid.any(dim=d)

    # Find bounding box of valid data
    y_valid = valid.any(dim='x')
    x_valid = valid.any(dim='y')

    y_inds = np.where(y_valid.values)[0]
    x_inds = np.where(x_valid.values)[0]

    if len(y_inds) == 0 or len(x_inds) == 0:
        # All masked — return empty
        return to_geotensor(obj.isel(y=slice(0, 0), x=slice(0, 0)), obj)

    y_start = max(0, int(y_inds[0]) - margin)
    y_end = min(len(obj.coords['y']), int(y_inds[-1]) + 1 + margin)
    x_start = max(0, int(x_inds[0]) - margin)
    x_end = min(len(obj.coords['x']), int(x_inds[-1]) + 1 + margin)

    result = obj.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
    return to_geotensor(result, obj)


def fill(obj: xr.DataArray, **kwargs: Any) -> GeoTensor:
    """Fill masked values using a specified method or constant.

    Args:
        obj: Input DataArray / GeoTensor.
        **kwargs: Fill parameters:
            - ``value``: Constant value to fill (e.g. ``np.nan``, ``0``).
            - ``method``: Interpolation method (``'nearest'``, ``'linear'``,
              ``'cubic'``). If ``'time'``, fills along the time dimension
              using linear interpolation.
            - ``dim``: Dimension along which to fill (``'x'``, ``'y'``,
              ``'time'``). When ``method='time'``, ``dim`` is
              automatically set to ``'time'``.
            - ``limit``: Max consecutive fill pixels.

    Returns:
        A :class:`GeoTensor` with masked values filled.
    """
    mask_obj = get_mask(obj)
    combined = mask_obj.combined(mode='any')
    if combined is None:
        return to_geotensor(obj, obj)

    value = kwargs.get('value')
    method = kwargs.get('method')
    dim = kwargs.get('dim', 'x')
    limit = kwargs.get('limit', 100)
    order = kwargs.get('order', 1)

    if value is not None:
        result = obj.where(~combined, other=value)
    elif method is not None:
        masked = obj.where(~combined)

        if method == 'time' and 'time' in masked.dims:
            dim = 'time'
            interp_method = 'linear'
        else:
            interp_method = method

        kwargs_interp = {'dim': dim}
        if interp_method and interp_method != 'time':
            kwargs_interp['method'] = interp_method
        if limit is not None:
            kwargs_interp['limit'] = limit

        result = masked.interpolate_na(**kwargs_interp)
    else:
        raise ValueError(
            "Provide either 'value' for constant fill or "
            "'method' for interpolation fill."
        )

    return to_geotensor(result, obj)


# ====================================================================
# Internal helpers
# ====================================================================


def _drop_fully_masked(
    obj: xr.DataArray,
    mask: xr.DataArray,
) -> xr.DataArray:
    """Drop spatial dimensions where all pixels are masked."""
    for d in ('y', 'x'):
        if d in obj.dims:
            # Check if any non-masked pixel exists along this dim
            valid = (~mask).any(dim=[dim for dim in obj.dims if dim != d])
            obj = obj.isel({d: valid})

    return obj