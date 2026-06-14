"""Temporal operations for GeoTensor (§5.4, §7 of GeoTensor Spec).

Provides time coordinate management, resampling, interpolation,
aggregation, and event-based sampling on :class:`xr.DataArray` objects
through the ``.geotensor`` accessor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .core import GeoTensor, to_geotensor


# ====================================================================
# Temporal operations — called from GeoTensorAccessor
# ====================================================================


def resample_time(
    obj: xr.DataArray,
    rule: str = '1D',
    method: str = 'nearest',
    fill_value: float = np.nan,
) -> GeoTensor:
    """Resample the time dimension to a regular frequency.

    Args:
        obj: Input DataArray / GeoTensor.
        rule: Resampling rule (e.g. ``'1D'``, ``'3h'``, ``'MS'``).
        method: Filling method (``'nearest'``, ``'ffill'``, ``'bfill'``,
                ``'linear'``).
        fill_value: Fallback for values that cannot be filled.

    Returns:
        A new :class:`GeoTensor` with resampled time axis.
    """
    if 'time' not in obj.dims:
        raise ValueError("Data has no 'time' dimension.")

    if obj.chunks is not None:
        obj = obj.load()

    resampled = obj.resample(time=rule)

    if method == 'nearest':
        result = resampled.nearest()
    elif method == 'ffill':
        result = resampled.ffill()
    elif method == 'bfill':
        result = resampled.bfill()
    elif method == 'linear':
        result = resampled.interpolate('linear')
    elif method == 'pad':
        result = resampled.pad()
    else:
        raise ValueError(f"Unknown fill method: {method}")

    if fill_value is not None and not np.isnan(fill_value):
        result = result.fillna(fill_value)

    return to_geotensor(result, obj)


def temporal_interpolate(
    obj: xr.DataArray,
    method: str = 'linear',
    limit: int = 2,
) -> GeoTensor:
    """Interpolate missing values along the time dimension.

    Args:
        obj: Input DataArray / GeoTensor.
        method: Interpolation method (``'linear'``, ``'nearest'``,
                ``'cubic'``).
        limit: Maximum number of consecutive NaNs to fill.

    Returns:
        A :class:`GeoTensor` with interpolated time gaps.
    """
    if 'time' not in obj.dims:
        raise ValueError("Data has no 'time' dimension.")

    result = obj.interpolate_na(
        dim='time',
        method=method,
    )
    return to_geotensor(result, obj)


def temporal_window(
    obj: xr.DataArray,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
) -> GeoTensor:
    """Select a temporal slice by start/end bounds.

    Args:
        obj: Input DataArray / GeoTensor.
        start: Start time (inclusive).
        end: End time (inclusive).

    Returns:
        A time-sliced :class:`GeoTensor`.
    """
    if 'time' not in obj.dims:
        raise ValueError("Data has no 'time' dimension.")

    result = obj.sel(time=slice(start, end))
    return to_geotensor(result, obj)


def temporal_aggregate(
    obj: xr.DataArray,
    period: str = 'M',
    func: str = 'mean',
) -> GeoTensor:
    """Aggregate the time dimension over a period.

    Args:
        obj: Input DataArray / GeoTensor.
        period: Temporal period (``'M'``, ``'Y'``, ``'season'``, etc.).
        func: Aggregation function (``'mean'``, ``'sum'``, ``'min'``,
              ``'max'``, ``'median'``, ``'std'``).

    Returns:
        An aggregated :class:`GeoTensor`.
    """
    if 'time' not in obj.dims:
        raise ValueError("Data has no 'time' dimension.")

    if obj.chunks is not None:
        obj = obj.load()

    resampled = obj.resample(time=period)
    result = getattr(resampled, func)(dim='time')
    return to_geotensor(result, obj)


def select_by_event(
    obj: xr.DataArray,
    event_times: Sequence[Union[str, pd.Timestamp]],
    tolerance: str = '1D',
) -> GeoTensor:
    """Select time slices nearest to a list of event times.

    Args:
        obj: Input DataArray / GeoTensor.
        event_times: List of timestamps for events of interest.
        tolerance: Maximum time offset to consider a match.

    Returns:
        A :class:`GeoTensor` indexed by the closest event times.
    """
    if 'time' not in obj.dims:
        raise ValueError("Data has no 'time' dimension.")

    event_times = pd.DatetimeIndex(event_times)
    selected = obj.sel(time=event_times, method='nearest', tolerance=tolerance)

    if selected.isnull().any():
        import warnings
        warnings.warn(
            "Some event times had no match within tolerance. "
            "Dropping NaN slices."
        )
        selected = selected.dropna(dim='time', how='all')

    return to_geotensor(selected, obj)


def time_to_relative(obj: xr.DataArray) -> GeoTensor:
    """Replace absolute time coordinates with relative time deltas.

    The first time step becomes ``0``, subsequent steps are
    expressed as :class:`pd.Timedelta` offsets from the first step.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        A :class:`GeoTensor` with relative time coordinates.
    """
    if 'time' not in obj.dims:
        raise ValueError("Data has no 'time' dimension.")

    time_vals = pd.DatetimeIndex(obj.coords['time'].values)
    if len(time_vals) == 0:
        return to_geotensor(obj, obj)

    t0 = time_vals[0]
    relative = time_vals - t0

    result = obj.assign_coords(time=relative)
    result = result.rename({'time': 'time_relative'})
    return to_geotensor(result, obj)


def time_interval(obj: xr.DataArray) -> Optional[pd.Timedelta]:
    """Nominal time spacing (may be irregular — returns median gap).

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        Median :class:`pd.Timedelta` between consecutive time steps,
        or ``None`` if no time dimension.
    """
    if 'time' not in obj.dims or len(obj.coords['time']) < 2:
        return None
    diffs = pd.Series(obj.coords['time'].values).diff().dropna()
    return pd.Timedelta(diffs.median())


def time_metadata(obj: xr.DataArray) -> Dict[str, Any]:
    """Return dict with start, end, and acquisition mode.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        Dict with keys ``start``, ``end``, ``acquisition_mode``.
    """
    if 'time' not in obj.dims:
        return {}
    time_vals = pd.DatetimeIndex(obj.coords['time'].values)
    if len(time_vals) == 0:
        return {}
    return {
        'start': time_vals[0],
        'end': time_vals[-1],
        'acquisition_mode': obj.attrs.get('acquisition_mode', 'instantaneous'),
    }
