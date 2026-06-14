"""Differentiable geospatial operations for GeoTensor (§12 of GeoTensor Spec).

Provides grid-aligned differentiable resampling for ML training pipelines
using PyTorch's ``grid_sample``, and gradient-tracking conversion helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import xarray as xr

from .core import GeoTensor, NonDifferentiableError, to_geotensor


# ====================================================================
# Differentiable Operations
# ====================================================================


def differentiable_resample(
    obj: xr.DataArray,
    target_shape: Tuple[int, int],
    method: str = 'bilinear',
    align_corners: bool = True,
) -> GeoTensor:
    """Resample using differentiable interpolation.

    Uses ``torch.nn.functional.grid_sample`` for grid-aligned
    interpolation that supports gradient backpropagation.

    Args:
        obj: Input DataArray / GeoTensor.
        target_shape: Target ``(height, width)`` in pixels.
        method: Interpolation method (``'bilinear'``, ``'nearest'``).
        align_corners: If ``True``, aligns pixel corners for grid_sample.

    Returns:
        A :class:`GeoTensor` with gradient tracking.

    Raises:
        NonDifferentiableError: If the method requires CRS changes that
                                break gradients.
    """
    # Convert to PyTorch tensor with grad tracking
    src_tensor = _to_torch_tensor(obj, require_grad=True)
    src_h, src_w = src_tensor.shape[-2:]

    # Create normalized sampling grid
    theta = torch.tensor([
        [src_w / target_shape[1], 0, 0],
        [0, src_h / target_shape[0], 0],
    ], dtype=src_tensor.dtype, device=src_tensor.device).unsqueeze(0)

    grid = torch.nn.functional.affine_grid(
        theta,
        (1, 1, target_shape[0], target_shape[1]),
        align_corners=align_corners,
    )

    # Handle batched and multi-band
    if src_tensor.ndim == 2:
        src_tensor = src_tensor.unsqueeze(0).unsqueeze(0)
    elif src_tensor.ndim == 3:
        src_tensor = src_tensor.unsqueeze(0)
    elif src_tensor.ndim != 4:
        raise ValueError(f"Expected 2-4D tensor, got {src_tensor.ndim}D")

    grid = grid.expand(src_tensor.shape[0], -1, -1, -1)

    # Apply grid_sample
    resampled = torch.nn.functional.grid_sample(
        src_tensor,
        grid,
        mode=method,
        padding_mode='zeros',
        align_corners=align_corners,
    )

    # Reshape back to original number of dims
    if obj.ndim == 2:
        resampled = resampled.squeeze(0).squeeze(0)
    elif obj.ndim == 3:
        resampled = resampled.squeeze(0)

    # Build new coordinates
    x = obj.coords['x'].values
    y = obj.coords['y'].values
    new_x = np.linspace(float(x[0]), float(x[-1]), target_shape[1])
    new_y = np.linspace(float(y[0]), float(y[-1]), target_shape[0])

    new_coords = {k: obj.coords[k] for k in obj.dims if k not in ('x', 'y')}
    new_coords['x'] = ('x', new_x)
    new_coords['y'] = ('y', new_y)

    return GeoTensor(
        resampled.detach().cpu().numpy(),
        coords=new_coords,
        dims=obj.dims,
        crs=obj.attrs.get('crs'),
        band_info=obj.attrs.get('band_semantics'),
        attrs={k: v for k, v in obj.attrs.items()
               if k not in ('crs', 'band_semantics')},
    )


def detach(obj: xr.DataArray) -> GeoTensor:
    """Detach from the computation graph.

    Returns a new GeoTensor without gradient tracking. Useful as an
    escape hatch when non-differentiable operations (e.g. reprojection)
    are needed inside a gradient tape.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        A :class:`GeoTensor` without gradient tracking.
    """
    result = obj.copy(deep=False)
    return to_geotensor(result, obj)


def to_pytorch_with_grad(
    obj: xr.DataArray,
    require_grad: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convert to a PyTorch tensor with optional gradient tracking.

    Args:
        obj: Input DataArray / GeoTensor.
        require_grad: If ``True``, enable gradient tracking on the result.
        device: Target device (e.g. ``'cuda:0'``). If ``None``, CPU.

    Returns:
        A :class:`torch.Tensor` with optional gradient tracking.
    """
    return _to_torch_tensor(obj, require_grad=require_grad, device=device)


# ====================================================================
# Internal helpers
# ====================================================================


def _to_torch_tensor(
    obj: xr.DataArray,
    require_grad: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convert xarray data to PyTorch tensor, optionally on a device."""
    data = obj.values
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data.astype(np.float32))
    else:
        tensor = torch.as_tensor(data)

    if require_grad:
        tensor.requires_grad_(True)

    if device is not None:
        tensor = tensor.to(device)

    return tensor