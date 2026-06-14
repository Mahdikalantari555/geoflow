"""Geographic patch sampling for GeoTensor (§11 of GeoTensor Spec).

Provides tile / random / grid / centroid sampling strategies and the
:class:`GeoTensorPatchDataset` adapter for ML training loops.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from .core import GeoTensor, to_geotensor


# ====================================================================
# Sampling Methods
# ====================================================================


def sample_tiles(
    obj: xr.DataArray,
    size: Union[int, str] = 256,
    stride: Union[int, float] = 128,
    extent: Union[str, Tuple[float, float, float, float]] = 'full',
    mask_threshold: float = 0.2,
    random_shift: bool = False,
    format: str = 'dict',
    rng: Optional[np.random.Generator] = None,
) -> List[Any]:
    """Extract tiles / patches via a sliding window.

    Args:
        obj: Input DataArray / GeoTensor.
        size: Patch size in pixels. Can be ``int`` or a string like
              ``'256m'`` to specify size in metres.
        stride: Step between patches in pixels. Can be ``int`` or
                ``float`` (fraction of size).
        extent: ``'full'`` to use the full data extent, or a
                ``(xmin, ymin, xmax, ymax)`` bounding box.
        mask_threshold: Maximum fraction of masked pixels allowed
                        (0.0-1.0). Patches exceeding this are skipped.
        random_shift: If ``True``, apply random jitter to each patch
                      position.
        format: Output format — ``'dict'`` returns dicts with coords,
                ``'list'`` returns raw arrays.
        rng: Optional random number generator for jitter.

    Returns:
        List of patch dicts or arrays.
    """
    if isinstance(size, str) and size.endswith('m'):
        # Size in metres — approximate to pixels
        try:
            res = abs(obj.geotensor.resolution[0])
            size = int(round(float(size[:-1]) / res))
        except Exception:
            size = 256

    if isinstance(stride, float) and stride < 1.0:
        stride = int(size * stride)

    if rng is None:
        rng = np.random.default_rng(42)

    # Determine spatial extent
    if isinstance(extent, (tuple, list)):
        x_coords = obj.coords['x'].values
        y_coords = obj.coords['y'].values
        x_mask = (x_coords >= extent[0]) & (x_coords <= extent[2])
        y_mask = (y_coords >= extent[1]) & (y_coords <= extent[3])
        sub = obj.isel(x=x_mask, y=y_mask)
    else:
        sub = obj

    h, w = len(sub.coords['y']), len(sub.coords['x'])
    if h < size or w < size:
        return []

    patches = []
    for row in range(0, h - size + 1, stride):
        for col in range(0, w - size + 1, stride):
            row_off = row
            col_off = col

            if random_shift:
                max_shift = min(stride // 2, size // 4)
                if max_shift > 0:
                    row_off = max(0, min(h - size, row + rng.integers(-max_shift, max_shift)))
                    col_off = max(0, min(w - size, col + rng.integers(-max_shift, max_shift)))

            patch_data = sub.isel(y=slice(row_off, row_off + size),
                                  x=slice(col_off, col_off + size))

            # Apply mask threshold if applicable
            if mask_threshold > 0.0:
                mask = patch_data.attrs.get('mask')
                if mask is not None and hasattr(mask, 'combined'):
                    combined = mask.combined(mode='any')
                    if combined is not None and combined.any():
                        frac_masked = float(combined.mean().values)
                        if frac_masked > mask_threshold:
                            continue

            if format == 'dict':
                # Spatial coords of patch center
                cx = float(patch_data.coords['x'].values[size // 2])
                cy = float(patch_data.coords['y'].values[size // 2])
                patches.append({
                    'data': patch_data,
                    'center': (cx, cy),
                    'bounds': (
                        float(patch_data.coords['x'].values[0]),
                        float(patch_data.coords['y'].values[0]),
                        float(patch_data.coords['x'].values[-1]),
                        float(patch_data.coords['y'].values[-1]),
                    ),
                    'row': row_off,
                    'col': col_off,
                })
            else:
                patches.append(patch_data.values)

    return patches


def sample_random(
    obj: xr.DataArray,
    num_patches: int = 1000,
    size: int = 256,
    region: Optional[Any] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Any]:
    """Extract random spatial patches.

    Args:
        obj: Input DataArray / GeoTensor.
        num_patches: Number of patches to extract.
        size: Patch size in pixels.
        region: Optional geometry or mask layer to constrain sampling.
        rng: Optional random number generator.

    Returns:
        List of patch dicts.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    h, w = len(obj.coords['y']), len(obj.coords['x'])
    if h < size or w < size:
        return []

    patches = []
    attempts = 0
    max_attempts = num_patches * 5

    while len(patches) < num_patches and attempts < max_attempts:
        attempts += 1
        row = rng.integers(0, h - size)
        col = rng.integers(0, w - size)

        # Region constraint
        if region is not None:
            if callable(region):
                if not region(obj, row, col, size):
                    continue
            elif isinstance(region, xr.DataArray):
                # Use mask layer to check
                patch_region = region.isel(y=slice(row, row + size),
                                           x=slice(col, col + size))
                if patch_region.mean().values < 0.5:
                    continue

        patch_data = obj.isel(y=slice(row, row + size),
                              x=slice(col, col + size))
        patches.append({
            'data': patch_data,
            'center': (
                float(patch_data.coords['x'].values[size // 2]),
                float(patch_data.coords['y'].values[size // 2]),
            ),
            'row': row,
            'col': col,
        })

    return patches


def sample_grid(
    obj: xr.DataArray,
    nx: int = 10,
    ny: int = 10,
    size: int = 256,
) -> List[Any]:
    """Extract patches on a regular grid.

    Args:
        obj: Input DataArray / GeoTensor.
        nx: Number of columns in the sampling grid.
        ny: Number of rows in the sampling grid.
        size: Patch size in pixels.

    Returns:
        List of patch dicts.
    """
    h, w = len(obj.coords['y']), len(obj.coords['x'])
    if h < size or w < size:
        return []

    step_y = (h - size) // max(1, ny)
    step_x = (w - size) // max(1, nx)

    patches = []
    for i in range(ny):
        row = i * step_y
        if row + size > h:
            continue
        for j in range(nx):
            col = j * step_x
            if col + size > w:
                continue
            patch_data = obj.isel(y=slice(row, row + size),
                                  x=slice(col, col + size))
            patches.append({
                'data': patch_data,
                'center': (
                    float(patch_data.coords['x'].values[size // 2]),
                    float(patch_data.coords['y'].values[size // 2]),
                ),
                'row': row,
                'col': col,
            })

    return patches


def sample_centroids(
    obj: xr.DataArray,
    points: Any,
    size: int = 256,
) -> List[Any]:
    """Extract patches around point locations (centroids).

    Args:
        obj: Input DataArray / GeoTensor.
        points: GeoDataFrame or list of ``(x, y)`` coordinate tuples.
        size: Patch size in pixels.

    Returns:
        List of patch dicts.
    """
    # Convert points to list of (x, y) coords
    if hasattr(points, 'geometry'):
        # geopandas GeoDataFrame
        coords = [(p.x, p.y) for p in points.geometry]
    elif hasattr(points, '__iter__'):
        coords = [(float(p[0]), float(p[1])) for p in points]
    else:
        coords = [(0, 0)]

    x_coords = obj.coords['x'].values
    y_coords = obj.coords['y'].values

    patches = []
    for cx, cy in coords:
        # Find nearest pixel indices
        col = int(np.argmin(np.abs(x_coords - cx)))
        row = int(np.argmin(np.abs(y_coords - cy)))

        half = size // 2
        row_start = max(0, row - half)
        col_start = max(0, col - half)
        row_end = min(len(y_coords), row_start + size)
        col_end = min(len(x_coords), col_start + size)

        # Adjust start if needed
        if row_end - row_start < size:
            row_start = max(0, row_end - size)
        if col_end - col_start < size:
            col_start = max(0, col_end - size)

        if row_end <= row_start or col_end <= col_start:
            continue

        patch_data = obj.isel(y=slice(row_start, row_end),
                              x=slice(col_start, col_end))

        # Pad if needed
        if patch_data.sizes['y'] < size or patch_data.sizes['x'] < size:
            pad_y = size - patch_data.sizes['y']
            pad_x = size - patch_data.sizes['x']
            pad_dict = {k: (0, 0) for k in patch_data.dims}
            pad_dict['y'] = (0, pad_y)
            pad_dict['x'] = (0, pad_x)
            patch_data = patch_data.pad(**pad_dict, mode='constant')

        patches.append({
            'data': patch_data,
            'center': (cx, cy),
            'pixel': (col, row),
        })

    return patches


# ====================================================================
# GeoTensorPatchDataset — ML Dataset Adapter
# ====================================================================


class GeoTensorPatchDataset:
    """Dataset of geographic patches for ML training loops.

    Implements ``torch.utils.data.Dataset`` interface for lazy loading
    and on-the-fly augmentation.

    Args:
        geotensor: Source :class:`GeoTensor` or :class:`xr.DataArray`.
        patch_size: Spatial size of each patch in pixels.
        stride: Step between patches.
        mask_threshold: Max fraction of masked pixels allowed.
        augment: If ``True``, apply random augmentations.
        transform: Optional callable for custom transforms.
        shuffle: If ``True``, shuffle patch indices.
        rng_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        geotensor: GeoTensor,
        patch_size: int = 256,
        stride: int = 128,
        mask_threshold: float = 0.2,
        augment: bool = False,
        transform: Optional[Callable] = None,
        shuffle: bool = False,
        rng_seed: int = 42,
    ) -> None:
        self.geotensor = geotensor
        self.patch_size = patch_size
        self.stride = stride
        self.mask_threshold = mask_threshold
        self.augment = augment
        self.transform = transform
        self.rng = np.random.default_rng(rng_seed)

        # Generate patch indices
        h = len(geotensor.coords['y'])
        w = len(geotensor.coords['x'])
        self._indices: List[Tuple[int, int]] = []
        for row in range(0, h - patch_size + 1, stride):
            for col in range(0, w - patch_size + 1, stride):
                self._indices.append((row, col))

        if shuffle:
            self._order = list(range(len(self._indices)))
            self.rng.shuffle(self._order)
        else:
            self._order = list(range(len(self._indices)))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Any:
        actual_idx = self._order[idx]
        row, col = self._indices[actual_idx]

        patch = self.geotensor.isel(
            y=slice(row, row + self.patch_size),
            x=slice(col, col + self.patch_size),
        )

        # Apply mask threshold
        if self.mask_threshold > 0.0:
            mask_attr = patch.attrs.get('mask')
            if mask_attr is not None and hasattr(mask_attr, 'combined'):
                combined = mask_attr.combined(mode='any')
                if combined is not None and combined.mean().values > self.mask_threshold:
                    # Return zeros as fallback
                    patch = patch * 0

        # Augmentation
        if self.augment:
            patch = self._augment(patch)

        # Custom transform
        if self.transform is not None:
            patch = self.transform(patch)

        return patch

    def _augment(self, patch: xr.DataArray) -> xr.DataArray:
        """Apply random augmentations."""
        if self.rng.random() > 0.5:
            patch = patch.isel(x=slice(None, None, -1))  # Horizontal flip

        if self.rng.random() > 0.5:
            patch = patch.isel(y=slice(None, None, -1))  # Vertical flip

        # Random rotation (90-degree increments via transpose)
        k = self.rng.integers(0, 4)
        if k > 0:
            for _ in range(k):
                dims = list(patch.dims)
                if 'y' in dims and 'x' in dims:
                    yi = dims.index('y')
                    xi = dims.index('x')
                    dims[yi], dims[xi] = dims[xi], dims[yi]
                    patch = patch.transpose(*dims)

        return patch

    def to_torch_dataset(self) -> Any:
        """Return self as a :class:`torch.utils.data.Dataset`.

        Returns:
            ``self`` (compatible with PyTorch DataLoader).
        """
        return self


# ====================================================================
# Accessor helper
# ====================================================================


def to_patch_dataset(
    obj: xr.DataArray,
    patch_size: int = 256,
    stride: int = 128,
    mask_threshold: float = 0.2,
    augment: bool = False,
    transform: Optional[Callable] = None,
    shuffle: bool = False,
) -> GeoTensorPatchDataset:
    """Convert to a :class:`GeoTensorPatchDataset` for ML training.

    Args:
        obj: Input DataArray / GeoTensor.
        patch_size: Patch size in pixels.
        stride: Step between patches.
        mask_threshold: Max masked fraction per patch.
        augment: Enable random augmentations.
        transform: Optional custom transform callable.
        shuffle: Shuffle patch order.

    Returns:
        A :class:`GeoTensorPatchDataset`.
    """
    return GeoTensorPatchDataset(
        geotensor=to_geotensor(obj, obj),
        patch_size=patch_size,
        stride=stride,
        mask_threshold=mask_threshold,
        augment=augment,
        transform=transform,
        shuffle=shuffle,
    )
