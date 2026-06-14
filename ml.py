"""Machine learning interoperability for GeoTensor (§15 of GeoTensor Spec).

Provides framework conversions (PyTorch, TensorFlow, JAX), DataLoader
integration, batch management, and normalization/denormalization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from .core import GeoTensor, to_geotensor


# ====================================================================
# Framework Conversions
# ====================================================================


def to_numpy(obj: xr.DataArray) -> np.ndarray:
    """Convert to a NumPy ndarray.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        NumPy :class:`ndarray`.
    """
    return obj.values


def to_pytorch(
    obj: xr.DataArray,
    require_grad: bool = False,
    device: Optional[Union[str, Any]] = None,
) -> Any:
    """Convert to a PyTorch tensor.

    Args:
        obj: Input DataArray / GeoTensor.
        require_grad: If ``True``, enable gradient tracking.
        device: Target device (e.g. ``'cuda:0'``). If ``None``, CPU.

    Returns:
        A :class:`torch.Tensor`.
    """
    import torch

    data = obj.values
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data.astype(np.float32))
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)

    if require_grad:
        tensor.requires_grad_(True)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def to_tensorflow(obj: xr.DataArray) -> Any:
    """Convert to a TensorFlow tensor.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        A :class:`tf.Tensor`.
    """
    import tensorflow as tf
    data = obj.values
    return tf.convert_to_tensor(data, dtype=tf.float32)


def to_jax(obj: xr.DataArray) -> Any:
    """Convert to a JAX array.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        A :class:`jax.Array`.
    """
    import jax.numpy as jnp
    data = obj.values
    return jnp.array(data)


def to_xarray(obj: xr.DataArray) -> xr.DataArray:
    """Convert to a plain :class:`xr.DataArray` (zero-copy).

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        An :class:`xr.DataArray` sharing the same data.
    """
    if isinstance(obj, GeoTensor):
        return xr.DataArray(
            obj.data,
            coords=obj.coords,
            dims=obj.dims,
            attrs=obj.attrs,
            name=obj.name,
        )
    return obj


# ====================================================================
# DataLoader Integration
# ====================================================================


def to_dataloader(
    obj: xr.DataArray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    batch_dim: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a PyTorch DataLoader from this GeoTensor.

    Args:
        obj: Input DataArray / GeoTensor.
        batch_size: Number of samples per batch.
        shuffle: If ``True``, shuffle indices each epoch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If ``True``, use pinned memory (faster GPU transfer).
        batch_dim: Dimension to use as batch/split dimension. If ``None``,
                   uses the first dimension.
        **kwargs: Additional arguments for :class:`torch.utils.data.DataLoader`.

    Returns:
        A :class:`torch.utils.data.DataLoader`.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader

    split_dim = batch_dim if batch_dim is not None else obj.dims[0]

    class _GeoTensorDataset(Dataset):
        def __init__(self, data: xr.DataArray):
            self.data = data
            self.split_dim = split_dim
            self.length = len(data.coords[split_dim])

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int) -> torch.Tensor:
            sample = self.data.isel({self.split_dim: idx})
            tensor = torch.from_numpy(
                sample.values.astype(np.float32)
            )
            return tensor

    dataset = _GeoTensorDataset(obj)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def to_tf_dataset(
    obj: xr.DataArray,
    batch_size: int = 32,
    shuffle: bool = True,
    prefetch: int = 2,
    **kwargs: Any,
) -> Any:
    """Create a TensorFlow ``tf.data.Dataset`` from this GeoTensor.

    Args:
        obj: Input DataArray / GeoTensor.
        batch_size: Number of samples per batch.
        shuffle: If ``True``, shuffle the dataset.
        prefetch: Number of batches to prefetch.
        **kwargs: Additional arguments for dataset creation.

    Returns:
        A :class:`tf.data.Dataset`.
    """
    import tensorflow as tf

    data = obj.values.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size)
    if prefetch > 0:
        dataset = dataset.prefetch(prefetch)

    return dataset


# ====================================================================
# Normalization
# ====================================================================


def compute_normalization(
    obj: xr.DataArray,
    method: str = 'zscore',
    dim: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Compute normalization parameters from the data.

    Args:
        obj: Input DataArray / GeoTensor.
        method: Normalization method (``'zscore'``, ``'minmax'``,
                ``'robust'``).
        dim: Dimensions over which to compute statistics.
             If ``None``, uses all spatial dims ``['x', 'y']``.

    Returns:
        Dict of normalization parameters (e.g. ``{'mean': ..., 'std': ...}``).
    """
    if dim is None:
        dim = ['x', 'y']

    if method == 'zscore':
        mean = obj.mean(dim=dim).values
        std = obj.std(dim=dim).values
        std = np.where(std < 1e-8, 1.0, std)
        return {'method': 'zscore', 'mean': mean, 'std': std}

    elif method == 'minmax':
        min_val = obj.min(dim=dim).values
        max_val = obj.max(dim=dim).values
        range_val = max_val - min_val
        range_val = np.where(range_val < 1e-8, 1.0, range_val)
        return {'method': 'minmax', 'min': min_val, 'max': max_val, 'range': range_val}

    elif method == 'robust':
        q25 = obj.quantile(0.25, dim=dim).values
        q75 = obj.quantile(0.75, dim=dim).values
        median = obj.median(dim=dim).values
        iqr = q75 - q25
        iqr = np.where(iqr < 1e-8, 1.0, iqr)
        return {'method': 'robust', 'median': median, 'iqr': iqr}

    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def normalize(
    obj: xr.DataArray,
    norm_params: Dict[str, Any],
) -> GeoTensor:
    """Apply normalization to the data.

    Args:
        obj: Input DataArray / GeoTensor.
        norm_params: Normalization parameters from
                     :meth:`compute_normalization`.

    Returns:
        A normalized :class:`GeoTensor`.
    """
    method = norm_params.get('method', 'zscore')

    if method == 'zscore':
        mean = norm_params['mean']
        std = norm_params['std']
        if isinstance(mean, np.ndarray) and mean.ndim == 1 and obj.ndim > 1:
            mean = mean.reshape((-1,) + (1,) * (obj.ndim - 1))
            std = np.where(std < 1e-8, 1.0, std)
            std = std.reshape((-1,) + (1,) * (obj.ndim - 1))
        result = (obj - mean) / std
    elif method == 'minmax':
        min_val = norm_params['min']
        range_val = norm_params['range']
        if isinstance(min_val, np.ndarray) and min_val.ndim == 1 and obj.ndim > 1:
            min_val = min_val.reshape((-1,) + (1,) * (obj.ndim - 1))
            range_val = range_val.reshape((-1,) + (1,) * (obj.ndim - 1))
        result = (obj - min_val) / range_val
    elif method == 'robust':
        median = norm_params['median']
        iqr = norm_params['iqr']
        if isinstance(median, np.ndarray) and median.ndim == 1 and obj.ndim > 1:
            median = median.reshape((-1,) + (1,) * (obj.ndim - 1))
            iqr = iqr.reshape((-1,) + (1,) * (obj.ndim - 1))
        result = (obj - median) / iqr
    else:
        raise ValueError(f"Unknown method: '{method}'")

    # Store normalization params in attrs for denormalization
    result.attrs['_norm_params'] = norm_params

    return to_geotensor(result, obj)


def denormalize(obj: xr.DataArray) -> GeoTensor:
    """Reverse a previous normalization.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        A denormalized :class:`GeoTensor`.
    """
    norm_params = obj.attrs.get('_norm_params')
    if norm_params is None:
        raise ValueError(
            "No normalization parameters found in attrs. "
            "This may not be a normalized tensor."
        )

    method = norm_params.get('method', 'zscore')

    if method == 'zscore':
        result = obj * norm_params['std'] + norm_params['mean']
    elif method == 'minmax':
        result = obj * norm_params['range'] + norm_params['min']
    elif method == 'robust':
        result = obj * norm_params['iqr'] + norm_params['median']
    else:
        raise ValueError(f"Unknown method: '{method}'")

    # Remove norm params after denormalization
    result.attrs.pop('_norm_params', None)

    return to_geotensor(result, obj)