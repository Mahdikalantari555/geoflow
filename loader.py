"""Data loading utilities for geospatial remote sensing datasets."""

from __future__ import annotations

import copy
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DatasetConfig:
    """Configuration for GeoDataset."""

    seed: int = 42
    shuffle: bool = True
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last: bool = False
    collate_fn: Optional[Callable] = None
    # Geospatial-specific config
    spatial_partition_size: float = 1.0  # km
    min_sample_coverage: float = 0.8
    include_nodata: bool = False
    # Augmentation config
    augmentation_prob: float = 0.5
    max_augmentations_per_sample: int = 3
    # Validation config
    validate_on_load: bool = True
    integrity_check_level: str = "basic"  # basic/comprehensive


class GeoDataset(Dataset):
    """Formal geospatial dataset abstraction for remote sensing.

    Provides a consistent interface for loading, accessing, and managing
    geospatial data with spatial/temporal metadata handling, integrity
    validation, and train/val/test splitting.

    Args:
        data_path: Path to dataset directory or file.
        schema: Optional GeoSchema for validation.
        transform: Transform to apply to inputs.
        target_transform: Transform to apply to targets.
        split: Dataset split (train/val/test/all).
        config: Dataset configuration.
    """

    def __init__(
        self,
        data_path: str,
        schema: Optional[Any] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split: str = "all",
        config: Optional[DatasetConfig] = None,
    ) -> None:
        self.data_path = data_path
        self.schema = schema
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.config = config or DatasetConfig()

        self.samples: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from disk. Subclasses may override."""
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")

        if data_path.is_dir():
            self._load_from_directory(data_path)
        elif data_path.suffix in (".json", ".geojson"):
            self._load_from_geojson(data_path)
        else:
            self._load_from_file(data_path)

    def _load_from_directory(self, path: Path) -> None:
        """Load samples from a directory structure (class folders)."""
        class_dirs = sorted([d for d in path.iterdir() if d.is_dir()])
        for class_idx, class_dir in enumerate(class_dirs):
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"):
                    self.samples.append({
                        "input": str(img_file),
                        "target": class_idx,
                        "class_name": class_dir.name,
                    })

    def _load_from_geojson(self, path: Path) -> None:
        """Load samples from a GeoJSON annotation file."""
        import json
        with open(path) as f:
            data = json.load(f)
        for feature in data.get("features", []):
            self.samples.append({
                "input": feature.get("properties", {}).get("image_path", ""),
                "target": feature.get("properties", {}).get("label", None),
                "geometry": feature.get("geometry"),
                "metadata": feature.get("properties", {}),
            })

    def _load_from_file(self, path: Path) -> None:
        """Load a single file as the only sample."""
        self.samples.append({
            "input": str(path),
            "target": None,
        })

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample at index with transforms applied."""
        sample = dict(self.samples[idx])
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform and sample.get("target") is not None:
            sample = self.target_transform(sample)
        return sample

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample."""
        return self.metadata.get(str(idx), self.samples[idx].get("metadata", {}))

    def spatial_extent(self) -> Dict[str, float]:
        """Get approximate spatial extent of the dataset."""
        return {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}

    def temporal_coverage(self) -> Dict[str, Optional[str]]:
        """Get temporal coverage."""
        return {"start": None, "end": None}

    def class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        distribution: Dict[str, int] = {}
        for sample in self.samples:
            target = sample.get("target")
            if target is not None:
                key = sample.get("class_name", str(target))
                distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        spatial_split: bool = True,
    ) -> Dict[str, "GeoDataset"]:
        """Split dataset into train/val/test."""
        n = len(self.samples)
        indices = np.random.RandomState(self.config.seed).permutation(n)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {}
        for split_name, split_indices in [
            ("train", indices[:train_end]),
            ("val", indices[train_end:val_end]),
            ("test", indices[val_end:]),
        ]:
            split_ds = GeoDataset(
                self.data_path,
                self.schema,
                self.transform,
                self.target_transform,
                split=split_name,
                config=self.config,
            )
            split_ds.samples = [self.samples[i] for i in split_indices]
            split_ds.metadata = self.metadata
            splits[split_name] = split_ds
        return splits

    def augment_dataset(
        self,
        augmentor: Any,
        augmentation_factor: float = 1.0,
        save_to_disk: bool = False,
        output_path: Optional[str] = None,
    ) -> "GeoDataset":
        """Augment dataset and return a new dataset with augmented samples."""
        augmented_ds = GeoDataset(
            self.data_path,
            self.schema,
            self.transform,
            self.target_transform,
            split=self.split,
            config=self.config,
        )
        augmented_samples = list(self.samples)
        num_augmented = int(len(self.samples) * (augmentation_factor - 1))

        for i in range(num_augmented):
            sample = copy.deepcopy(self.samples[i % len(self.samples)])
            if hasattr(augmentor, "__call__"):
                sample = augmentor(sample)
            augmented_samples.append(sample)

        augmented_ds.samples = augmented_samples
        return augmented_ds

    def to(self, device: torch.device) -> None:
        """Move dataset tensors to device (no-op for file-based datasets)."""
        pass


class GeoDataLoader(DataLoader):
    """PyTorch DataLoader with geospatial awareness.

    Extends torch.utils.data.DataLoader with geospatial-specific defaults
    such as pin_memory, geospatial collation, and worker settings.

    Args:
        dataset: GeoDataset instance.
        batch_size: Number of samples per batch.
        shuffle: Shuffle data at every epoch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Pin memory for faster GPU transfer.
        persistent_workers: Keep workers alive between epochs.
        collate_fn: Custom collate function.
        prefetch_factor: Number of batches to prefetch per worker.
        **kwargs: Additional arguments to torch DataLoader.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        collate_fn: Optional[Callable] = None,
        prefetch_factor: int = 2,
        **kwargs: Any,
    ) -> None:
        self.geodataset = dataset

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn or self.default_geo_collate,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            **kwargs,
        )

    @staticmethod
    def default_geo_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default collate function that stacks inputs and batches targets."""
        inputs = []
        targets = []
        metadata = []

        for sample in batch:
            if isinstance(sample.get("input"), torch.Tensor):
                inputs.append(sample["input"])
            elif isinstance(sample.get("input"), str):
                inputs.append(sample["input"])
            if "target" in sample and sample["target"] is not None:
                targets.append(sample["target"])
            metadata.append(sample.get("metadata", {}))

        result: Dict[str, Any] = {"metadata": metadata}
        if inputs and isinstance(inputs[0], torch.Tensor):
            result["inputs"] = torch.stack(inputs, dim=0)
        else:
            result["inputs"] = inputs
        if targets:
            if isinstance(targets[0], torch.Tensor):
                result["targets"] = torch.stack(targets, dim=0)
            else:
                result["targets"] = torch.tensor(targets)
        return result

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for distributed training."""
        if hasattr(self, "sampler") and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class FaultTolerantLoader:
    """Retry and error handling for cloud/remote data loading.

    Args:
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries in seconds.
        backoff_factor: Exponential backoff multiplier.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        timeout: float = 30.0,
    ) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def load_with_retry(
        self,
        url: str,
        loader_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Any:
        """Load data with automatic retry on failure.

        Args:
            url: Data URL or path.
            loader_fn: Callable that performs the actual loading.
            **kwargs: Additional arguments passed to loader_fn.

        Returns:
            Loaded data.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_error: Optional[Exception] = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                if loader_fn is not None:
                    return loader_fn(url, **kwargs)
                # Default: try to load with torch/rasterio
                if url.endswith((".pt", ".pth")):
                    return torch.load(url, **kwargs)
                if url.endswith((".tif", ".tiff")):
                    import rasterio
                    with rasterio.open(url) as src:
                        return src.read()
                if url.endswith((".npy", ".npz")):
                    return np.load(url, **kwargs)
                # Fallback: try with generic open
                import urllib.request
                with urllib.request.urlopen(url, timeout=self.timeout) as response:
                    return response.read()

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= self.backoff_factor

        raise RuntimeError(
            f"Failed to load {url} after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )