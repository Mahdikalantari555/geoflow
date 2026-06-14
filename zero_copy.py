"""Zero-copy memory management utilities for GPU-accelerated geo-training."""

from __future__ import annotations

from typing import Tuple

import torch


class ZeroCopyUtils:
    """Static utility methods for zero-copy tensor operations."""

    @staticmethod
    def pin_memory(tensor: torch.Tensor) -> torch.Tensor:
        """Register tensor in page-locked (pinned) host memory.

        Args:
            tensor: Input tensor (CPU).

        Returns:
            The same tensor with pinned memory. Returns unchanged if already pinned or on GPU.
        """
        if tensor.device.type != "cpu" or tensor.is_pinned():
            return tensor
        return tensor.pin_memory()

    @staticmethod
    def async_gpu_transfer(
        tensor: torch.Tensor,
        device: torch.device,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Asynchronous GPU transfer to overlap I/O and compute.

        Args:
            tensor: Input tensor (should be pinned memory).
            device: Target device.
            non_blocking: Async transfer if True.

        Returns:
            Tensor on target device.
        """
        if not tensor.is_pinned() and tensor.device.type == "cpu":
            tensor = tensor.pin_memory()
        return tensor.to(device, non_blocking=non_blocking)

    @staticmethod
    def create_shared_memory(tensor: torch.Tensor) -> torch.Tensor:
        """Create a memory-shared tensor for inter-process communication.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor with shared memory.
        """
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        return tensor.share_memory_()

    @staticmethod
    def get_tensor_storage(tensor: torch.Tensor) -> Tuple[int, int, int]:
        """Get low-level storage information for zero-copy operations.

        Returns:
            Tuple of (data_ptr, storage_offset, numel).
        """
        storage = tensor.untyped_storage()
        return (storage.data_ptr(), storage.storage_offset() if hasattr(storage, 'storage_offset') else 0, tensor.numel())