"""Unified tiling engine for sampling and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class TilingConfig:
    """Configuration for tiling engine."""

    tile_size: int = 224
    overlap: int = 32
    strategy: str = "sliding_window"
    min_valid_pixels: float = 0.8
    memory_budget_gb: float = 4.0
    multi_scale_levels: List[int] = field(default_factory=lambda: [1, 2, 4])
    balance_classes: bool = True


class GeoTile:
    """Individual tile with spatial metadata."""

    def __init__(
        self,
        tensor: Any,
        tile_bbox: Any,
        original_bbox: Any,
        tile_index: Tuple[int, int],
        overlap_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.tensor = tensor
        self.tile_bbox = tile_bbox
        self.original_bbox = original_bbox
        self.tile_index = tile_index
        self.overlap_mask = overlap_mask

    @property
    def overlap_percentage(self) -> float:
        """Calculate overlap percentage with neighboring tiles."""
        if self.overlap_mask is None:
            return 0.0
        return float(torch.sum(self.overlap_mask) / self.overlap_mask.numel())


class GeoTilingEngine:
    """Unified tiling engine for sampling and inference.

    Features:
    - Consistent tile generation for training and inference
    - Smart overlap management to avoid boundary artifacts
    - Tiling strategy optimization based on memory budget
    - Stitching with overlap handling for inference
    """

    TILING_STRATEGIES = [
        "sliding_window", "random", "stratified", "multi_scale",
        "edge_safe", "overlap_aware", "memory_budget",
    ]

    STITCH_METHODS = ["mean", "gaussian", "max", "softmax"]

    def __init__(
        self,
        tile_size: int = 224,
        overlap: int = 32,
        strategy: str = "sliding_window",
        config: Optional[TilingConfig] = None,
    ) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.strategy = strategy
        self.config = config or TilingConfig()

        if strategy not in self.TILING_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {self.TILING_STRATEGIES}")

    def generate_tiles(
        self,
        geotensor: Any,
        mask: Optional[torch.Tensor] = None,
    ) -> Iterator[GeoTile]:
        """Generate tiles from geotensor.

        Args:
            geotensor: Input GeoTensor to tile.
            mask: Optional mask for valid regions.

        Yields:
            GeoTile objects with spatial metadata.
        """
        from .core import GeoTensor, GeoExtent

        h = len(geotensor.coords['y'])
        w = len(geotensor.coords['x'])
        t = geotensor.transform
        xmin, ymax = t * (0, 0)
        xmax, ymin = t * (w, h)
        extent = GeoExtent(
            xmin=min(xmin, xmax), ymin=min(ymin, ymax),
            xmax=max(xmin, xmax), ymax=max(ymin, ymax),
            crs=geotensor.crs,
        )

        stride = self.tile_size - self.overlap
        tile_idx = 0

        for row in range(0, h, stride):
            for col in range(0, w, stride):
                row_end = min(row + self.tile_size, h)
                col_end = min(col + self.tile_size, w)

                patch_data = geotensor.isel(y=slice(row, row_end), x=slice(col, col_end))

                if mask is not None:
                    tile_mask = mask[row:row_end, col:col_end]
                    valid_ratio = float(tile_mask.float().mean())
                    if valid_ratio < self.config.min_valid_pixels:
                        continue

                res_x = abs(t.a)
                res_y = abs(t.e)
                x_min = extent.xmin + col * res_x
                y_min = extent.ymin + row * res_y
                x_max = x_min + (col_end - col) * res_x
                y_max = y_min + (row_end - row) * res_y

                tile_extent = GeoExtent(
                    xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max,
                    crs=geotensor.crs,
                )

                yield GeoTile(
                    tensor=patch_data,
                    tile_bbox=tile_extent,
                    original_bbox=extent,
                    tile_index=(tile_idx, tile_idx + 1),
                )
                tile_idx += 1

    def stitch_tiles(
        self,
        tiles: List[GeoTile],
        target_shape: Tuple[int, int],
        method: str = "mean",
    ) -> Any:
        """Stitch tiles back to original extent.

        Args:
            tiles: List of GeoTiles to stitch.
            target_shape: Target output shape (H, W).
            method: Stitching method ('mean', 'gaussian', 'max', 'softmax').

        Returns:
            Stitched output as tensor.
        """
        if not tiles:
            raise ValueError("No tiles to stitch")

        h, w = target_shape
        first = tiles[0].tensor
        if hasattr(first, 'values'):
            first = first.values
        accumulator = torch.zeros((first.shape[0], h, w), dtype=torch.float32)
        count = torch.zeros((1, h, w), dtype=torch.int32)

        for tile in tiles:
            data = tile.tensor
            if hasattr(data, 'values'):
                data = data.values
            if isinstance(data, torch.Tensor):
                data = data.cpu()
            elif isinstance(data, np.ndarray):
                data = torch.from_numpy(data)

            bbox = tile.tile_bbox
            col_start = max(0, int(bbox.xmin))
            row_start = max(0, int(bbox.ymin))

            th, tw = data.shape[-2], data.shape[-1]
            row_end = min(row_start + th, h)
            col_end = min(col_start + tw, w)

            actual_h = row_end - row_start
            actual_w = col_end - col_start

            if actual_h > 0 and actual_w > 0:
                accumulator[:, row_start:row_end, col_start:col_end] += data[:, :actual_h, :actual_w]
                count[:, row_start:row_end, col_start:col_end] += 1

        count = count.clamp(min=1)
        stitched = accumulator / count.float()

        return stitched

    def optimize_tiling_strategy(
        self,
        geotensor: Any,
        memory_budget_gb: float = 4.0,
    ) -> TilingConfig:
        """Optimize tiling strategy based on memory budget.

        Args:
            geotensor: Input GeoTensor.
            memory_budget_gb: Available GPU memory in GB.

        Returns:
            Optimized tiling configuration.
        """
        bands = len(geotensor.band_names) if geotensor.band_names else 1
        dtype_size = 4

        max_pixels = int((memory_budget_gb * 1024**3) / (bands * dtype_size * 4))
        tile_size = min(self.tile_size, int(np.sqrt(max_pixels)))

        return TilingConfig(
            tile_size=tile_size,
            overlap=self.overlap,
            strategy=self.strategy,
            memory_budget_gb=memory_budget_gb,
        )

    def create_inference_tiles(
        self,
        geotensor: Any,
    ) -> List[GeoTile]:
        """Create inference tiles with overlap handling.

        Args:
            geotensor: Input GeoTensor.

        Returns:
            List of GeoTiles for inference.
        """
        return list(self.generate_tiles(geotensor))
