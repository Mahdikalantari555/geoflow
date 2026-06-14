"""Visualisation utilities for geospatial remote sensing data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

try:
    import folium
except ImportError:
    folium = None


class GeoVisualizer:
    """Visualization tools for multi-band geospatial imagery.

    Provides methods for generating RGB composites, exploring band
    combinations, creating temporal animations, and rendering
    interactive maps.
    """

    def false_color_composite(
        self,
        geotensor: Any,
        bands: Tuple[int, int, int] = (0, 1, 2),
        normalize: bool = True,
    ) -> np.ndarray:
        """Create a false-color RGB composite.

        Args:
            geotensor: Input GeoTensor [C, H, W].
            bands: Indices of bands to map to (R, G, B).
            normalize: Whether to apply percentile stretch.

        Returns:
            uint8 RGB array [H, W, 3].
        """
        data = geotensor.values
        if data.ndim == 4:
            data = data[0]
        elif data.ndim == 2:
            data = np.stack([data] * 3, axis=0)

        r, g, b = data[bands[0]], data[bands[1]], data[bands[2]]
        rgb = np.stack([r, g, b], axis=-1)

        if normalize:
            for c in range(3):
                low, high = np.percentile(rgb[..., c], [2, 98])
                rgb[..., c] = np.clip((rgb[..., c] - low) / (high - low + 1e-6), 0, 1)

        return (rgb * 255).astype(np.uint8)

    def band_combinations(
        self,
        geotensor: Any,
        combinations: Optional[List[Tuple[int, int, int]]] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """Display multiple band combinations.

        Args:
            geotensor: Input GeoTensor [C, H, W].
            combinations: List of (R, G, B) band tuples.
            figsize: Figure size.

        Returns:
            Matplotlib Figure with subplots.
        """
        n_bands = len(geotensor.band_names) if geotensor.band_names else geotensor.shape[0]
        if combinations is None:
            combinations = [(0, 1, 2)]
            if n_bands >= 4:
                combinations.append((0, 1, 3))
            if n_bands >= 5:
                combinations.append((1, 2, 4))

        fig, axes = plt.subplots(1, len(combinations), figsize=figsize)
        if len(combinations) == 1:
            axes = [axes]

        for ax, combo in zip(axes, combinations):
            rgb = self.false_color_composite(geotensor, bands=combo)
            ax.imshow(rgb)
            ax.set_title(f"Bands {combo}")
            ax.axis("off")

        plt.tight_layout()
        return fig

    def temporal_animation(
        self,
        tensors: List[Any],
        bands: Tuple[int, int, int] = (0, 1, 2),
        interval: int = 500,
        output_path: Optional[str] = None,
        fps: int = 10,
    ) -> Any:
        """Create an animated visualization from a temporal stack.

        Args:
            tensors: List of GeoTensors in temporal order.
            bands: RGB band indices.
            interval: Frame interval in ms.
            output_path: Optional path to save animation.
            fps: Frames per second for saved output.

        Returns:
            Matplotlib FuncAnimation object.
        """
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(8, 8))
        rgb = self.false_color_composite(tensors[0], bands=bands)
        im = ax.imshow(rgb)
        ax.axis("off")
        title = ax.set_title("Frame 0")

        def update(frame_idx: int) -> List[Any]:
            rgb = self.false_color_composite(tensors[frame_idx], bands=bands)
            im.set_data(rgb)
            title.set_text(f"Frame {frame_idx}")
            return [im, title]

        anim = FuncAnimation(fig, update, frames=len(tensors), interval=interval, blit=True)

        if output_path:
            anim.save(output_path, writer="pillow", fps=fps)

        return anim

    def interactive_map(
        self,
        geotensor: Any,
        bands: Tuple[int, int, int] = (0, 1, 2),
        colormap: str = "viridis",
    ) -> Any:
        """Create an interactive folium map.

        Args:
            geotensor: Input GeoTensor [C, H, W].
            bands: RGB band indices.
            colormap: Matplotlib colormap name.

        Returns:
            Folium Map object.
        """
        if folium is None:
            raise ImportError("folium is required for interactive_map. Install with: pip install folium")

        rgb = self.false_color_composite(geotensor, bands=bands)

        try:
            import rasterio
            from rasterio.crs import CRS
            from rasterio.warp import reproject, calculate_default_transform

            h, w = rgb.shape[:2]
            src_crs = geotensor.crs
            src_transform = geotensor.transform

            bounds = geotensor.bounds
            dst_crs = CRS.from_epsg(4326)
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, w, h,
                bounds[0], bounds[1], bounds[2], bounds[3],
            )

            dst_array = np.zeros((3, dst_height, dst_width), dtype=np.float32)
            for band in range(3):
                reproject(
                    source=rgb[..., band].astype(np.float32),
                    destination=dst_array[band],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                )

            rgb_dst = np.stack([dst_array[0], dst_array[1], dst_array[2]], axis=-1)
            rgb_dst = (rgb_dst * 255).astype(np.uint8)
        except ImportError:
            rgb_dst = rgb

        bounds = geotensor.bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        m = folium.Map(location=[center_y, center_x], zoom_start=12)

        m.add_child(folium.raster_layers.ImageOverlay(
            image=rgb_dst,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            colormap=lambda x: plt.get_cmap(colormap)(x),
        ))

        return m
