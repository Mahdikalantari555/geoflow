"""Spatial bias diagnostics and mitigation tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class SpatialBiasReport:
    """Report on spatial bias diagnostics."""

    spatial_coverage: float
    sample_density: np.ndarray
    class_distribution_bias: Dict[str, float]
    overall_bias_score: float
    underrepresented_regions: List[Any]
    recommendations: List[str]


class SpatialBiasDiagnostics:
    """Spatial bias diagnostics and mitigation for geospatial datasets.

    Features:
    - Spatial distribution analysis of training samples
    - Bias scoring based on spatial coverage
    - Mitigation strategies for biased datasets
    - Visualization of spatial bias patterns
    """

    BIAS_METRICS = [
        "spatial_coverage", "sample_density", "class_distribution",
        "distance_from_center", "boundary_proximity", "elevation_bias",
    ]

    MITIGATION_STRATEGIES = [
        "oversample_underrepresented", "spatial_stratified_sampling",
        "adaptive_weighting", "synthetic_data_generation",
    ]

    def __init__(self, crs: str = "EPSG:4326") -> None:
        self.crs = crs
        self.bias_metrics: Dict[str, float] = {}
        self.mitigation_strategy: str = "spatial_stratified_sampling"

    def compute_spatial_coverage(
        self,
        samples: List[Any],
        study_area: Any,
    ) -> float:
        """Compute spatial coverage percentage of samples within study area.

        Args:
            samples: List of training samples with spatial coordinates.
            study_area: Study area extent.

        Returns:
            Coverage percentage [0-1].
        """
        if not samples or not study_area:
            return 0.0

        sample_bboxes = []
        for sample in samples:
            if hasattr(sample, "extent") and hasattr(sample, "input"):
                sample_bboxes.append(sample.input.extent)
            elif isinstance(sample, dict) and "bbox" in sample:
                sample_bboxes.append(sample["bbox"])

        if not sample_bboxes:
            return 0.0

        # Calculate coverage using grid approximation
        grid_size = 100
        study_width = float(study_area.width) if hasattr(study_area, "width") else 1.0
        study_height = float(study_area.height) if hasattr(study_area, "height") else 1.0

        grid_cells = np.zeros((grid_size, grid_size))
        cell_w = study_width / grid_size
        cell_h = study_height / grid_size

        study_min_x = float(study_area.min_x) if hasattr(study_area, "min_x") else 0.0
        study_min_y = float(study_area.min_y) if hasattr(study_area, "min_y") else 0.0

        for bbox in sample_bboxes:
            if hasattr(bbox, "min_x"):
                sx = int((bbox.min_x - study_min_x) / cell_w)
                sy = int((bbox.min_y - study_min_y) / cell_h)
                ex = int((bbox.max_x - study_min_x) / cell_w)
                ey = int((bbox.max_y - study_min_y) / cell_h)
                sx, sy = max(0, sx), max(0, sy)
                ex, ey = min(grid_size, ex), min(grid_size, ey)
                grid_cells[sy:ey, sx:ex] = 1

        coverage = float(grid_cells.mean())
        self.bias_metrics["spatial_coverage"] = coverage
        return coverage

    def compute_sample_density(
        self,
        samples: List[Any],
        study_area: Any,
        grid_size: float = 1.0,
    ) -> np.ndarray:
        """Compute sample density across study area.

        Args:
            samples: List of training samples.
            study_area: Study area extent.
            grid_size: Grid cell size in kilometers.

        Returns:
            2D array of sample densities.
        """
        n_grid = 50
        density = np.zeros((n_grid, n_grid))

        for sample in samples:
            if hasattr(sample, "extent") and hasattr(sample, "input"):
                bbox = sample.input.extent
            elif isinstance(sample, dict):
                bbox = sample.get("bbox", [0, 0, 0, 0])
            else:
                continue

            center_x = (bbox.max_x + bbox.min_x) / 2 if hasattr(bbox, "max_x") else 0
            center_y = (bbox.max_y + bbox.min_y) / 2 if hasattr(bbox, "max_y") else 0

            cx = int((center_x - 0) / (1 / n_grid))
            cy = int((center_y - 0) / (1 / n_grid))
            cx = max(0, min(n_grid - 1, cx))
            cy = max(0, min(n_grid - 1, cy))
            density[cy, cx] += 1

        self.bias_metrics["sample_density"] = float(density.std() / density.mean() if density.mean() > 0 else 0)
        return density

    def compute_class_distribution_bias(
        self,
        samples: List[Any],
        study_area: Any,
    ) -> Dict[str, float]:
        """Compute class distribution bias across spatial regions.

        Returns:
            Dictionary of class-specific bias scores.
        """
        class_counts: Dict[str, int] = {}
        for sample in samples:
            if isinstance(sample, dict):
                cls = str(sample.get("target", sample.get("class_name", "unknown")))
            else:
                cls = "unknown"
            class_counts[cls] = class_counts.get(cls, 0) + 1

        total = sum(class_counts.values()) or 1
        class_bias = {cls: count / total for cls, count in class_counts.items()}
        self.bias_metrics["class_distribution"] = 1.0 - (len(class_bias) / max(len(class_bias), 1))
        return class_bias

    def calculate_bias_score(
        self,
        samples: List[Any],
        study_area: Any,
        method: str = "entropy",
    ) -> float:
        """Calculate overall bias score for dataset.

        Args:
            samples: List of training samples.
            study_area: Study area extent.
            method: Score method ('entropy', 'coverage', 'density').

        Returns:
            Bias score [0-1], higher = more biased.
        """
        if method == "coverage":
            coverage = self.compute_spatial_coverage(samples, study_area)
            return max(0.0, 1.0 - coverage)
        elif method == "density":
            density = self.compute_sample_density(samples, study_area)
            cv = float(density.std() / (density.mean() + 1e-8))
            return min(1.0, cv / 2.0)
        else:  # entropy
            n_locations = len(samples)
            if n_locations < 2:
                return 1.0

            self.compute_sample_density(samples, study_area)
            density = self.compute_sample_density(samples, study_area)
            flat = density.flatten() + 1e-10
            flat /= flat.sum()
            entropy = -np.sum(flat * np.log(flat))
            max_entropy = np.log(len(flat))
            return max(0.0, 1.0 - entropy / max_entropy)

    def mitigate_bias(
        self,
        samples: List[Any],
        strategy: str = "spatial_stratified_sampling",
        **kwargs: Any,
    ) -> List[Any]:
        """Apply bias mitigation strategy to dataset.

        Args:
            samples: Original dataset.
            strategy: Mitigation strategy.
            **kwargs: Strategy-specific parameters.

        Returns:
            Mitigated dataset.
        """
        # For now, simply return the samples
        # In production, this would rebalance the dataset
        return samples

    def visualize_spatial_bias(
        self,
        samples: List[Any],
        study_area: Any,
        metric: str = "sample_density",
    ) -> Any:
        """Visualize spatial bias patterns.

        Returns:
            Matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        density = self.compute_sample_density(samples, study_area)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(density, cmap="hot", interpolation="nearest")
        ax.set_title(f"Spatial Bias Map ({metric})")
        ax.set_xlabel("Longitude Grid Cell")
        ax.set_ylabel("Latitude Grid Cell")
        plt.colorbar(im, ax=ax, label="Sample Density")

        return fig