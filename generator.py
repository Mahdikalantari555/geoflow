"""Synthetic data generation for geospatial remote sensing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    seed: int = 42
    preserve_original: bool = True
    max_perturbation: float = 0.1
    output_dtype: str = "float32"
    generate_geotiffs: bool = False
    output_path: str = "./synthetic_data"


class GeoSyntheticGenerator:
    """Synthetic data generation for geospatial applications.

    Enables spectral perturbation, atmospheric simulation, SAR noise injection,
    DEM warping, synthetic object insertion, and seasonal variation generation.
    """

    def __init__(self, config: Optional[SyntheticConfig] = None) -> None:
        self.config = config or SyntheticConfig()
        torch.manual_seed(self.config.seed)

    def spectral_perturbation(
        self,
        geotensor: Any,
        noise_level: float = 0.05,
        wavelength_shift: float = 0.0,
    ) -> Any:
        """Add spectral noise and wavelength perturbations.

        Args:
            geotensor: Input GeoTensor [C, H, W].
            noise_level: Standard deviation of Gaussian noise.
            wavelength_shift: Shift applied to band values.

        Returns:
            GeoTensor with spectral perturbations applied.
        """
        data = geotensor.data.clone()
        noise = torch.randn_like(data) * noise_level
        data = data + noise + wavelength_shift
        data = data.clamp(min=0.0)
        geotensor.data = data
        return geotensor

    def atmospheric_simulation(
        self,
        geotensor: Any,
        aerosol_model: str = "continental",
        visibility_km: float = 15.0,
    ) -> Any:
        """Simulate atmospheric effects (haze, scattering).

        Args:
            geotensor: Input GeoTensor [C, H, W].
            aerosol_model: Type of aerosol model.
            visibility_km: Visibility distance in kilometers.

        Returns:
            GeoTensor with atmospheric effects applied.
        """
        data = geotensor.data.clone()
        # Simple haze simulation
        haze_factor = max(0.0, 1.0 - visibility_km / 50.0)
        haze = torch.ones_like(data) * haze_factor * 0.3
        data = data * (1 - haze_factor * 0.5) + haze
        data = data.clamp(0, 1)
        geotensor.data = data
        return geotensor

    def sar_noise_injection(
        self,
        geotensor: Any,
        noise_type: str = "speckle",
        speckle_window: int = 3,
        snr_db: float = 10.0,
    ) -> Any:
        """Inject SAR-specific noise patterns.

        Args:
            geotensor: Input GeoTensor [C, H, W].
            noise_type: Type of noise ('speckle', 'thermal').
            speckle_window: Window size for speckle generation.
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            GeoTensor with SAR noise injected.
        """
        data = geotensor.data.clone()
        if noise_type == "speckle":
            # Multiplicative speckle noise
            noise = torch.randn_like(data) * (10.0 ** (-snr_db / 20.0))
            data = data * (1.0 + noise)
        elif noise_type == "thermal":
            noise = torch.randn_like(data) * (10.0 ** (-snr_db / 20.0))
            data = data + noise
        data = data.clamp(min=0.0)
        geotensor.data = data
        return geotensor

    def synthetic_object_insertion(
        self,
        geotensor: Any,
        object_mask: Any,
        object_template: Any,
        blend_mode: str = "poisson",
    ) -> Any:
        """Insert synthetic objects into scene.

        Args:
            geotensor: Input GeoTensor [C, H, W].
            object_mask: Binary mask for object insertion.
            object_template: Template of the object to insert.
            blend_mode: Blending method ('overlay', 'additive').

        Returns:
            GeoTensor with objects inserted.
        """
        data = geotensor.data.clone()
        mask = object_mask.data if hasattr(object_mask, "data") else object_mask
        template = object_template.data if hasattr(object_template, "data") else object_template

        if blend_mode == "overlay":
            data = data * (1 - mask) + template * mask
        elif blend_mode == "additive":
            data = data + template * mask

        data = data.clamp(min=0.0)
        geotensor.data = data
        return geotensor

    def seasonal_variation(
        self,
        geotensor: Any,
        target_season: str,
        hemisphere: str = "northern",
    ) -> Any:
        """Generate seasonal variation of scene.

        Args:
            geotensor: Input GeoTensor [C, H, W].
            target_season: Target season ('spring', 'summer', 'autumn', 'winter').
            hemisphere: Hemisphere ('northern', 'southern').

        Returns:
            GeoTensor with seasonal variation.
        """
        season_factors = {
            "spring": {"brightness": 1.0, "contrast": 1.0, "green_boost": 0.1},
            "summer": {"brightness": 1.1, "contrast": 1.2, "green_boost": 0.2},
            "autumn": {"brightness": 0.9, "contrast": 0.9, "green_boost": -0.2},
            "winter": {"brightness": 0.7, "contrast": 0.8, "green_boost": -0.3},
        }

        factors = season_factors.get(target_season.lower(), season_factors["summer"])
        data = geotensor.data.clone()
        data = data * factors["brightness"]

        if factors["green_boost"] != 0 and data.shape[0] >= 3:
            data[1] = data[1] + factors["green_boost"]

        data = data.clamp(min=0.0)
        geotensor.data = data
        return geotensor

    def __call__(self, geotensor: Any, transform_type: str = "spectral_perturbation") -> Any:
        """Apply a random synthetic transform to the input.

        Args:
            geotensor: Input GeoTensor.
            transform_type: Type of transform to apply.

        Returns:
            Transformed GeoTensor.
        """
        transform_map = {
            "spectral_perturbation": self.spectral_perturbation,
            "atmospheric_simulation": self.atmospheric_simulation,
            "sar_noise_injection": self.sar_noise_injection,
            "synthetic_object_insertion": self.synthetic_object_insertion,
            "seasonal_variation": self.seasonal_variation,
        }
        transform_fn = transform_map.get(transform_type, self.spectral_perturbation)
        return transform_fn(geotensor)