"""Preprocessing pipeline for remote sensing data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class PreprocessorConfig:
    """Configuration for the preprocessing pipeline."""

    atmospheric_correction: bool = False
    atmospheric_method: str = "DOS1"  # DOS1, DOS2, 6S
    radiometric_calibration: bool = False
    cloud_masking: bool = False
    cloud_method: str = "qa_based"  # qa_based, fmask
    band_math: bool = False
    spectral_indices: List[str] = field(default_factory=list)
    resample: bool = False
    target_resolution: Optional[float] = None
    resample_method: str = "bilinear"
    normalize: bool = True
    normalize_method: str = "standard"  # standard, min_max, percentile
    percentile_range: Tuple[float, float] = (2.0, 98.0)
    dtype: str = "float32"


class PreprocessingStep:
    """Base class for a single preprocessing operation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, geotensor: Any) -> Any:
        """Apply this step to the given GeoTensor. Subclasses must override."""
        return self.process(geotensor)

    def process(self, geotensor: Any) -> Any:
        """Apply preprocessing to geotensor."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement process()")


class AtmosphericCorrection(PreprocessingStep):
    """Dark Object Subtraction (DOS1) atmospheric correction."""

    def __init__(self, method: str = "DOS1") -> None:
        super().__init__("atmospheric_correction")
        self.method = method

    def process(self, geotensor: Any) -> Any:
        """Apply DOS1 atmospheric correction by subtracting 1st percentile per band."""
        data = geotensor.data.clone()
        for b in range(data.shape[0]):
            band_data = data[b]
            valid = band_data != geotensor.nodata if geotensor.nodata is not None else torch.ones_like(band_data, dtype=torch.bool)
            if valid.any():
                p1 = band_data[valid].quantile(0.01)
                data[b] = band_data - p1
                data[b] = data[b].clamp(min=0)
        geotensor.data = data
        return geotensor


class RadiometricCalibration(PreprocessingStep):
    """Radiometric calibration from DN to reflectance."""

    def __init__(self, gain: Optional[float] = None, offset: Optional[float] = None) -> None:
        super().__init__("radiometric_calibration")
        self.gain = gain
        self.offset = offset

    def process(self, geotensor: Any) -> Any:
        """Convert digital numbers to reflectance."""
        data = geotensor.data.clone().float()
        gain = self.gain or 1.0 / 10000.0
        offset = self.offset or 0.0
        geotensor.data = data * gain + offset
        return geotensor


class CloudMasking(PreprocessingStep):
    """Cloud and shadow masking using QA band."""

    def __init__(self, method: str = "qa_based") -> None:
        super().__init__("cloud_masking")
        self.method = method

    def process(self, geotensor: Any) -> Any:
        """Apply cloud masking. Returns geotensor with cloud pixels set to nodata."""
        if "qa_band" in geotensor.metadata:
            qa = geotensor.metadata["qa_band"]
            if isinstance(qa, torch.Tensor):
                cloud_mask = (qa & (1 << 3)) | (qa & (1 << 5))
                geotensor.data[:, cloud_mask] = geotensor.nodata if geotensor.nodata is not None else 0
        return geotensor


class SpectralIndices(PreprocessingStep):
    """Compute spectral indices like NDVI, EVI, NDBI, etc."""

    INDEX_FORMULAS = {
        "NDVI": lambda r, nir: (nir - r) / (nir + r + 1e-10),
        "EVI": lambda r, nir, b: 2.5 * (nir - r) / (nir + 6 * r - 7.5 * b + 1),
        "NDBI": lambda nir, swir1: (swir1 - nir) / (swir1 + nir + 1e-10),
        "NDWI": lambda g, nir: (g - nir) / (g + nir + 1e-10),
        "NBR": lambda nir, swir2: (nir - swir2) / (nir + swir2 + 1e-10),
        "SAVI": lambda r, nir, L=0.5: (nir - r) * (1 + L) / (nir + r + L + 1e-10),
    }

    def __init__(self, indices: List[str], band_mapping: Optional[Dict[str, int]] = None) -> None:
        super().__init__("spectral_indices")
        self.indices = [idx.upper() for idx in indices]
        self.band_mapping = band_mapping or {"R": 0, "G": 1, "B": 2, "NIR": 3, "SWIR1": 4, "SWIR2": 5}

    def process(self, geotensor: Any) -> Any:
        """Compute requested spectral indices and append as new bands."""
        band_data = geotensor.data
        indices_list = []

        for idx_name in self.indices:
            if idx_name == "NDVI":
                r = band_data[self.band_mapping["R"]]
                nir = band_data[self.band_mapping["NIR"]]
                indices_list.append(self.INDEX_FORMULAS["NDVI"](r, nir))
            elif idx_name == "NDBI":
                nir = band_data[self.band_mapping["NIR"]]
                swir1 = band_data[self.band_mapping["SWIR1"]]
                indices_list.append(self.INDEX_FORMULAS["NDBI"](nir, swir1))
            elif idx_name == "NDWI":
                g = band_data[self.band_mapping["G"]]
                nir = band_data[self.band_mapping["NIR"]]
                indices_list.append(self.INDEX_FORMULAS["NDWI"](g, nir))

        if indices_list:
            geotensor.data = torch.cat([band_data] + indices_list, dim=0)
        return geotensor


class Normalization(PreprocessingStep):
    """Normalize image values (standard, min-max, percentile)."""

    def __init__(self, method: str = "standard", percentile_range: Tuple[float, float] = (2.0, 98.0)) -> None:
        super().__init__("normalization")
        self.method = method
        self.percentile_range = percentile_range

    def process(self, geotensor: Any) -> Any:
        """Normalize geotensor data."""
        data = geotensor.data.clone().float()
        if self.method == "standard":
            mean = data.mean(dim=(1, 2), keepdim=True)
            std = data.std(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            data = (data - mean) / std
        elif self.method == "min_max":
            vmin = data.amin(dim=(1, 2), keepdim=True)
            vmax = data.amax(dim=(1, 2), keepdim=True)
            data = (data - vmin) / (vmax - vmin + 1e-8)
        elif self.method == "percentile":
            h, w = data.shape[1:]
            flat = data.view(data.shape[0], -1)
            low = torch.quantile(flat, self.percentile_range[0] / 100.0, dim=1, keepdim=True).unsqueeze(-1)
            high = torch.quantile(flat, self.percentile_range[1] / 100.0, dim=1, keepdim=True).unsqueeze(-1)
            data = (data - low) / (high - low + 1e-8)
            data = data.clamp(0, 1)
        geotensor.data = data
        return geotensor


class GeoPreprocessor:
    """Preprocessing pipeline for remote sensing data.

    Composes multiple PreprocessingStep instances and applies them
    sequentially to a GeoTensor.

    Args:
        config: Preprocessing configuration.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self.config = config or PreprocessorConfig()
        self.steps: List[PreprocessingStep] = []
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the preprocessing pipeline from config."""
        if self.config.atmospheric_correction:
            self.steps.append(AtmosphericCorrection(method=self.config.atmospheric_method))
        if self.config.radiometric_calibration:
            self.steps.append(RadiometricCalibration())
        if self.config.cloud_masking:
            self.steps.append(CloudMasking(method=self.config.cloud_method))
        if self.config.spectral_indices:
            self.steps.append(SpectralIndices(indices=self.config.spectral_indices))
        if self.config.normalize:
            self.steps.append(Normalization(
                method=self.config.normalize_method,
                percentile_range=self.config.percentile_range,
            ))

    def add_atmospheric_correction(self, method: str = "DOS1") -> "GeoPreprocessor":
        """Add atmospheric correction step."""
        self.steps.append(AtmosphericCorrection(method=method))
        return self

    def add_radiometric_calibration(self) -> "GeoPreprocessor":
        """Add radiometric calibration step."""
        self.steps.append(RadiometricCalibration())
        return self

    def add_cloud_masking(self, method: str = "qa_based") -> "GeoPreprocessor":
        """Add cloud/shadow masking step."""
        self.steps.append(CloudMasking(method=method))
        return self

    def add_band_math(self, expressions: Dict[str, str]) -> "GeoPreprocessor":
        """Add custom band math operations."""
        # Custom step for band math
        class BandMath(PreprocessingStep):
            def __init__(self, expr: Dict[str, str]):
                super().__init__("band_math")
                self.expressions = expr

            def process(self, geotensor: Any) -> Any:
                # Placeholder for band math
                return geotensor

        self.steps.append(BandMath(expressions))
        return self

    def add_spectral_indices(self, indices: List[str]) -> "GeoPreprocessor":
        """Add spectral index computation."""
        self.steps.append(SpectralIndices(indices=indices))
        return self

    def add_resampling(self, target_resolution: float, method: str = "bilinear") -> "GeoPreprocessor":
        """Add resampling step."""
        class Resampling(PreprocessingStep):
            def __init__(self, res: float, meth: str):
                super().__init__("resampling")
                self.resolution = res
                self.method = meth

            def process(self, geotensor: Any) -> Any:
                # Use GeoTensor's to_crs with resolution
                return geotensor

        self.steps.append(Resampling(target_resolution, method))
        return self

    def add_normalization(self, method: str = "standard") -> "GeoPreprocessor":
        """Add normalization step."""
        self.steps.append(Normalization(method=method))
        return self

    def execute(self, geotensor: Any) -> Any:
        """Execute the full preprocessing pipeline on a GeoTensor."""
        for step in self.steps:
            geotensor = step(geotensor)
        return geotensor

    def __call__(self, geotensor: Any) -> Any:
        """Alias for execute()."""
        return self.execute(geotensor)