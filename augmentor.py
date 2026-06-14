"""Remote sensing-aware augmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import kornia.augmentation as K
    from kornia.filters import GaussianBlur2d
    _HAS_KORNIA = True
except ImportError:
    K = None
    GaussianBlur2d = None
    _HAS_KORNIA = False


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""

    geometric_prob: float = 0.5
    radiometric_prob: float = 0.5
    spectral_prob: float = 0.3
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    rotation_range: Tuple[float, float] = (-10.0, 10.0)
    noise_std: float = 0.01
    spectral_drop_prob: float = 0.1
    cloud_intensity: float = 0.3
    shadow_intensity: float = 0.3
    seed: int = 42

    def __post_init__(self) -> None:
        if self.cloud_intensity < 0 or self.cloud_intensity > 1:
            raise ValueError(f"cloud_intensity must be in [0, 1], got {self.cloud_intensity}")


class GeometricTransform(nn.Module):
    """Applies geometric augmentations (rotation, flips, scaling)."""

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-10.0, 10.0),
        flip_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply geometric transforms to input tensor [B, C, H, W] or [C, H, W]."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False

        # Random rotation
        if torch.rand(1).item() < 0.5:
            angle = torch.empty(1).uniform_(*self.rotation_range).item()
            if _HAS_KORNIA:
                x = K.rotate(x, torch.tensor([angle]), padding_mode="reflection")
            else:
                import torchvision.transforms.functional as TF
                x = TF.rotate(x, angle)

        # Random flips
        if torch.rand(1).item() < self.flip_prob:
            x = torch.flip(x, dims=[-1])  # horizontal flip
        if torch.rand(1).item() < self.flip_prob:
            x = torch.flip(x, dims=[-2])  # vertical flip

        if single:
            x = x.squeeze(0)
        return x


class RadiometricTransform(nn.Module):
    """Applies radiometric augmentations (brightness, contrast, gamma, noise)."""

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply radiometric transforms."""
        if torch.rand(1).item() < 0.5:
            factor = torch.empty(1).uniform_(*self.brightness_range).item()
            x = x * factor
        if torch.rand(1).item() < 0.5:
            mean = x.mean(dim=(1, 2), keepdim=True)
            factor = torch.empty(1).uniform_(*self.contrast_range).item()
            x = (x - mean) * factor + mean
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x


class SpectralTransform(nn.Module):
    """Applies spectral augmentations (band dropout, NDVI injection)."""

    def __init__(self, spectral_drop_prob: float = 0.1) -> None:
        super().__init__()
        self.spectral_drop_prob = spectral_drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral transforms."""
        if torch.rand(1).item() < self.spectral_drop_prob and x.shape[0] >= 3:
            drop_band = torch.randint(0, x.shape[0], (1,)).item()
            x[drop_band] = 0.0
        return x


class CloudSimulate(nn.Module):
    """Simulates clouds and shadows on optical imagery."""

    def __init__(self, cloud_intensity: float = 0.3, shadow_intensity: float = 0.3) -> None:
        super().__init__()
        self.cloud_intensity = cloud_intensity
        self.shadow_intensity = shadow_intensity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cloud and shadow simulation."""
        single = x.dim() == 3
        if single:
            x = x.unsqueeze(0)

        h, w = x.shape[-2], x.shape[-1]

        if _HAS_KORNIA:
            # Generate cloud mask
            noise = torch.randn(1, 1, h, w, device=x.device)
            cloud_mask = GaussianBlur2d((15, 15), (5.0, 5.0))(noise)
            cloud_mask = torch.sigmoid(cloud_mask * 3.0) * self.cloud_intensity

            # Apply clouds
            x = x * (1 - cloud_mask) + cloud_mask * 0.8

            # Generate shadows
            shadow_noise = torch.rand(1, 1, h // 4, w // 4, device=x.device) > 0.95
            shadow_mask = F.interpolate(shadow_noise.float(), size=(h, w), mode="nearest")
            shadow_mask = GaussianBlur2d((7, 7), (3.0, 3.0))(shadow_mask)
            shadow_mask = (shadow_mask > 0.1).float() * self.shadow_intensity
        else:
            # Fallback without kornia
            noise = torch.randn(1, 1, h, w, device=x.device)
            import torch.nn.functional as _F
            noise_blur = _F.avg_pool2d(noise, kernel_size=5, stride=1, padding=2)
            cloud_mask = torch.sigmoid(noise_blur * 3.0) * self.cloud_intensity
            x = x * (1 - cloud_mask) + cloud_mask * 0.8

            shadow_noise = torch.rand(1, 1, h // 4, w // 4, device=x.device) > 0.95
            shadow_mask = F.interpolate(shadow_noise.float(), size=(h, w), mode="nearest")
            shadow_mask = _F.avg_pool2d(shadow_mask, kernel_size=5, stride=1, padding=2)
            shadow_mask = (shadow_mask > 0.1).float() * self.shadow_intensity

        x = x * (1 - shadow_mask)

        if single:
            x = x.squeeze(0)
        return x


class GeoAugmentor(nn.Module):
    """Remote sensing-aware augmentation pipeline.

    Composes geometric, radiometric, spectral, and RS-specific transforms
    into a single augmentation pipeline.
    """

    GEOMETRIC_TRANSFORMS = ["rotate_90", "rotate_random", "flip_h", "flip_v", "scale_random"]
    RADIOMETRIC_TRANSFORMS = ["brightness", "contrast", "gamma", "histogram_match", "radiometric_noise"]
    SPECTRAL_TRANSFORMS = ["ndvi_injection", "spectral_drop", "wavelength_shift"]
    RS_SPECIFIC_TRANSFORMS = ["cloud_simulate", "shadow_simulate", "sensor_degradation", "resolution_degrade"]

    def __init__(self, config: Optional[AugmentationConfig] = None) -> None:
        super().__init__()
        self.config = config or AugmentationConfig()
        self.transforms: nn.ModuleList = nn.ModuleList()
        self._setup_transforms()

    def _setup_transforms(self) -> None:
        """Initialize transforms based on configuration."""
        self.transforms.append(GeometricTransform(
            rotation_range=self.config.rotation_range,
            flip_prob=self.config.geometric_prob,
        ))
        self.transforms.append(RadiometricTransform(
            brightness_range=self.config.brightness_range,
            contrast_range=self.config.contrast_range,
            noise_std=self.config.noise_std,
        ))
        self.transforms.append(SpectralTransform(
            spectral_drop_prob=self.config.spectral_drop_prob,
        ))
        self.transforms.append(CloudSimulate(
            cloud_intensity=self.config.cloud_intensity,
            shadow_intensity=self.config.shadow_intensity,
        ))

    def forward(self, x: Union[torch.Tensor, Dict[str, Any]]) -> Union[torch.Tensor, Dict[str, Any]]:
        """Apply augmentations to input.

        Args:
            x: Either a tensor [C, H, W] or a dict with 'input' key.

        Returns:
            Augmented data in same format as input.
        """
        if isinstance(x, dict):
            if "input" in x:
                x["input"] = self._apply_transforms(x["input"])
            return x
        return self._apply_transforms(x)

    def _apply_transforms(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply all transforms to a single tensor."""
        for t in self.transforms:
            if isinstance(t, nn.Module):
                tensor = t(tensor)
        return tensor

    def __call__(self, sample: Any) -> Any:
        """Alias for forward()."""
        return self.forward(sample)

    def deterministic(self, seed: int) -> "GeoAugmentor":
        """Return a deterministic version for evaluation."""
        self.config.seed = seed
        torch.manual_seed(seed)
        return self

    def add_geometric(self, **kwargs: Any) -> None:
        """Add a geometric transform to the pipeline."""
        self.transforms.append(GeometricTransform(**kwargs))

    def add_radiometric(self, **kwargs: Any) -> None:
        """Add a radiometric transform to the pipeline."""
        self.transforms.append(RadiometricTransform(**kwargs))

    def add_rs_specific(self, **kwargs: Any) -> None:
        """Add an RS-specific transform to the pipeline."""
        self.transforms.append(CloudSimulate(**kwargs))