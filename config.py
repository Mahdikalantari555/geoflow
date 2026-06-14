"""Global configuration for GeoFlow.

Controls provenance depth, default chunking, unit safety, and memory limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class GeoFlowConfig:
    """Global GeoFlow configuration.

    Attributes:
        provenance_depth: Max number of provenance nodes to retain
                          (``None`` = unlimited).
        provenance_mode: ``'minimal'`` (operation name only) or
                         ``'full'`` (all parameters).
        default_chunks: Default Dask chunk sizes.
        unit_safe_by_default: If ``True``, unit safety checks are
                              enabled by default.
        memory_limit_gb: Optional memory limit (GB) for ``.compute()``.
    """
    provenance_depth: Optional[int] = None
    provenance_mode: str = 'full'
    default_chunks: Dict[str, int] = field(default_factory=lambda: {
        'x': 256, 'y': 256, 'time': 1, 'band': 4,
    })
    unit_safe_by_default: bool = True
    memory_limit_gb: Optional[float] = None


# Global singleton
config = GeoFlowConfig()

__all__ = ['config', 'GeoFlowConfig']