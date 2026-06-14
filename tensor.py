"""
Deprecated — use ``geoflow.core`` instead.

All geospatial tensor functionality has moved to :mod:`geoflow.core`
(:class:`~geoflow.core.GeoTensor`), :mod:`geoflow.accessor`
(:class:`~geoflow.accessor.GeoTensorAccessor`), and submodules.

This module is kept for backward compatibility only.
"""

import warnings

warnings.warn(
    "geoflow.tensor is deprecated. Use geoflow.core (GeoTensor) "
    "and geoflow.accessor (.geotensor accessor) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .core import (
    BandInfo,
    GeoExtent,
    GeoGrid,
    GeoTensor,
    GeoFlowError,
    CRSMismatchError,
    ResolutionMismatchError,
    UnitMismatchError,
    ImmutableGeoTensorError,
    NonDifferentiableError,
)
from .provenance import ProvenanceNode, ProvenanceGraph

# Re-export legacy names
BandMetadata = BandInfo
ProvenanceRecord = ProvenanceNode

__all__ = [
    'GeoTensor',
    'BandMetadata',
    'BandInfo',
    'GeoExtent',
    'GeoGrid',
    'ProvenanceRecord',
    'ProvenanceNode',
    'ProvenanceGraph',
    'GeoFlowError',
    'CRSMismatchError',
    'ResolutionMismatchError',
    'UnitMismatchError',
    'ImmutableGeoTensorError',
    'NonDifferentiableError',
]