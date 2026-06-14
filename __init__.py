"""
GeoFlow – Geospatial tensor operations built on Xarray.

Core data type: :class:`GeoTensor` (subclass of :class:`xr.DataArray`),
providing labelled multi-dimensional arrays with CRS, geotransform,
band semantics, mask, and provenance tracking.

Accessor: ``.geotensor`` is registered on every :class:`xr.DataArray`,
providing geospatial, temporal, spectral, and ML operations.

Usage::

    import geoflow as gt

    # Open data
    s2 = gt.open_cog("s2.tif", band_info={"B04": gt.BandInfo("B04")})

    # Align multiple sources
    s1, s2 = gt.align(s1, s2)

    # Sample patches for ML
    patches = s2.geotensor.sample_tiles(size=256)

    # Convert to PyTorch
    tensor = s2.geotensor.to_pytorch()
"""

from __future__ import annotations

from .core import (
    # Exception classes
    CRSMismatchError,
    DataNotAvailableError,
    GeoFlowError,
    ImmutableGeoTensorError,
    NonDifferentiableError,
    ResolutionMismatchError,
    UnitMismatchError,
    # Dataclasses
    BandInfo,
    GeoExtent,
    GeoGrid,
    # Core type
    GeoTensor,
    # Helpers
    assert_compatible,
)

from .config import config, GeoFlowConfig

from .align import (
    AlignmentSpec,
    SpatialAlignment,
    TemporalAlignment,
    SpectralAlignment,
    align,
    align_to_reference,
)

from .backends import (
    open_cog,
    open_zarr,
    open_netcdf,
    to_zarr,
    to_cog,
    VRT,
)

from .provenance import (
    ProvenanceNode,
    ProvenanceGraph,
    export_provenance,
    reproduce,
    freeze,
)

from .units import (
    GeoUnitRegistry,
    unit_safe,
)

from .streaming import (
    STACStreamingEngine,
    STACConfig,
    STACQuery,
    STACItem,
    open_stac,
)

# Trigger registration of the ``.geotensor`` accessor
from . import accessor  # noqa: F401

__all__ = [
    # Core
    'GeoTensor',
    'BandInfo',
    'GeoExtent',
    'GeoGrid',
    # Exceptions
    'GeoFlowError',
    'CRSMismatchError',
    'DataNotAvailableError',
    'ResolutionMismatchError',
    'UnitMismatchError',
    'ImmutableGeoTensorError',
    'NonDifferentiableError',
    # Helpers
    'assert_compatible',
    # Config
    'config',
    'GeoFlowConfig',
    # Alignment
    'AlignmentSpec',
    'SpatialAlignment',
    'TemporalAlignment',
    'SpectralAlignment',
    'align',
    'align_to_reference',
    # I/O
    'open_cog',
    'open_zarr',
    'open_netcdf',
    'to_zarr',
    'to_cog',
    'VRT',
    # Provenance
    'ProvenanceNode',
    'ProvenanceGraph',
    'export_provenance',
    'reproduce',
    'freeze',
    # Units
    'GeoUnitRegistry',
    'unit_safe',
    # STAC
    'STACStreamingEngine',
    'STACConfig',
    'STACQuery',
    'STACItem',
    'open_stac',
]

__version__ = '0.0.1'