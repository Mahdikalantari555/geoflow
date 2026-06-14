# Alignment Engine

The alignment engine brings multiple GeoTensors to a common spatial grid, temporal axis, and spectral band set.

## AlignmentSpec

```python
from geoflow import AlignmentSpec, SpatialAlignment, TemporalAlignment, SpectralAlignment

spec = AlignmentSpec(
    spatial=SpatialAlignment(
        crs='EPSG:32632',           # Target CRS (None = auto)
        resolution=10.0,            # Target resolution (None = finest)
        method='bilinear',          # Resampling method
        extent='intersection',      # intersection, union, or bbox
    ),
    temporal=TemporalAlignment(
        method='intersection',      # intersection, union, reference
        interpolation='nearest',
        tolerance='1D',
    ),
    spectral=SpectralAlignment(
        method='match_band',        # match_band, interpolate, drop
        bandset='union',            # union, intersection, or list
        wavelength_tolerance_nm=10,
    ),
)
```

## Multi-Tensor Alignment

```python
aligned = gt.align(sentinel1, sentinel2, spec=spec)
s1_aligned, s2_aligned = aligned

# Now safe to combine
result = s1_aligned + s2_aligned
```

## Alignment to Reference

```python
aligned = gt.align_to_reference(
    sentinel2, landsat,
    reference=sentinel1,
)
```

## Spatial Alignment

Resamples all tensors to a common CRS, resolution, and extent.

| Parameter | Options | Description |
|-----------|---------|-------------|
| `crs` | Any CRS string | Target coordinate reference system |
| `resolution` | Float | Target resolution in CRS units |
| `method` | `bilinear`, `nearest`, `cubic`, `average` | Resampling method |
| `extent` | `intersection`, `union`, BoundingBox | How to compute output extent |

## Temporal Alignment

Aligns time axes across tensors.

| Parameter | Options | Description |
|-----------|---------|-------------|
| `method` | `intersection`, `union`, `reference` | How to combine timestamps |
| `interpolation` | `nearest`, `linear`, `ffill` | Fill method for union |
| `tolerance` | Time string (e.g., `1D`, `5D`) | Max gap for matching |

## Spectral Alignment

Aligns band dimensions across tensors.

| Parameter | Options | Description |
|-----------|---------|-------------|
| `method` | `match_band`, `interpolate`, `drop` | Band matching strategy |
| `bandset` | `union`, `intersection`, list | Which bands to keep |
| `wavelength_tolerance_nm` | Float | Max wavelength difference |

## Performance

- Alignment uses `rioxarray.reproject_match` for spatial operations
- Lazy evaluation: reprojection weights computed only when data is read
- For repeated alignments, pre-compute the target grid
