# Examples Index

Runnable scripts in the `examples/` folder:

| Script | Description |
|--------|-------------|
| `01_basic_geotensor.py` | Create GeoTensor, inspect properties |
| `02_open_and_write.py` | Open COG/Zarr, write to Zarr/COG |
| `03_spatial_alignment.py` | Align two tensors with different CRS/resolution |
| `04_temporal_operations.py` | Resample time, interpolate, aggregate |
| `05_masking.py` | Create masks, combine, fill |
| `06_patch_sampling.py` | Tile/random/grid sampling for ML |
| `07_ml_pipeline.py` | Full ML pipeline: open → normalize → DataLoader → train |
| `08_unit_safe_math.py` | Unit-aware arithmetic, conversion |
| `09_provenance_tracking.py` | Track ops, export, inspect lineage |
| `10_differentiable.py` | Differentiable resampling, grad tracking |

## Running Examples

```bash
cd examples
python 01_basic_geotensor.py
```

All examples use synthetic data and require no external files.
