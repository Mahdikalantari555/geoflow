# I/O Backends

## Opening Data

```python
# COG (Cloud Optimized GeoTIFF)
tensor = gt.open_cog(
    's3://bucket/file.tif',
    band_info={'B04': gt.BandInfo('B04')},
    chunks={'x': 256, 'y': 256},
)

# Zarr
tensor = gt.open_zarr('s3://bucket/data.zarr', group='/S2')

# NetCDF
tensor = gt.open_netcdf('data.nc')

# STAC
tensor = gt.open_stac(
    url='https://earth-search.aws.element84.com/v1',
    collection='sentinel-2-l2a',
    datetime='2023-06-01/2023-09-30',
    bbox=(-120, 35, -119, 36),
    bands=['B04', 'B08'],
)
```

## Writing Data

```python
# Zarr
tensor.geotensor.to_zarr('output.zarr', compute=True, consolidated=True)

# COG
tensor.geotensor.to_cog('output.tif', dtype='uint16', overviews=[2, 4, 8, 16])
```

## Virtual Mosaic

```python
mosaic = gt.VRT(
    ['file1.tif', 'file2.tif', 'file3.tif'],
    method='median',    # median, mean, min, max, first
)
```

## Lazy Computation

```python
result = tensor1 + tensor2  # lazy

arr = tensor.geotensor.compute()        # numpy array
mem = tensor.geotensor.load()           # in-memory GeoTensor
cached = tensor.geotensor.persist()     # persisted Dask
rechunked = tensor.geotensor.chunk({'x': 512, 'y': 512})
```
