# Temporal Operations

## Time Properties

```python
tensor.geotensor.time_interval    # pd.Timedelta (median gap)
tensor.geotensor.time_metadata    # {'start': ..., 'end': ..., 'acquisition_mode': ...}
```

## Resampling

```python
regular = tensor.geotensor.resample_time(
    rule='1D',               # resampling rule
    method='nearest',        # nearest, ffill, bfill, linear
    fill_value=np.nan,
)
```

## Interpolation

```python
interpolated = tensor.geotensor.temporal_interpolate(
    method='linear',         # linear, nearest, cubic
    limit=2,                 # max consecutive NaNs to fill
)
```

## Windowing

```python
window = tensor.geotensor.temporal_window(
    start='2023-06-01',
    end='2023-09-30',
)
```

## Aggregation

```python
monthly = tensor.geotensor.temporal_aggregate(
    period='M',              # M, Y, season
    func='mean',             # mean, sum, min, max, median, std
)
```

## Event-Based Sampling

```python
events = tensor.geotensor.select_by_event(
    event_times=['2023-06-15', '2023-07-01'],
    tolerance='1D',
)

relative = tensor.geotensor.time_to_relative()
```
