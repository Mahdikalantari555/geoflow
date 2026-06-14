# Unit Safety

## Unit Registry

```python
ureg = gt.GeoUnitRegistry()
ureg('reflectance')                # <Unit('dimensionless')>
ureg('radiance')                   # <Unit('W / m**2 / sr / um')>
ureg('brightness_temperature')     # <Unit('K')>
```

## Band Units

```python
units = tensor.geotensor.band_units()
# {'B04': 'dimensionless', 'B08': 'dimensionless'}
```

## Unit Conversion

```python
converted = tensor.geotensor.convert_units(to='radiance')
converted = tensor.geotensor.convert_units(to='K', from_unit='brightness_temperature')
```

## Unit Safety Context

```python
with gt.unit_safe(False):
    normalized = (tensor - mean) / std
```

## Unit-Aware Arithmetic

```python
# Same units: allowed
result = reflectance_tensor + reflectance_tensor

# Different units: raises UnitMismatchError
result = radiance_tensor + reflectance_tensor  # Error!
```
