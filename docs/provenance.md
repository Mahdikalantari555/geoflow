# Provenance Tracking

## Provenance Graph

```python
prov = tensor.geotensor.provenance  # ProvenanceGraph
print(prov.graph)                    # list of operation dicts
```

## Export Formats

```python
# JSON
json_str = tensor.geotensor.export_provenance(format='json')

# JSON-LD
jsonld_str = tensor.geotensor.export_provenance(format='json-ld')

# Graphviz DOT
dot_str = tensor.geotensor.export_provenance(format='dot')
```

## Freeze / Unfreeze

```python
tensor.geotensor.freeze()    # read-only
tensor.geotensor.unfreeze()  # writable again
gt.unfreeze(tensor)          # alternative
```

## Reproduction

```python
# Export provenance
with open('provenance.json', 'w') as f:
    f.write(tensor.geotensor.export_provenance())

# Reproduce from provenance
reproduced = gt.reproduce('provenance.json')
```

## Limitations

- Reproduction supports: reproject, resample, slice_bbox, resample_time, temporal_window, temporal_aggregate, fill, convert_units, align
- Source data must be accessible for reproduction
- Unknown operations raise `NotImplementedError`
