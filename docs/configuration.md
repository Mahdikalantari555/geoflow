# Configuration

Global configuration via `gt.config`:

```python
from geoflow import config

# Provenance settings
config.provenance_depth = 10      # keep last N nodes (None = unlimited)
config.provenance_mode = 'minimal' # 'minimal' | 'full'

# Chunking defaults
config.default_chunks = {'x': 512, 'y': 512, 'time': 1, 'band': 4}

# Unit safety
config.unit_safe_by_default = True

# Memory limit
config.memory_limit_gb = 8.0  # None = no limit
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `provenance_depth` | `None` | Max provenance nodes to retain |
| `provenance_mode` | `'full'` | `'minimal'` or `'full'` |
| `default_chunks` | `{x: 256, y: 256, time: 1, band: 4}` | Default Dask chunk sizes |
| `unit_safe_by_default` | `True` | Enable unit safety checks |
| `memory_limit_gb` | `None` | Memory limit for `.compute()` |
