"""Example 09: Provenance Tracking

Demonstrates lineage tracking and export.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import geoflow as gt


def main():
    print("=== Provenance Tracking ===")

    # Create initial tensor
    data = np.random.rand(4, 32, 32).astype(np.float32)
    tensor = gt.GeoTensor(
        data,
        coords={
            'x': np.linspace(500000, 500320, 32),
            'y': np.linspace(4000000, 4000320, 32),
            'band': ['B02', 'B03', 'B04', 'B08'],
        },
        dims=['band', 'y', 'x'],
        crs='EPSG:32632',
    )

    # Access provenance
    print("\n1. Provenance Graph:")
    prov = tensor.geotensor.provenance
    print(f"   Nodes: {len(prov)}")
    print(f"   Graph: {prov.graph}")

    # Export to JSON
    print("\n2. Export to JSON:")
    json_str = tensor.geotensor.export_provenance(format='json')
    print(f"   JSON length: {len(json_str)} chars")

    # Export to DOT
    print("\n3. Export to DOT:")
    dot_str = tensor.geotensor.export_provenance(format='dot')
    print(f"   DOT preview:\n{dot_str[:200]}...")

    # Freeze tensor
    print("\n4. Freeze Tensor:")
    tensor.geotensor.freeze()
    print(f"   Frozen: {prov.frozen}")

    # Try to modify (should raise error)
    try:
        tensor.geotensor.fill(value=0.0)
    except gt.ImmutableGeoTensorError as e:
        print(f"   Error caught: {type(e).__name__}")

    # Unfreeze
    tensor.geotensor.unfreeze()
    print(f"   Unfrozen: {prov.frozen}")

    print("\nDone!")


if __name__ == '__main__':
    main()
