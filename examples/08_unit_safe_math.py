"""Example 08: Unit-Safe Math

Demonstrates unit-aware arithmetic and conversion.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import geoflow as gt


def main():
    print("=== Unit-Safe Math ===")

    # Create tensors with different units
    reflectance_data = np.random.rand(4, 32, 32).astype(np.float32) * 0.5
    reflectance = gt.GeoTensor(
        reflectance_data,
        coords={
            'x': np.linspace(500000, 500320, 32),
            'y': np.linspace(4000000, 4000320, 32),
            'band': ['B02', 'B03', 'B04', 'B08'],
        },
        dims=['band', 'y', 'x'],
        crs='EPSG:32632',
        band_info={
            'B02': gt.BandInfo('B02', unit='reflectance'),
            'B03': gt.BandInfo('B03', unit='reflectance'),
            'B04': gt.BandInfo('B04', unit='reflectance'),
            'B08': gt.BandInfo('B08', unit='reflectance'),
        },
    )

    # Unit registry
    print("\n1. Unit Registry:")
    ureg = gt.GeoUnitRegistry()
    print(f"   reflectance: {ureg('reflectance')}")
    print(f"   radiance: {ureg('radiance')}")
    print(f"   brightness_temperature: {ureg('brightness_temperature')}")

    # Band units
    print("\n2. Band Units:")
    units = reflectance.geotensor.band_units()
    print(f"   Units: {units}")

    # Safe arithmetic (same units)
    print("\n3. Safe Arithmetic (same units):")
    result = reflectance + reflectance
    print(f"   reflectance + reflectance: shape={result.shape}")

    # Unit safety context
    print("\n4. Unit Safety Context:")
    with gt.unit_safe(False):
        result = reflectance + reflectance
        print(f"   With unit_safe(False): shape={result.shape}")

    print("\nDone!")


if __name__ == '__main__':
    main()
