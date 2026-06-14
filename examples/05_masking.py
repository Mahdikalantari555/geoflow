"""Example 05: Masking

Demonstrates mask creation, combination, and filling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import geoflow as gt


def main():
    # Create a GeoTensor with some invalid data
    data = np.random.rand(4, 32, 32).astype(np.float32)
    # Set some pixels to NaN (simulating nodata)
    data[0, 5:10, 5:10] = np.nan
    data[1, 15:20, 15:20] = np.nan

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

    print("=== Mask Operations ===")

    # Access mask
    mask = tensor.geotensor.mask
    print(f"Mask layers: {mask.layer_names}")

    # Create mask from condition
    print("\nCreating mask from condition...")
    masked = tensor.geotensor.where(tensor > 0.5, drop=False)
    print(f"Masked tensor shape: {masked.shape}")

    # Apply mask (set to NaN)
    print("\nApplying mask...")
    applied = tensor.geotensor.apply_mask()
    print(f"Applied shape: {applied.shape}")

    # Fill with constant
    print("\nFilling with constant...")
    filled = tensor.geotensor.fill(value=0.0)
    nan_count = np.isnan(filled.values).sum()
    print(f"NaN count after fill: {nan_count}")

    # Trim masked regions
    print("\nTrimming masked regions...")
    trimmed = tensor.geotensor.trim_masked(margin=2)
    print(f"Trimmed shape: {trimmed.shape}")

    print("\nDone!")


if __name__ == '__main__':
    main()
