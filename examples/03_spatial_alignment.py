"""Example 03: Spatial Alignment

Aligns two GeoTensors with different CRS and resolution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import geoflow as gt


def main():
    # Create two tensors with different CRS and resolution
    # Tensor 1: UTM zone 32N, 10m resolution
    data1 = np.random.rand(3, 50, 50).astype(np.float32)
    tensor1 = gt.GeoTensor(
        data1,
        coords={
            'x': np.linspace(500000, 500500, 50),
            'y': np.linspace(4000000, 4000500, 50),
            'band': ['B04', 'B08', 'NDVI'],
        },
        dims=['band', 'y', 'x'],
        crs='EPSG:32632',
    )

    # Tensor 2: UTM zone 33N, 20m resolution
    data2 = np.random.rand(3, 30, 30).astype(np.float32)
    tensor2 = gt.GeoTensor(
        data2,
        coords={
            'x': np.linspace(600000, 600600, 30),
            'y': np.linspace(5000000, 5000600, 30),
            'band': ['B04', 'B08', 'NDVI'],
        },
        dims=['band', 'y', 'x'],
        crs='EPSG:32633',
    )

    print("=== Before Alignment ===")
    print(f"Tensor1: CRS={tensor1.geotensor.crs}, Res={tensor1.geotensor.resolution}")
    print(f"Tensor2: CRS={tensor2.geotensor.crs}, Res={tensor2.geotensor.resolution}")

    # Align to common grid
    print("\nAligning...")
    aligned1, aligned2 = gt.align(
        tensor1, tensor2,
        spec=gt.AlignmentSpec(
            spatial=gt.SpatialAlignment(
                crs='EPSG:32632',
                resolution=20.0,
                method='bilinear',
                extent='intersection',
            ),
        )
    )

    print("\n=== After Alignment ===")
    print(f"Aligned1: CRS={aligned1.geotensor.crs}, Res={aligned1.geotensor.resolution}")
    print(f"Aligned2: CRS={aligned2.geotensor.crs}, Res={aligned2.geotensor.resolution}")

    # Now safe to combine
    combined = aligned1 + aligned2
    print(f"\nCombined shape: {combined.shape}")

    # Resample within same CRS
    print("\nResampling to 50m...")
    resampled = aligned1.geotensor.resample(resolution=50.0)
    print(f"Resampled shape: {resampled.shape}, resolution: {resampled.geotensor.resolution}")

    print("\nDone!")


if __name__ == '__main__':
    main()
