"""Example 06: Patch Sampling

Demonstrates different patch sampling strategies for ML.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import geoflow as gt


def main():
    # Create a larger GeoTensor for sampling
    data = np.random.rand(4, 128, 128).astype(np.float32)

    tensor = gt.GeoTensor(
        data,
        coords={
            'x': np.linspace(500000, 501280, 128),
            'y': np.linspace(4000000, 4001280, 128),
            'band': ['B02', 'B03', 'B04', 'B08'],
        },
        dims=['band', 'y', 'x'],
        crs='EPSG:32632',
    )

    print("=== Patch Sampling ===")

    # Tile sampling (sliding window)
    print("\n1. Tile Sampling:")
    tiles = tensor.geotensor.sample_tiles(
        size=32,
        stride=16,
        mask_threshold=0.5,
    )
    print(f"   Number of tiles: {len(tiles)}")
    if tiles:
        print(f"   Tile shape: {tiles[0]['data'].shape}")

    # Grid sampling
    print("\n2. Grid Sampling:")
    grid_patches = tensor.geotensor.sample_grid(
        nx=4, ny=4,
        size=32,
    )
    print(f"   Number of grid patches: {len(grid_patches)}")

    # Random sampling
    print("\n3. Random Sampling:")
    random_patches = tensor.geotensor.sample_random(
        num_patches=10,
        size=32,
    )
    print(f"   Number of random patches: {len(random_patches)}")

    # Patch dataset for PyTorch
    print("\n4. Patch Dataset:")
    dataset = tensor.geotensor.to_patch_dataset(
        patch_size=32,
        stride=16,
        augment=True,
        shuffle=True,
    )
    print(f"   Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"   Sample shape: {sample.shape}")

    print("\nDone!")


if __name__ == '__main__':
    main()
