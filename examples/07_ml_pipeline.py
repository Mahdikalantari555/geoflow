"""Example 07: ML Pipeline

Full ML pipeline: create data → normalize → create DataLoader.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import geoflow as gt


def main():
    print("=== ML Pipeline ===")

    # Create synthetic training data
    print("\n1. Creating training data...")
    data = np.random.rand(10, 4, 64, 64).astype(np.float32)
    tensor = gt.GeoTensor(
        data,
        coords={
            'time': range(10),
            'x': np.linspace(500000, 500640, 64),
            'y': np.linspace(4000000, 4000640, 64),
            'band': ['B02', 'B03', 'B04', 'B08'],
        },
        dims=['time', 'band', 'y', 'x'],
        crs='EPSG:32632',
    )
    print(f"   Shape: {tensor.shape}")

    # Compute normalization
    print("\n2. Computing normalization...")
    norm_params = tensor.geotensor.compute_normalization(
        method='zscore',
        dim=['x', 'y'],
    )
    print(f"   Mean shape: {norm_params['mean'].shape}")
    print(f"   Std shape: {norm_params['std'].shape}")

    # Apply normalization
    print("\n3. Applying normalization...")
    normalized = tensor.geotensor.normalize(norm_params)
    print(f"   Normalized mean: {normalized.values.mean():.4f}")
    print(f"   Normalized std: {normalized.values.std():.4f}")

    # Create DataLoader
    print("\n4. Creating DataLoader...")
    loader = normalized.geotensor.to_dataloader(
        batch_size=3,
        shuffle=True,
        num_workers=0,
        batch_dim='time',
    )
    print(f"   DataLoader batches: {len(loader)}")

    # Iterate
    print("\n5. Iterating...")
    for i, batch in enumerate(loader):
        print(f"   Batch {i}: shape={batch.shape}")
        if i >= 2:
            break

    # Denormalize
    print("\n6. Denormalizing...")
    original = normalized.geotensor.denormalize()
    print(f"   Denormalized mean: {original.values.mean():.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
