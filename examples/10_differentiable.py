"""Example 10: Differentiable Operations

Demonstrates gradient-aware operations and the escape hatch.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import geoflow as gt


def main():
    print("=== Differentiable Operations ===")

    # Create a GeoTensor
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

    # Convert to PyTorch with gradients
    print("\n1. Convert to PyTorch with gradients:")
    torch_tensor = tensor.geotensor.to_pytorch(require_grad=True)
    print(f"   Shape: {torch_tensor.shape}")
    print(f"   Requires grad: {torch_tensor.requires_grad}")

    # Simple gradient computation
    print("\n2. Gradient computation:")
    output = torch_tensor.sin().mean()
    output.backward()
    print(f"   Gradient shape: {torch_tensor.grad.shape}")
    print(f"   Gradient mean: {torch_tensor.grad.mean():.4f}")

    # Differentiable resampling
    print("\n3. Differentiable resampling:")
    resampled = tensor.geotensor.differentiable_resample(
        target_shape=(16, 16),
        method='bilinear',
    )
    print(f"   Resampled shape: {resampled.shape}")

    # Detach from computation graph
    print("\n4. Detach from computation graph:")
    detached = tensor.geotensor.detach()
    torch_detached = detached.geotensor.to_pytorch()
    print(f"   Detached requires grad: {torch_detached.requires_grad}")

    # Reproject inside gradient tape (should raise error)
    print("\n5. Reproject inside gradient tape:")
    try:
        torch.enable_grad()
        tensor.geotensor.reproject('EPSG:4326')
    except gt.NonDifferentiableError as e:
        print(f"   Error caught: {type(e).__name__}")
    finally:
        torch.enable_grad(False)

    print("\nDone!")


if __name__ == '__main__':
    main()
