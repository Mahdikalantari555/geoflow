"""Example 02: Open and Write Operations

Demonstrates opening data from different formats and writing results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import geoflow as gt


def main():
    # Create synthetic data to write
    data = np.random.rand(4, 64, 64).astype(np.float32)
    x_coords = np.linspace(500000, 500640, 64)
    y_coords = np.linspace(4000000, 4000640, 64)

    tensor = gt.GeoTensor(
        data,
        coords={
            'x': x_coords,
            'y': y_coords,
            'band': ['B02', 'B03', 'B04', 'B08'],
        },
        dims=['band', 'y', 'x'],
        crs='EPSG:32632',
    )

    # Write to Zarr
    print("Writing to Zarr...")
    zarr_path = 'temp_output.zarr'
    tensor.geotensor.to_zarr(zarr_path, compute=True)
    print(f"Written to {zarr_path}")

    # Read back from Zarr
    print("\nReading from Zarr...")
    loaded = gt.open_zarr(zarr_path)
    print(f"Loaded shape: {loaded.shape}")

    # Write to COG
    print("\nWriting to COG...")
    cog_path = 'temp_output.tif'
    tensor.geotensor.to_cog(cog_path, dtype='float32')
    print(f"Written to {cog_path}")

    # Lazy computation
    print("\nLazy computation...")
    result = tensor + 1.0  # Returns GeoTensor with Dask graph
    print(f"Result type: {type(result)}")

    # Compute to memory
    arr = tensor.geotensor.compute()
    print(f"Computed shape: {arr.shape}")

    # Cleanup
    import shutil
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    if os.path.exists(cog_path):
        os.remove(cog_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
