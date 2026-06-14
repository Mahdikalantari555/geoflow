"""Example 01: Basic GeoTensor Operations

Creates a GeoTensor from scratch and inspects its properties.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import geoflow as gt


def main():
    # Create synthetic data
    data = np.random.rand(4, 100, 100).astype(np.float32)
    x_coords = np.linspace(500000, 501000, 100)
    y_coords = np.linspace(4000000, 4001000, 100)
    band_names = ['B02', 'B03', 'B04', 'B08']

    # Create GeoTensor
    tensor = gt.GeoTensor(
        data,
        coords={
            'x': x_coords,
            'y': y_coords,
            'band': band_names,
        },
        dims=['band', 'y', 'x'],
        crs='EPSG:32632',
        band_info={
            'B02': gt.BandInfo('B02', wavelength_nm=490.0),
            'B03': gt.BandInfo('B03', wavelength_nm=560.0),
            'B04': gt.BandInfo('B04', wavelength_nm=665.0),
            'B08': gt.BandInfo('B08', wavelength_nm=842.0),
        },
    )

    # Inspect properties
    print("=== GeoTensor Properties ===")
    print(f"Shape: {tensor.shape}")
    print(f"Dims: {tensor.dims}")
    print(f"CRS: {tensor.geotensor.crs}")
    print(f"Resolution: {tensor.geotensor.resolution}")
    print(f"Bounds: {tensor.geotensor.bounds}")

    # Select by coordinates
    pixel = tensor.geotensor.latlon(lat=40.5, lon=-9.5)
    print(f"\nPixel at (40.5, -9.5): {pixel.shape}")

    # Slice by bounding box
    subset = tensor.geotensor.slice_bbox(
        west=-9.8, south=40.2,
        east=-9.2, north=40.8,
        crs='EPSG:4326',
    )
    print(f"Subset shape: {subset.shape}")

    print("\nDone!")


if __name__ == '__main__':
    main()
