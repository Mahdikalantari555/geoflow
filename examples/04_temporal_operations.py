"""Example 04: Temporal Operations

Demonstrates time handling, resampling, and aggregation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import geoflow as gt


def main():
    # Create a time-series GeoTensor
    n_times = 12
    data = np.random.rand(n_times, 4, 32, 32).astype(np.float32)
    time_coords = pd.date_range('2023-01-01', periods=n_times, freq='MS')

    tensor = gt.GeoTensor(
        data,
        coords={
            'time': time_coords,
            'x': np.linspace(500000, 500320, 32),
            'y': np.linspace(4000000, 4000320, 32),
            'band': ['B02', 'B03', 'B04', 'B08'],
        },
        dims=['time', 'band', 'y', 'x'],
        crs='EPSG:32632',
    )

    print("=== Time Properties ===")
    print(f"Time interval: {tensor.geotensor.time_interval}")
    print(f"Time metadata: {tensor.geotensor.time_metadata}")

    # Temporal window
    print("\n=== Temporal Window ===")
    windowed = tensor.geotensor.temporal_window(
        start='2023-03-01',
        end='2023-09-30',
    )
    print(f"Windowed shape: {windowed.shape}")

    # Resample time
    print("\n=== Resample Time ===")
    resampled = tensor.geotensor.resample_time(rule='QS', method='nearest')
    print(f"Resampled shape: {resampled.shape}")

    # Temporal aggregation
    print("\n=== Temporal Aggregation ===")
    monthly_mean = tensor.geotensor.temporal_aggregate(period='M', func='mean')
    print(f"Monthly mean shape: {monthly_mean.shape}")

    # Event-based sampling
    print("\n=== Event-Based Sampling ===")
    events = tensor.geotensor.select_by_event(
        event_times=['2023-02-15', '2023-06-01', '2023-10-01'],
        tolerance='15D',
    )
    print(f"Events shape: {events.shape}")

    # Relative time
    print("\n=== Relative Time ===")
    relative = tensor.geotensor.time_to_relative()
    print(f"Relative time coords: {relative.coords['time'].values[:3]}")

    print("\nDone!")


if __name__ == '__main__':
    main()
