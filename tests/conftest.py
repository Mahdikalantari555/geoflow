"""Conftest for geoflow tests — ensures package is importable."""
import sys
import os

# Add GeoWebLab/ to sys.path so 'import geoflow' resolves to geoflow/__init__.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
