"""Unit registry and unit-safe arithmetic for GeoTensor (§8 of GeoTensor Spec).

Provides :class:`GeoUnitRegistry` wrapping Pint with predefined Earth
observation units, unit conversion, and unit-safe arithmetic operations
on :class:`GeoTensor` objects.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np
import pint
import xarray as xr

from .core import GeoTensor, UnitMismatchError, to_geotensor

# ====================================================================
# Global unit registry (singleton via module-level)
# ====================================================================

_GLOBAL_UNIT_SAFE = True


@contextlib.contextmanager
def unit_safe(enabled: bool = True) -> Iterator[None]:
    """Context manager to enable/disable unit safety checks globally.

    Args:
        enabled: If ``False``, unit checks are disabled inside the block.

    Example::

        with gt.unit_safe(False):
            normalized = (gt - mean) / std
    """
    global _GLOBAL_UNIT_SAFE
    previous = _GLOBAL_UNIT_SAFE
    _GLOBAL_UNIT_SAFE = enabled
    try:
        yield
    finally:
        _GLOBAL_UNIT_SAFE = previous


def unit_safe_enabled() -> bool:
    """Return current global unit safety state."""
    return _GLOBAL_UNIT_SAFE


# ====================================================================
# GeoUnitRegistry
# ====================================================================


class GeoUnitRegistry:
    """Pint-based unit registry with predefined Earth observation units.

    Predefined EO measurement types:

    * ``reflectance`` — dimensionless
    * ``radiance`` — W / m**2 / sr / um
    * ``brightness_temperature`` — K
    * ``backscatter`` — dimensionless (linear) or dB
    * ``DN`` — dimensionless (scaled)
    * ``wavelength`` — nm
    * ``resolution`` — m or degree

    Usage::

        ureg = GeoUnitRegistry()
        ureg('reflectance')  # <Unit('dimensionless')>
        ureg('radiance')     # <Unit('watt / meter ** 2 / steradian / micrometer')>
    """

    _instance: Optional[pint.UnitRegistry] = None

    # Mapping from EO measurement types to Pint unit strings
    EO_UNITS: Dict[str, str] = {
        'reflectance': 'dimensionless',
        'radiance': 'W / m**2 / sr / um',
        'brightness_temperature': 'K',
        'backscatter': 'dimensionless',
        'backscatter_db': 'dB',
        'DN': 'dimensionless',
        'wavelength': 'nm',
        'resolution': 'm',
    }

    # Unit compatibility groups — operations between groups raise errors
    _COMPATIBILITY_GROUPS: Dict[str, Tuple[str, ...]] = {
        'reflectance': ('reflectance', 'DN', 'backscatter'),
        'radiance': ('radiance',),
        'temperature': ('brightness_temperature',),
        'dimensionless': ('reflectance', 'DN', 'backscatter', 'dimensionless'),
    }

    def __new__(cls) -> 'GeoUnitRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance  # type: ignore[return-value]

    def __init__(self) -> None:
        if not hasattr(self, '_registry'):
            self._registry = pint.UnitRegistry()
            # dB is already defined in modern pint; if not, define via string
            try:
                self._registry.parse_units('dB')
            except pint.UndefinedUnitError:
                self._registry.define('dB = deci[bell]')
            try:
                self._registry.parse_units('sr')
            except pint.UndefinedUnitError:
                self._registry.define('sr = rad ** 2 = steradian')

    def __call__(self, measurement_type: str) -> pint.Unit:
        """Return the Pint unit for an EO measurement type.

        Args:
            measurement_type: ``'reflectance'``, ``'radiance'``, etc.

        Returns:
            :class:`pint.Unit` corresponding to the measurement type.
        """
        unit_str = self.EO_UNITS.get(measurement_type, measurement_type)
        return self._registry.parse_units(unit_str)

    @property
    def registry(self) -> pint.UnitRegistry:
        """The underlying :class:`pint.UnitRegistry`."""
        return self._registry

    def band_to_unit_string(self, band_info: Any) -> str:
        """Convert a :class:`BandInfo` measurement type to a Pint unit string.

        Args:
            band_info: A :class:`~.core.BandInfo` instance.

        Returns:
            Pint-compatible unit string.
        """
        mtype = band_info.measurement_type
        return self.EO_UNITS.get(mtype, 'dimensionless')

    def check_compatible(
        self,
        unit_a: str,
        unit_b: str,
    ) -> bool:
        """Check if two EO unit strings are compatible.

        Args:
            unit_a: First EO unit string.
            unit_b: Second EO unit string.

        Returns:
            ``True`` if the units are compatible.
        """
        if not _GLOBAL_UNIT_SAFE:
            return True
        # Check compatibility groups first
        for group_key, members in self._COMPATIBILITY_GROUPS.items():
            if unit_a in members and unit_b in members:
                return True
        try:
            ua = self._registry.parse_units(self.EO_UNITS.get(unit_a, unit_a))
            ub = self._registry.parse_units(self.EO_UNITS.get(unit_b, unit_b))
            return ua.is_compatible_with(ub)
        except Exception:
            return unit_a == unit_b

    def assert_compatible(self, unit_a: str, unit_b: str) -> None:
        """Raise :class:`UnitMismatchError` if units are incompatible.

        Args:
            unit_a: First EO unit string.
            unit_b: Second EO unit string.

        Raises:
            UnitMismatchError: If units are not compatible.
        """
        if not self.check_compatible(unit_a, unit_b):
            raise UnitMismatchError(
                f"Incompatible units: '{unit_a}' vs '{unit_b}'"
            )


# ====================================================================
# Unit conversion utilities
# ====================================================================


_UREG = GeoUnitRegistry()


def convert_units(
    obj: xr.DataArray,
    to: str,
    from_unit: Optional[str] = None,
    band_info: Optional[Dict[str, Any]] = None,
) -> GeoTensor:
    """Convert band values between EO measurement types.

    Applies sensor-specific calibration if applicable.

    Args:
        obj: Input DataArray / GeoTensor.
        to: Target measurement type (e.g. ``'reflectance'``, ``'radiance'``).
        from_unit: Source measurement type. If ``None``, inferred from
                   band semantics.
        band_info: Optional band metadata. If ``None``, inferred from
                   ``obj.attrs.get('band_semantics')``.

    Returns:
        A :class:`GeoTensor` with converted units.
    """
    data = obj.copy(deep=False)
    band_semantics = band_info or data.attrs.get('band_semantics', {})

    if from_unit is None:
        from_unit = 'reflectance'  # default assumption

    target_unit = _UREG.EO_UNITS.get(to, to)

    # Apply per-band conversion
    for band_name in data.band_names if 'band' in data.dims else [None]:
        info = band_semantics.get(band_name) if band_name else None
        if info is not None:
            src_unit = _UREG.band_to_unit_string(info)
        else:
            src_unit = _UREG.EO_UNITS.get(from_unit, from_unit)

        if src_unit == target_unit:
            continue

        # Apply calibration coefficients if available
        if info and info.calibration_coeffs:
            coeffs = info.calibration_coeffs
            gain = coeffs.get('gain', coeffs.get('reflectance_gain', 1.0))
            offset = coeffs.get('offset', coeffs.get('reflectance_offset', 0.0))
            if band_name:
                band_data = data.sel(band=band_name)
                data = data.where(data.band != band_name, band_data * gain + offset)
            else:
                data = data * gain + offset

        # Update band semantics to reflect new unit
        if info and band_semantics:
            import copy
            updated_info = copy.copy(info)
            updated_info.measurement_type = to
            updated_info.unit = target_unit
            band_semantics[band_name] = updated_info

    if band_semantics:
        data.attrs['band_semantics'] = band_semantics
    data.attrs['unit'] = target_unit

    return to_geotensor(data, obj)


def band_units(
    obj: xr.DataArray,
) -> Dict[str, str]:
    """Return mapping of band names to Pint-compatible unit strings.

    Args:
        obj: Input DataArray / GeoTensor.

    Returns:
        ``dict[str, str]`` mapping band names to unit strings.
    """
    band_semantics = obj.attrs.get('band_semantics', {})
    if band_semantics:
        return {
            name: _UREG.band_to_unit_string(info)
            for name, info in band_semantics.items()
        }

    # Fallback: try to get a global unit from attrs
    unit = obj.attrs.get('unit', 'dimensionless')
    if 'band' in obj.dims:
        return {str(b): unit for b in obj.coords['band'].values}
    return {}


# ====================================================================
# Unit-safe arithmetic wrappers
# ====================================================================


def unit_safe_add(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """Add with unit compatibility check."""
    _check_unit_compatibility('add', a, b)
    return a + b


def unit_safe_sub(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """Subtract with unit compatibility check."""
    _check_unit_compatibility('sub', a, b)
    return a - b


def unit_safe_mul(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """Multiply with unit propagation."""
    return a * b


def unit_safe_div(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """Divide with unit propagation."""
    return a / b


def _check_unit_compatibility(op: str, a: xr.DataArray, b: xr.DataArray) -> None:
    """Check unit compatibility for +/- operations."""
    if not _GLOBAL_UNIT_SAFE:
        return

    units_a = band_units(a)
    units_b = band_units(b)

    if not units_a or not units_b:
        return  # No unit info, skip check

    common_bands = set(units_a.keys()) & set(units_b.keys())
    for band in common_bands:
        if not _UREG.check_compatible(units_a[band], units_b[band]):
            raise UnitMismatchError(
                f"Cannot {op}: band '{band}' has incompatible units "
                f"'{units_a[band]}' vs '{units_b[band]}'. "
                "Use gt.unit_safe(False) to override or convert_units() first."
            )