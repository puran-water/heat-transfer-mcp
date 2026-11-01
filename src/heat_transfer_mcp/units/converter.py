"""Unit parsing and conversion for heat-transfer schemas."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pint import UnitRegistry

ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
Q_ = ureg.Quantity


TARGET_UNITS: Dict[str, str] = {
    "dimensionless": "",
    "temperature_abs": "kelvin",
    "temperature_delta": "kelvin",
    "htc": "watt/meter**2/kelvin",
    "thermal_conductivity": "watt/meter/kelvin",
    "fouling": "meter**2*kelvin/watt",
    "length": "meter",
    "area": "meter**2",
    "volume": "meter**3",
    "velocity": "meter/second",
    "pressure": "pascal",
    "flow_rate_mass": "kilogram/second",
    "flow_rate_volume": "meter**3/second",
    "mass_flux": "kilogram/meter**2/second",
    "power": "watt",
    "linear_power": "watt/meter",
    "energy": "joule",
    "specific_energy": "joule/kilogram",
    "density": "kilogram/meter**3",
    "viscosity_dynamic": "pascal*second",
    "specific_heat": "joule/kilogram/kelvin",
    "roughness": "meter",
    "irradiance": "watt/meter**2",
    "angle": "degree",
}


@dataclass(frozen=True)
class Quantity:
    value: float
    kind: str
    unit: str


_VALUE_UNIT_RE = re.compile(r"^\s*(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?P<unit>.*)$")


def convert_quantity(value: Any, kind: str, *, density_kg_m3: Optional[float] = None, target_unit: Optional[str] = None) -> float:
    """Convert a numeric or string quantity to the SI unit for `kind`."""
    if kind == "dimensionless":
        return _ensure_finite(float(value), kind)
    if isinstance(value, (int, float)):
        return _ensure_finite(float(value), kind)
    if not isinstance(value, str):
        raise TypeError(f"Cannot convert {type(value).__name__} as {kind}")

    magnitude, unit_expr = _split_value_unit(value)
    if not unit_expr:
        return _ensure_finite(float(magnitude), kind)

    normalized = normalize_unit_expression(unit_expr, kind)
    if kind == "flow_rate_volume" and density_kg_m3 is not None:
        volume_rate = Q_(magnitude, normalized).to(TARGET_UNITS["flow_rate_volume"]).magnitude
        return _ensure_finite(float(volume_rate * density_kg_m3), kind)

    target = normalize_unit_expression(target_unit, kind) if target_unit else TARGET_UNITS[kind]
    if not target:
        return _ensure_finite(float(Q_(magnitude, normalized).to_base_units().magnitude), kind)
    return _ensure_finite(float(Q_(magnitude, normalized).to(target).magnitude), kind)


def normalize_unit_expression(unit_expr: str, kind: str) -> str:
    """Normalize heat-transfer unit spellings before Pint parses them.

    Bare F/C are temperature markers only for temperature quantities and
    per-temperature-gradient properties: htc, thermal conductivity, fouling,
    and specific heat. Other kinds leave F/C to Pint so Farad/Coulomb remain
    available where they are dimensionally meaningful.
    """
    unit = unit_expr.strip()
    unit = unit.replace("·", "*").replace("−", "-")
    unit = unit.replace("²", "^2").replace("³", "^3")
    unit = unit.replace("°F", "degF").replace("℉", "degF")
    unit = unit.replace("°C", "degC").replace("℃", "degC")
    unit = re.sub(r"\bBTU\b", "Btu", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bBtu/hr\b", "Btu/hour", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bhr\b", "hour", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bsec\b", "second", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bft\^?2\b", "foot**2", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bft2\b", "foot**2", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bft\b", "foot", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bm\^?2\b", "meter**2", unit)
    unit = re.sub(r"\bm2\b", "meter**2", unit)

    if kind == "temperature_delta":
        unit = _normalize_temperature_marker(unit, delta=True)
    elif kind == "temperature_abs":
        unit = _normalize_temperature_marker(unit, delta=False)
    elif kind in {"htc", "thermal_conductivity", "specific_heat"}:
        unit = _normalize_compound_heat_unit(unit)
        unit = re.sub(r"-K\b", "/kelvin", unit)
        unit = re.sub(r"-delta_degF\b", "/delta_degF", unit)
        unit = re.sub(r"-delta_degC\b", "/delta_degC", unit)
    elif kind == "fouling":
        unit = _normalize_fouling_unit(unit)

    unit = _normalize_btu_dash_expression(unit, kind)
    return unit


def _split_value_unit(value: str) -> Tuple[float, str]:
    match = _VALUE_UNIT_RE.match(value)
    if not match:
        raise ValueError(f"Could not parse quantity: {value!r}")
    magnitude = float(match.group("value"))
    _ensure_finite(magnitude, "quantity magnitude")
    return magnitude, match.group("unit").strip()


def _ensure_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite value")
    return value


def _normalize_temperature_marker(unit: str, *, delta: bool) -> str:
    fahrenheit = "delta_degF" if delta else "degF"
    celsius = "delta_degC" if delta else "degC"
    unit = re.sub(r"\bdelta[_\s-]?F\b", "delta_degF", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bdelta[_\s-]?C\b", "delta_degC", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bdegF\b", fahrenheit, unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bdegC\b", celsius, unit, flags=re.IGNORECASE)
    unit = re.sub(r"(?<![A-Za-z_])F(?![A-Za-z_])", fahrenheit, unit)
    unit = re.sub(r"(?<![A-Za-z_])C(?![A-Za-z_])", celsius, unit)
    return unit


def _normalize_compound_heat_unit(unit: str) -> str:
    # In heat transfer coefficients and conductivities, a trailing F/C is a
    # temperature difference, not an absolute temperature and not farad/coulomb.
    unit = re.sub(r"\bdegF\b", "delta_degF", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bdegC\b", "delta_degC", unit, flags=re.IGNORECASE)
    unit = re.sub(r"(?<![A-Za-z_])F(?![A-Za-z_])", "delta_degF", unit)
    unit = re.sub(r"(?<![A-Za-z_])C(?![A-Za-z_])", "delta_degC", unit)
    return unit


def _normalize_fouling_unit(unit: str) -> str:
    unit = re.sub(r"\bdegF\b", "delta_degF", unit, flags=re.IGNORECASE)
    unit = re.sub(r"\bdegC\b", "delta_degC", unit, flags=re.IGNORECASE)
    unit = re.sub(r"(?<![A-Za-z_])F(?![A-Za-z_])", "delta_degF", unit)
    unit = re.sub(r"(?<![A-Za-z_])C(?![A-Za-z_])", "delta_degC", unit)
    return unit


def _normalize_btu_dash_expression(unit: str, kind: str) -> str:
    compact = unit.replace(" ", "")
    if kind == "htc":
        if re.fullmatch(r"Btu/hour[-/]foot\*\*2[-/]delta_degF", compact, flags=re.IGNORECASE):
            return "Btu/hour/foot**2/delta_degF"
        if re.fullmatch(r"Btu/hour[-/]foot\*\*2[-/]delta_degC", compact, flags=re.IGNORECASE):
            return "Btu/hour/foot**2/delta_degC"
    if kind == "thermal_conductivity":
        if re.fullmatch(r"Btu/hour[-/]foot[-/]delta_degF", compact, flags=re.IGNORECASE):
            return "Btu/hour/foot/delta_degF"
        if re.fullmatch(r"Btu/hour[-/]foot[-/]delta_degC", compact, flags=re.IGNORECASE):
            return "Btu/hour/foot/delta_degC"
    if kind == "specific_heat":
        if re.fullmatch(r"Btu[-/]lb[-/]delta_degF", compact, flags=re.IGNORECASE):
            return "Btu/lb/delta_degF"
        if re.fullmatch(r"Btu[-/]lb[-/]delta_degC", compact, flags=re.IGNORECASE):
            return "Btu/lb/delta_degC"
    if kind == "fouling":
        if re.fullmatch(r"hour[-*]foot\*\*2[-*]delta_degF/Btu", compact, flags=re.IGNORECASE):
            return "hour*foot**2*delta_degF/Btu"
    return unit
