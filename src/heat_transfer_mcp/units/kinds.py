"""Quantity kind constants used by schema metadata."""

from __future__ import annotations

from typing import Literal

QuantityKind = Literal[
    "dimensionless",
    "temperature_abs",
    "temperature_delta",
    "htc",
    "thermal_conductivity",
    "fouling",
    "length",
    "area",
    "volume",
    "velocity",
    "pressure",
    "flow_rate_mass",
    "flow_rate_volume",
    "mass_flux",
    "power",
    "linear_power",
    "energy",
    "specific_energy",
    "density",
    "viscosity_dynamic",
    "specific_heat",
    "roughness",
    "irradiance",
    "angle",
]
