"""Single-phase convection validation utilities."""

from __future__ import annotations

import math
from typing import Optional

from heat_transfer_mcp.errors import CalculationError


def validate_positive_finite(value: Optional[float], field: str) -> float:
    """Return a finite positive float or raise a structured calculation error."""
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CalculationError(
            code="invalid_convection_input",
            message=f"{field} must be a positive finite value",
            field=field,
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise CalculationError(
            code="invalid_convection_input",
            message=f"{field} must be a positive finite value",
            field=field,
        )
    return numeric


def require_prandtl_number(
    prandtl_number: Optional[float],
    specific_heat_cp: Optional[float],
    dynamic_viscosity: Optional[float],
    thermal_conductivity: Optional[float],
) -> float:
    """Require explicit Pr or derive it from Cp, viscosity, and conductivity.

    Correlations must not receive ``None`` for Prandtl number; doing so can
    produce late TypeErrors or NaN-like downstream behavior. This helper keeps
    the failure close to the missing property data.
    """
    if prandtl_number is not None:
        return validate_positive_finite(prandtl_number, "prandtl_number")

    missing = [
        name
        for name, value in (
            ("specific_heat_cp", specific_heat_cp),
            ("dynamic_viscosity", dynamic_viscosity),
            ("thermal_conductivity", thermal_conductivity),
        )
        if value is None
    ]
    if missing:
        raise CalculationError(
            code="missing_prandtl_number",
            message=(
                "Prandtl number is required or must be derivable from "
                "specific_heat_cp, dynamic_viscosity, and thermal_conductivity"
            ),
            field="prandtl_number",
            details={"missing": missing},
        )

    cp = validate_positive_finite(specific_heat_cp, "specific_heat_cp")
    mu = validate_positive_finite(dynamic_viscosity, "dynamic_viscosity")
    k = validate_positive_finite(thermal_conductivity, "thermal_conductivity")
    return mu * cp / k
