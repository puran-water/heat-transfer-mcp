"""Mixed free/forced convection blending."""

from __future__ import annotations

from heat_transfer_mcp.convection.single_phase import validate_positive_finite


def blend_nusselt_numbers(nu_forced: float, nu_free: float, *, n: float = 3.0) -> float:
    """Blend forced and natural Nusselt numbers without a regime discontinuity."""
    nu_forced = validate_positive_finite(nu_forced, "nu_forced")
    nu_free = validate_positive_finite(nu_free, "nu_free")
    n = validate_positive_finite(n, "n")
    blended = (nu_forced**n + nu_free**n) ** (1.0 / n)
    return max(blended, nu_forced, nu_free)


def blend_convection_coefficients(h_forced: float, h_free: float, *, n: float = 3.0) -> float:
    """Blend forced and natural heat-transfer coefficients."""
    h_forced = validate_positive_finite(h_forced, "h_forced")
    h_free = validate_positive_finite(h_free, "h_free")
    n = validate_positive_finite(n, "n")
    blended = (h_forced**n + h_free**n) ** (1.0 / n)
    return max(blended, h_forced, h_free)
