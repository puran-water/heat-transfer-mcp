"""
Shared heat exchanger utilities.

This module provides common functions used across heat exchanger sizing and rating tools.
All functions leverage upstream ht/fluids libraries where available.
"""

import logging
import math
from typing import Dict, Optional, Tuple

from utils.import_helpers import HT_AVAILABLE

logger = logging.getLogger("heat-transfer-mcp.hx_common")


def calculate_lmtd_with_check(
    Thi: float,
    Tho: float,
    Tci: float,
    Tco: float,
    counterflow: bool = True
) -> Tuple[float, Optional[str]]:
    """
    Calculate LMTD with temperature crossover check.

    Uses ht.core.LMTD for accurate calculation.

    Args:
        Thi: Hot fluid inlet temperature (K)
        Tho: Hot fluid outlet temperature (K)
        Tci: Cold fluid inlet temperature (K)
        Tco: Cold fluid outlet temperature (K)
        counterflow: True for counterflow, False for parallel flow

    Returns:
        Tuple of (LMTD in K, error_message or None)
    """
    if not HT_AVAILABLE:
        raise ImportError("ht library required for LMTD calculation")

    from ht.core import LMTD

    # Check for temperature crossover
    if counterflow:
        dT1 = Thi - Tco
        dT2 = Tho - Tci
    else:
        dT1 = Thi - Tci
        dT2 = Tho - Tco

    if dT1 <= 0 or dT2 <= 0:
        return 0.0, f"Temperature crossover: dT1={dT1:.2f}K, dT2={dT2:.2f}K"

    try:
        lmtd = LMTD(Thi=Thi, Tho=Tho, Tci=Tci, Tco=Tco, counterflow=counterflow)
        return lmtd, None
    except Exception as e:
        return 0.0, str(e)


def calculate_f_correction(
    Thi: float,
    Tho: float,
    Tci: float,
    Tco: float,
    shells: int = 1
) -> float:
    """
    Calculate LMTD correction factor for multi-pass shell-tube HX.

    Uses ht.hx.F_LMTD_Fakheri for accurate calculation.

    Args:
        Thi: Hot fluid inlet temperature (K)
        Tho: Hot fluid outlet temperature (K)
        Tci: Cold fluid inlet temperature (K)
        Tco: Cold fluid outlet temperature (K)
        shells: Number of shell passes (default 1)

    Returns:
        F correction factor (0 < F <= 1)
    """
    if not HT_AVAILABLE:
        raise ImportError("ht library required for F correction calculation")

    from ht.hx import F_LMTD_Fakheri

    try:
        F = F_LMTD_Fakheri(Tci=Tci, Tco=Tco, Thi=Thi, Tho=Tho, shells=shells)
        return F
    except Exception as e:
        logger.warning(f"F_LMTD_Fakheri failed: {e}, returning 1.0")
        return 1.0


def verify_heat_balance(
    Q_duty: float,
    U: float,
    A: float,
    LMTD: float,
    F: float = 1.0,
    tolerance_pct: float = 5.0
) -> Dict:
    """
    Verify heat balance: Q = U * A * LMTD * F.

    Args:
        Q_duty: Heat duty from energy balance (W)
        U: Overall heat transfer coefficient (W/m²K)
        A: Heat transfer area (m²)
        LMTD: Log mean temperature difference (K)
        F: LMTD correction factor (default 1.0)
        tolerance_pct: Acceptable error percentage (default 5%)

    Returns:
        Dict with verification results
    """
    Q_from_UA = U * A * LMTD * F

    if Q_duty > 0:
        error_pct = abs(Q_from_UA - Q_duty) / Q_duty * 100
    else:
        error_pct = 0.0

    return {
        "Q_duty_W": Q_duty,
        "Q_from_UA_W": Q_from_UA,
        "error_pct": error_pct,
        "balance_satisfied": error_pct <= tolerance_pct,
        "tolerance_pct": tolerance_pct
    }


def format_temperature_output(
    Thi: float,
    Tho: float,
    Tci: float,
    Tco: float
) -> Dict:
    """
    Format temperatures in standard output structure.

    Args:
        Thi: Hot fluid inlet temperature (K)
        Tho: Hot fluid outlet temperature (K)
        Tci: Cold fluid inlet temperature (K)
        Tco: Cold fluid outlet temperature (K)

    Returns:
        Dict with temperatures in K and C
    """
    return {
        "hot_inlet_K": Thi,
        "hot_inlet_C": Thi - 273.15,
        "hot_outlet_K": Tho,
        "hot_outlet_C": Tho - 273.15,
        "cold_inlet_K": Tci,
        "cold_inlet_C": Tci - 273.15,
        "cold_outlet_K": Tco,
        "cold_outlet_C": Tco - 273.15,
        "approach_hot_end_K": Thi - Tco,
        "approach_cold_end_K": Tho - Tci,
    }


def check_temperature_crossover(
    Thi: float,
    Tho: float,
    Tci: float,
    Tco: float,
    counterflow: bool = True
) -> Tuple[bool, str]:
    """
    Check for invalid temperature crossover.

    Args:
        Thi: Hot fluid inlet temperature (K)
        Tho: Hot fluid outlet temperature (K)
        Tci: Cold fluid inlet temperature (K)
        Tco: Cold fluid outlet temperature (K)
        counterflow: True for counterflow, False for parallel flow

    Returns:
        Tuple of (is_valid, message)
    """
    if counterflow:
        dT1 = Thi - Tco  # Hot end
        dT2 = Tho - Tci  # Cold end
    else:
        dT1 = Thi - Tci  # Inlet end
        dT2 = Tho - Tco  # Outlet end

    if dT1 <= 0:
        return False, f"Temperature crossover at {'hot' if counterflow else 'inlet'} end: dT={dT1:.2f}K"
    if dT2 <= 0:
        return False, f"Temperature crossover at {'cold' if counterflow else 'outlet'} end: dT={dT2:.2f}K"

    return True, "Temperature profile valid"


def calculate_annulus_geometry(
    D_outer: float,
    D_inner: float
) -> Dict:
    """
    Calculate annulus geometry for double-pipe heat exchangers.

    Args:
        D_outer: Inner diameter of outer pipe (m)
        D_inner: Outer diameter of inner pipe (m)

    Returns:
        Dict with annulus geometry properties
    """
    if D_outer <= D_inner:
        raise ValueError(f"Outer diameter ({D_outer}m) must be greater than inner diameter ({D_inner}m)")

    # Cross-sectional area of annulus
    A_annulus = math.pi / 4 * (D_outer**2 - D_inner**2)

    # Hydraulic diameter for annulus: Dh = 4*A/P = D_outer - D_inner
    D_hydraulic = D_outer - D_inner

    # Wetted perimeter
    P_wetted = math.pi * (D_outer + D_inner)

    # Equivalent diameter for heat transfer (based on heated perimeter only - inner pipe)
    # De = 4*A / P_heated = (D_outer^2 - D_inner^2) / D_inner
    D_equivalent = (D_outer**2 - D_inner**2) / D_inner

    return {
        "D_outer_m": D_outer,
        "D_inner_m": D_inner,
        "A_annulus_m2": A_annulus,
        "D_hydraulic_m": D_hydraulic,
        "D_equivalent_m": D_equivalent,
        "P_wetted_m": P_wetted,
        "gap_m": (D_outer - D_inner) / 2
    }


def calculate_overall_U(
    h_inner: float,
    h_outer: float,
    D_inner: float,
    D_outer: float,
    wall_thickness: float,
    k_wall: float,
    fouling_inner: float = 0.0,
    fouling_outer: float = 0.0,
    reference: str = "outer"
) -> Dict:
    """
    Calculate overall heat transfer coefficient for cylindrical geometry.

    Uses standard cylindrical wall resistance formula:
    1/U_o = 1/h_o + R_fo + (r_o * ln(r_o/r_i))/k + R_fi*(r_o/r_i) + (r_o/r_i)/h_i

    Args:
        h_inner: Inner convection coefficient (W/m²K)
        h_outer: Outer convection coefficient (W/m²K)
        D_inner: Inner diameter (m)
        D_outer: Outer diameter (m)
        wall_thickness: Wall thickness (m), used to determine tube OD from ID
        k_wall: Wall thermal conductivity (W/m-K)
        fouling_inner: Inner fouling resistance (m²K/W)
        fouling_outer: Outer fouling resistance (m²K/W)
        reference: Reference surface ("inner" or "outer")

    Returns:
        Dict with U value and resistances
    """
    r_i = D_inner / 2
    r_o = D_inner / 2 + wall_thickness

    # Resistances (referenced to outer surface)
    R_conv_inner = r_o / (r_i * h_inner) if h_inner > 0 else float('inf')
    R_fouling_inner = fouling_inner * (r_o / r_i)
    R_wall = (r_o * math.log(r_o / r_i)) / k_wall if r_o > r_i else 0
    R_fouling_outer = fouling_outer
    R_conv_outer = 1 / h_outer if h_outer > 0 else float('inf')

    R_total = R_conv_inner + R_fouling_inner + R_wall + R_fouling_outer + R_conv_outer

    U_outer = 1 / R_total if R_total > 0 else 0

    # Convert to inner reference if needed
    if reference == "inner":
        U_inner = U_outer * (r_o / r_i)
        U = U_inner
    else:
        U = U_outer

    return {
        "U_W_m2K": U,
        "U_outer_W_m2K": U_outer,
        "R_total_m2K_W": R_total,
        "R_conv_inner_m2K_W": R_conv_inner,
        "R_fouling_inner_m2K_W": R_fouling_inner,
        "R_wall_m2K_W": R_wall,
        "R_fouling_outer_m2K_W": R_fouling_outer,
        "R_conv_outer_m2K_W": R_conv_outer,
        "reference_surface": reference
    }


def calculate_effectiveness_ntu(
    C_min: float,
    C_max: float,
    UA: float,
    subtype: str = "counterflow"
) -> Dict:
    """
    Calculate effectiveness and NTU for a heat exchanger.

    Uses ht.hx.effectiveness_from_NTU for accurate calculation.

    Args:
        C_min: Minimum heat capacity rate (W/K)
        C_max: Maximum heat capacity rate (W/K)
        UA: Overall conductance (W/K)
        subtype: Flow arrangement ('counterflow', 'parallel', etc.)

    Returns:
        Dict with effectiveness, NTU, and Cr
    """
    if not HT_AVAILABLE:
        raise ImportError("ht library required for effectiveness calculation")

    from ht.hx import effectiveness_from_NTU

    if C_min <= 0:
        raise ValueError("C_min must be positive")

    Cr = C_min / C_max if C_max > 0 else 0
    NTU = UA / C_min

    effectiveness = effectiveness_from_NTU(NTU, Cr, subtype=subtype)

    return {
        "effectiveness": effectiveness,
        "NTU": NTU,
        "Cr": Cr,
        "C_min_W_K": C_min,
        "C_max_W_K": C_max,
        "UA_W_K": UA,
        "subtype": subtype
    }
