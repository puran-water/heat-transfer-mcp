"""
Helper functions for Heat Transfer MCP server.

This module provides shared utility functions used by multiple tools in the server.
"""

import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple
from utils.constants import STEFAN_BOLTZMANN, DEG_C_to_K
from utils.import_helpers import HT_AVAILABLE

logger = logging.getLogger("heat-transfer-mcp.helpers")

def calculate_reynolds_number(
    velocity: float,
    characteristic_length: float,
    fluid_density: float,
    fluid_viscosity: float
) -> float:
    """
    Calculate Reynolds number for a flow.
    
    Args:
        velocity: Fluid velocity in m/s
        characteristic_length: Characteristic length/diameter in m
        fluid_density: Fluid density in kg/m³
        fluid_viscosity: Fluid dynamic viscosity in Pa·s
        
    Returns:
        Reynolds number (dimensionless)
    """
    if fluid_viscosity <= 0:
        raise ValueError("Fluid viscosity must be positive")
    
    return (fluid_density * velocity * characteristic_length) / fluid_viscosity

def calculate_prandtl_number(
    fluid_viscosity: float,
    fluid_specific_heat: float,
    fluid_thermal_conductivity: float
) -> float:
    """
    Calculate Prandtl number for a fluid.
    
    Args:
        fluid_viscosity: Fluid dynamic viscosity in Pa·s
        fluid_specific_heat: Fluid specific heat capacity in J/(kg·K)
        fluid_thermal_conductivity: Fluid thermal conductivity in W/(m·K)
        
    Returns:
        Prandtl number (dimensionless)
    """
    if fluid_thermal_conductivity <= 0:
        raise ValueError("Fluid thermal conductivity must be positive")
    
    return (fluid_viscosity * fluid_specific_heat) / fluid_thermal_conductivity

def calculate_nusselt_number_external_flow(
    reynolds_number: float,
    prandtl_number: float,
    geometry: str = "flat_plate",
    strict: bool = True
) -> float:
    """
    Calculate Nusselt number for external flow over various geometries.
    Uses ht library functions where available; fails loudly if unavailable.

    Args:
        reynolds_number: Reynolds number (dimensionless)
        prandtl_number: Prandtl number (dimensionless)
        geometry: Geometry type ("flat_plate", "cylinder", "sphere")
        strict: If True (default), require ht library for flat_plate/cylinder

    Returns:
        Nusselt number (dimensionless)

    Raises:
        ImportError: If ht library required but unavailable (strict mode)
        ValueError: If geometry not recognized
    """
    geometry_lower = geometry.lower()

    # For flat_plate and cylinder: REQUIRE ht library functions
    if geometry_lower == "flat_plate" or "flat_plate_external" in geometry_lower:
        if not HT_AVAILABLE:
            if strict:
                raise ImportError("ht library required for flat_plate external convection")
            # Non-strict mode: use standard flat plate correlation
            if reynolds_number < 5e5:  # Laminar
                return 0.664 * math.sqrt(reynolds_number) * prandtl_number**(1/3)
            else:
                return 0.037 * reynolds_number**0.8 * prandtl_number**(1/3)
        from ht.conv_external import Nu_external_horizontal_plate
        return Nu_external_horizontal_plate(reynolds_number, prandtl_number)

    if "cylinder" in geometry_lower:
        if not HT_AVAILABLE:
            if strict:
                raise ImportError("ht library required for cylinder external convection")
            # Non-strict: Churchill-Bernstein correlation
            return 0.3 + ((0.62 * math.sqrt(reynolds_number) * prandtl_number**(1/3) *
                          (1 + (reynolds_number/282000)**(5/8))**(4/5)) /
                          (1 + (0.4/prandtl_number)**(2/3))**(1/4))
        from ht.conv_external import Nu_external_cylinder
        return Nu_external_cylinder(reynolds_number, prandtl_number)

    if "sphere" in geometry_lower:
        # Whitaker correlation for forced convection over sphere
        # Standard textbook correlation - no ht equivalent exists
        # Nu = 2 + (0.4*Re^0.5 + 0.06*Re^(2/3)) * Pr^0.4 * (mu/mu_s)^0.25
        # Simplified form without viscosity ratio correction:
        return 2 + (0.4 * math.sqrt(reynolds_number) + 0.06 * reynolds_number**(2/3)) * prandtl_number**0.4

    raise ValueError(f"Geometry '{geometry}' not recognized for external Nusselt calculation")

def calculate_radiation_heat_transfer(
    emissivity: float,
    area: float,
    temperature_1: float,
    temperature_2: float
) -> float:
    """
    Calculate radiation heat transfer between two surfaces.
    
    Args:
        emissivity: Surface emissivity (dimensionless)
        area: Surface area in m²
        temperature_1: Temperature of first surface in K
        temperature_2: Temperature of second surface in K
        
    Returns:
        Radiation heat transfer rate in W
    """
    return emissivity * STEFAN_BOLTZMANN * area * (temperature_1**4 - temperature_2**4)

def estimate_sky_temperature(
    ambient_temperature: float,
    dew_point: Optional[float] = None,
    cloud_cover: Optional[float] = None
) -> float:
    """
    Estimate effective sky temperature for radiation calculations.
    
    Args:
        ambient_temperature: Ambient air temperature in K
        dew_point: Dew point temperature in K (optional)
        cloud_cover: Cloud cover fraction 0-1 (optional)
        
    Returns:
        Effective sky temperature in K
    """
    T_air = ambient_temperature
    
    if dew_point is None:
        # Swinbank correlation: compute emissivity first
        eps_clear = 9.37e-6 * (T_air ** 2)  # Clear sky emissivity
    else:
        # Better formula with dew point
        Td_C = dew_point - DEG_C_to_K if dew_point > 100 else dew_point
        eps_clear = 0.711 + 0.0056*Td_C + 0.000073*(Td_C**2)
        eps_clear = max(0.1, min(1.0, eps_clear))  # Clamp to valid range
    
    # CRITICAL FIX: Use emissivity^0.25, NOT direct multiplication
    T_sky = T_air * (eps_clear ** 0.25)  # This is the correct formula
    
    # Cloud cover adjustment if provided
    if cloud_cover is not None:
        eps_clouds = eps_clear * (1 + 0.17 * (cloud_cover ** 2))
        eps_clouds = min(1.0, eps_clouds)
        T_sky = T_air * (eps_clouds ** 0.25)
    
    return T_sky

def calculate_lmtd(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """
    Calculate Log Mean Temperature Difference (LMTD) for heat exchangers.

    Uses ht.core.LMTD from the CalebBell/ht library for accurate calculation.

    Args:
        t_hot_in: Hot fluid inlet temperature in K
        t_hot_out: Hot fluid outlet temperature in K
        t_cold_in: Cold fluid inlet temperature in K
        t_cold_out: Cold fluid outlet temperature in K
        flow_arrangement: Flow arrangement ("counterflow", "parallelflow")

    Returns:
        Log Mean Temperature Difference in K

    Raises:
        ImportError: If ht library is not available
        ValueError: If temperature crossover occurs (invalid LMTD)
    """
    if not HT_AVAILABLE:
        raise ImportError("ht library required for LMTD calculation")

    from ht.core import LMTD

    flow_lower = flow_arrangement.lower()

    # Map flow arrangement to ht.core.LMTD counterflow parameter
    if flow_lower == "counterflow":
        counterflow = True
    elif flow_lower in ("parallelflow", "parallel", "cocurrent"):
        counterflow = False
    else:
        logger.warning(f"Flow arrangement '{flow_arrangement}' not recognized, using counterflow")
        counterflow = True

    # ht.core.LMTD handles the calculation including edge cases
    # It raises ValueError if temperatures are invalid
    try:
        return LMTD(Thi=t_hot_in, Tho=t_hot_out, Tci=t_cold_in, Tco=t_cold_out,
                    counterflow=counterflow)
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"LMTD calculation failed: {e}. Check for temperature crossover.")

def calculate_ntu_effectiveness(
    ntu: float,
    capacity_ratio: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """
    Calculate heat exchanger effectiveness using the NTU method.

    Uses ht.hx.effectiveness_from_NTU from the CalebBell/ht library.

    Args:
        ntu: Number of Transfer Units (dimensionless)
        capacity_ratio: Heat capacity ratio Cmin/Cmax (dimensionless)
        flow_arrangement: Flow arrangement type - one of:
            - 'counterflow': Pure counterflow
            - 'parallelflow' or 'parallel': Co-current flow
            - 'crossflow': Crossflow, both fluids unmixed
            - 'shell_tube_1_pass' or 'shell_and_tube': TEMA E configuration

    Returns:
        Heat exchanger effectiveness (0-1)

    Raises:
        ImportError: If ht library is not available
    """
    if not HT_AVAILABLE:
        raise ImportError("ht library required for NTU-effectiveness calculation")

    from ht.hx import effectiveness_from_NTU

    flow_lower = flow_arrangement.lower()

    # Map flow arrangement names to ht library subtypes
    subtype_map = {
        'counterflow': 'counterflow',
        'parallelflow': 'parallel',
        'parallel': 'parallel',
        'cocurrent': 'parallel',
        'crossflow_unmixed': 'crossflow',
        'crossflow': 'crossflow',
        'shell_tube_1_pass': 'TEMA E',
        'shell_and_tube': 'TEMA E',
        'shell_tube': 'TEMA E',
        'tema_e': 'TEMA E',
        'tema_j': 'TEMA J',
        'tema_h': 'TEMA H',
        'tema_g': 'TEMA G',
    }

    subtype = subtype_map.get(flow_lower)
    if subtype is None:
        logger.warning(f"Flow arrangement '{flow_arrangement}' not recognized, using counterflow")
        subtype = 'counterflow'

    effectiveness = effectiveness_from_NTU(ntu, capacity_ratio, subtype=subtype)

    logger.debug(f"ht.effectiveness_from_NTU: NTU={ntu:.3f}, Cr={capacity_ratio:.3f}, "
                 f"subtype={subtype} → ε={effectiveness:.4f}")
    return effectiveness
