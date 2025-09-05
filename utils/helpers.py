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
    geometry: str = "flat_plate"
) -> float:
    """
    Calculate Nusselt number for external flow over various geometries.
    
    Args:
        reynolds_number: Reynolds number (dimensionless)
        prandtl_number: Prandtl number (dimensionless)
        geometry: Geometry type ("flat_plate", "cylinder", "sphere")
        
    Returns:
        Nusselt number (dimensionless)
    """
    geometry_lower = geometry.lower()

    # Prefer correlations from ht when available
    if HT_AVAILABLE:
        try:
            if geometry_lower == "flat_plate" or "flat_plate_external" in geometry_lower:
                from ht.conv_external import Nu_external_horizontal_plate
                return Nu_external_horizontal_plate(reynolds_number, prandtl_number)
            if "cylinder" in geometry_lower:
                from ht.conv_external import Nu_external_cylinder
                return Nu_external_cylinder(reynolds_number, prandtl_number)
        except Exception as e:
            logger.debug(f"ht external convection call failed ({geometry}): {e}; using fallback.")

    # Fallback simple correlations
    if geometry_lower == "flat_plate" or "flat_plate_external" in geometry_lower:
        if reynolds_number < 5e5:  # Laminar
            return 0.664 * math.sqrt(reynolds_number) * prandtl_number**(1/3)
        else:
            return 0.037 * reynolds_number**0.8 * prandtl_number**(1/3)
    if "cylinder" in geometry_lower:
        return 0.3 + ((0.62 * math.sqrt(reynolds_number) * prandtl_number**(1/3) *
                      (1 + (reynolds_number/282000)**(5/8))**(4/5)) /
                      (1 + (0.4/prandtl_number)**(2/3))**(1/4))
    if "sphere" in geometry_lower:
        # Whitaker-like simple fallback
        return 2 + 0.6 * math.sqrt(reynolds_number) * prandtl_number**(1/3)
    logger.warning(f"Geometry '{geometry}' not recognized for Nusselt calculation")
    return 2.0

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
    # Simple approximation if dew point not available
    if dew_point is None:
        # Clear sky approximation: T_sky ≈ 0.0552 * T_ambient^1.5
        t_sky_clear = 0.0552 * (ambient_temperature**1.5)
    else:
        # More accurate with dew point: T_sky ≈ T_ambient * (0.711 + 0.0056*T_dp + 0.000073*T_dp² + 0.013*cos(15t))
        # Where T_dp is dew point in °C and t is the hour of the day/24
        # Simplified version:
        dew_point_c = dew_point - DEG_C_to_K if dew_point > 100 else dew_point
        t_sky_clear = ambient_temperature * (0.711 + 0.0056 * dew_point_c + 0.000073 * dew_point_c**2)
    
    # Adjust for cloud cover if provided
    if cloud_cover is not None:
        # With clouds: T_sky = T_ambient * (clear_sky_factor * (1 - cloud_cover) + cloud_cover)
        clear_sky_factor = t_sky_clear / ambient_temperature
        t_sky = ambient_temperature * (clear_sky_factor * (1 - cloud_cover) + cloud_cover)
        return t_sky
    
    return t_sky_clear

def calculate_lmtd(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """
    Calculate Log Mean Temperature Difference (LMTD) for heat exchangers.
    
    Args:
        t_hot_in: Hot fluid inlet temperature in K
        t_hot_out: Hot fluid outlet temperature in K
        t_cold_in: Cold fluid inlet temperature in K
        t_cold_out: Cold fluid outlet temperature in K
        flow_arrangement: Flow arrangement ("counterflow", "parallelflow")
        
    Returns:
        Log Mean Temperature Difference in K
    """
    flow_lower = flow_arrangement.lower()
    
    if flow_lower == "counterflow":
        delta_t1 = t_hot_in - t_cold_out
        delta_t2 = t_hot_out - t_cold_in
    elif flow_lower == "parallelflow":
        delta_t1 = t_hot_in - t_cold_in
        delta_t2 = t_hot_out - t_cold_out
    else:
        logger.warning(f"Flow arrangement '{flow_arrangement}' not recognized, using counterflow")
        delta_t1 = t_hot_in - t_cold_out
        delta_t2 = t_hot_out - t_cold_in
    
    # Check for valid temperature differences
    if delta_t1 <= 0 or delta_t2 <= 0:
        raise ValueError("Temperature differences must be positive for valid LMTD")
    
    # Handle case when delta_t1 ≈ delta_t2
    if abs(delta_t1 - delta_t2) < 0.01:
        return delta_t1
    
    return (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

def calculate_ntu_effectiveness(
    ntu: float,
    capacity_ratio: float,
    flow_arrangement: str = "counterflow"
) -> float:
    """
    Calculate heat exchanger effectiveness using the NTU method.
    Uses ht library functions when available for maximum accuracy.
    
    Args:
        ntu: Number of Transfer Units (dimensionless)
        capacity_ratio: Heat capacity ratio Cmin/Cmax (dimensionless)
        flow_arrangement: Flow arrangement type
        
    Returns:
        Heat exchanger effectiveness (0-1)
    """
    flow_lower = flow_arrangement.lower()
    
    # Try to use ht library effectiveness functions first
    if HT_AVAILABLE:
        try:
            from ht.hx import effectiveness_from_NTU
            
            # Map our flow arrangement names to ht library subtypes
            subtype_map = {
                'counterflow': 'counterflow',
                'parallelflow': 'parallel',
                'crossflow_unmixed': 'crossflow',
                'crossflow': 'crossflow',
                'shell_tube_1_pass': 'TEMA E', # TEMA E shell-and-tube configuration
                'shell_and_tube': 'TEMA E'
            }
            
            subtype = subtype_map.get(flow_lower, 'counterflow')
            effectiveness = effectiveness_from_NTU(ntu, capacity_ratio, subtype=subtype)
            
            logger.debug(f"Used ht.effectiveness_from_NTU: ε={effectiveness:.4f} for {flow_arrangement}")
            return effectiveness
            
        except (ImportError, Exception) as e:
            logger.debug(f"ht library effectiveness calculation failed: {e}. Using fallback.")
    
    # Fallback to manual correlations if ht not available
    # Handle special case where Cmin/Cmax = 0 (e.g., condensing/evaporating fluid)
    if capacity_ratio < 0.001:
        return 1.0 - math.exp(-ntu)
    
    if flow_lower == "counterflow":
        # Counter flow effectiveness
        if abs(capacity_ratio - 1.0) < 0.001:
            # Special case for Cr = 1
            return ntu / (1.0 + ntu)
        else:
            return (1.0 - math.exp(-ntu * (1.0 - capacity_ratio))) / (1.0 - capacity_ratio * math.exp(-ntu * (1.0 - capacity_ratio)))
    
    elif flow_lower == "parallelflow":
        # Parallel flow effectiveness
        return (1.0 - math.exp(-ntu * (1.0 + capacity_ratio))) / (1.0 + capacity_ratio)
    
    elif flow_lower in ["crossflow", "crossflow_unmixed"]:
        # Cross flow, both fluids unmixed
        return 1.0 - math.exp((1.0 / capacity_ratio) * ntu**0.22 * (math.exp(-capacity_ratio * ntu**0.78) - 1.0))
    
    elif flow_lower == "shell_tube_1_pass":
        # Shell and tube, 1 shell pass, 2, 4, 6... tube passes
        term1 = 2.0 / (1.0 + capacity_ratio + math.sqrt(1.0 + capacity_ratio**2))
        term2 = 1.0 + math.exp(-ntu * math.sqrt(1.0 + capacity_ratio**2))
        term3 = 1.0 - math.exp(-ntu * math.sqrt(1.0 + capacity_ratio**2))
        return term1 * (term2 / term3)
    
    else:
        logger.warning(f"Flow arrangement '{flow_arrangement}' not recognized, using counterflow")
        if abs(capacity_ratio - 1.0) < 0.001:
            return ntu / (1.0 + ntu)
        else:
            return (1.0 - math.exp(-ntu * (1.0 - capacity_ratio))) / (1.0 - capacity_ratio * math.exp(-ntu * (1.0 - capacity_ratio)))
