"""
Fluid properties tool to retrieve thermophysical properties of fluids.

This module provides functionality to calculate fluid properties using the HT library.
"""

import json
import logging
import math
from functools import lru_cache
from typing import Dict, List, Optional, Union, Any

from utils.constants import DEG_C_to_K
from utils.import_helpers import HT_AVAILABLE, get_fluid_properties_fallback

logger = logging.getLogger("heat-transfer-mcp.fluid_properties")

@lru_cache(maxsize=128)
def _cached_get_chemical_properties(fluid_name: str, temperature: float, pressure: float) -> Optional[dict]:
    """Cached version of Chemical property lookup to avoid repeated expensive calls."""
    try:
        from thermo import Chemical
        
        # Handle special cases for common fluid names
        chemical_name = fluid_name
        if fluid_name.lower() == 'air':
            chemical_name = 'nitrogen'
        
        # Create Chemical object with specified conditions
        chem = Chemical(chemical_name, T=temperature, P=pressure)
        
        # Check phase - only allow liquid or gas phases
        if hasattr(chem, 'phase') and chem.phase not in ['l', 'g']:
            raise ValueError(f"Unsupported phase '{chem.phase}' for {fluid_name} at T={temperature}K, P={pressure}Pa")
        
        return {
            'rho': float(chem.rho) if hasattr(chem, 'rho') and chem.rho is not None else None,
            'Cp': float(chem.Cp) if hasattr(chem, 'Cp') and chem.Cp is not None else None,
            'k': float(chem.k) if hasattr(chem, 'k') and chem.k is not None else None,
            'mu': float(chem.mu) if hasattr(chem, 'mu') and chem.mu is not None else None,
            'nu': float(chem.nu) if hasattr(chem, 'nu') and chem.nu is not None else None,
            'Pr': float(chem.Pr) if hasattr(chem, 'Pr') and chem.Pr is not None else None,
            'phase': getattr(chem, 'phase', 'unknown')
        }
    except Exception:
        return None

def get_fluid_properties(
    fluid_name: str,
    temperature: float,
    pressure: float = 101325.0,
    strict: bool = False
) -> str:
    """Retrieves thermophysical properties of common fluids at specified conditions.
    
    Args:
        fluid_name: Name of the fluid (e.g., 'water', 'air', 'Ethanol')
        temperature: Temperature in Kelvin (K)
        pressure: Pressure in Pascals (Pa)
        strict: If True, require ht/thermo libraries and fail if not available
        
    Returns:
        JSON string with detailed fluid properties
    """
    try:
        result = {}
        
        if strict and not HT_AVAILABLE:
            raise ImportError("ht library required with strict=True")
            
        if HT_AVAILABLE:
            import ht
            
            try:
                # Try using thermo.Chemical for accurate fluid properties
                logger.info(f"Attempting to get properties for {fluid_name} using thermo.Chemical")
                props = None
                
                try:
                    # Use cached function to avoid repeated expensive Chemical lookups
                    cached_props = _cached_get_chemical_properties(fluid_name, temperature, pressure)
                    
                    if cached_props is not None:
                        # Create a props object with the cached properties
                        class ChemicalProps:
                            pass
                        props = ChemicalProps()
                        for key, value in cached_props.items():
                            setattr(props, key, value)
                        
                        logger.info(f"Successfully obtained fluid properties for {fluid_name} using cached thermo.Chemical")
                    else:
                        props = None
                except ImportError as ie:
                    if strict:
                        raise ImportError(f"thermo module required with strict=True: {ie}")
                    logger.error(f"thermo module not installed: {ie}")
                    props = None
                except (AttributeError, TypeError, ValueError) as chem_error:
                    if strict:
                        raise
                    logger.warning(f"Error using thermo.Chemical for {fluid_name}: {chem_error}")
                    props = None
                        
                if props is not None:
                    # Format properties
                    result = {
                        "fluid_name": fluid_name,
                        "temperature_k": temperature,
                        "pressure_pa": pressure,
                        "density": float(props.rho) if hasattr(props, 'rho') else None,
                        "specific_heat_cp": float(props.Cp) if hasattr(props, 'Cp') else None,
                        "thermal_conductivity": float(props.k) if hasattr(props, 'k') else None,
                        "dynamic_viscosity": float(props.mu) if hasattr(props, 'mu') else None,
                        "kinematic_viscosity": float(props.nu) if hasattr(props, 'nu') else None,
                        "prandtl_number": float(props.Pr) if hasattr(props, 'Pr') else None,
                        "phase": getattr(props, 'phase', 'unknown'),
                        "data_source": "thermo_library",
                        "accuracy": "high"
                    }
                else:
                    # Try fallback only if not in strict mode
                    if strict:
                        raise ValueError(f"Could not retrieve properties for '{fluid_name}' with strict=True")
                    logger.info(f"No properties found via ht for {fluid_name}, using fallback")
                    fallback_props = get_fluid_properties_fallback(fluid_name, temperature, pressure)
                    if fallback_props:
                        result = {
                            "fluid_name": fluid_name,
                            "temperature_k": temperature,
                            "pressure_pa": pressure,
                            "density": fallback_props.get("density"),
                            "specific_heat_cp": fallback_props.get("specific_heat_cp"),
                            "thermal_conductivity": fallback_props.get("thermal_conductivity"),
                            "dynamic_viscosity": fallback_props.get("dynamic_viscosity"),
                            "kinematic_viscosity": fallback_props.get("kinematic_viscosity"),
                            "prandtl_number": fallback_props.get("prandtl_number"),
                            "data_source": "fallback_calculations",
                            "accuracy": "approximate"
                        }
                    else:
                        return json.dumps({
                            "error": f"Could not retrieve properties for '{fluid_name}'. The fluid may not be supported."
                        })
            except Exception as e:
                if strict:
                    raise
                logger.error(f"Error in get_fluid_properties with ht: {e}", exc_info=True)
                
                # Try fallback
                fallback_props = get_fluid_properties_fallback(fluid_name, temperature, pressure)
                if fallback_props:
                    result = {
                        "fluid_name": fluid_name,
                        "temperature_k": temperature,
                        "pressure_pa": pressure,
                        "density": fallback_props.get("density"),
                        "specific_heat_cp": fallback_props.get("specific_heat_cp"),
                        "thermal_conductivity": fallback_props.get("thermal_conductivity"),
                        "dynamic_viscosity": fallback_props.get("dynamic_viscosity"),
                        "kinematic_viscosity": fallback_props.get("kinematic_viscosity"),
                        "prandtl_number": fallback_props.get("prandtl_number"),
                        "data_source": "fallback_calculations",
                        "accuracy": "approximate"
                    }
                else:
                    return json.dumps({
                        "error": f"Could not retrieve properties for '{fluid_name}' with ht or fallback. Error: {str(e)}"
                    })
        else:
            # Use the fallback function only if not in strict mode
            if strict:
                raise ImportError("ht library required with strict=True but not available")
            logger.info(f"HT library not available, using fallback for {fluid_name}")
            fallback_props = get_fluid_properties_fallback(fluid_name, temperature, pressure)
            if fallback_props:
                result = {
                    "fluid_name": fluid_name,
                    "temperature_k": temperature,
                    "pressure_pa": pressure,
                    "density": fallback_props.get("density"),
                    "specific_heat_cp": fallback_props.get("specific_heat_cp"),
                    "thermal_conductivity": fallback_props.get("thermal_conductivity"),
                    "dynamic_viscosity": fallback_props.get("dynamic_viscosity"),
                    "kinematic_viscosity": fallback_props.get("kinematic_viscosity"),
                    "prandtl_number": fallback_props.get("prandtl_number"),
                    "data_source": "fallback_calculations",
                    "accuracy": "approximate"
                }
            else:
                return json.dumps({
                    "error": f"Could not retrieve properties for '{fluid_name}'. The fluid may not be supported."
                })
                
        # Clean up result - remove None values
        clean_result = {k: v for k, v in result.items() if v is not None}
        
        # Add a note about the temperature in Celsius for convenience
        if "temperature_k" in clean_result:
            clean_result["temperature_c"] = round(clean_result["temperature_k"] - DEG_C_to_K, 2)
        
        return json.dumps(clean_result)
        
    except Exception as e:
        logger.error(f"Unexpected error in get_fluid_properties: {e}", exc_info=True)
        return json.dumps({
            "error": f"An unexpected error occurred: {str(e)}"
        })
