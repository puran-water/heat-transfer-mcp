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
from utils.import_helpers import (
    HT_AVAILABLE,
    THERMO_AVAILABLE,
    COOLPROP_AVAILABLE,
    get_fluid_properties_fallback,
    get_fluid_properties_coolprop,
)

logger = logging.getLogger("heat-transfer-mcp.fluid_properties")

@lru_cache(maxsize=128)
def _cached_get_properties_thermo(fluid_name: str, temperature: float, pressure: float) -> Optional[dict]:
    """Cached property lookup using thermo for pure fluids and common mixtures.

    - Uses Chemical for pure components
    - Uses DryAirLemmon/Stream for air
    """
    if not THERMO_AVAILABLE:
        return None
    try:
        # Handle common mixture aliases first
        fname = fluid_name.strip().lower()
        if fname == 'air':
            # Prefer accurate dry air EOS if available
            try:
                from thermo.phases.air_phase import DryAirLemmon
                air = DryAirLemmon(T=temperature, P=pressure)
                # DryAirLemmon returns molar properties, convert to mass basis
                # Molecular weight of air is approximately 28.965 g/mol
                MW_air = 0.028965  # kg/mol
                rho_molar = float(air.rho())  # mol/m³
                rho = rho_molar * MW_air  # kg/m³
                Cp_molar = float(air.Cp())  # J/mol/K
                Cp = Cp_molar / MW_air  # J/kg/K
                k = float(air.k())  # W/m/K (already intensive)
                mu = float(air.mu())  # Pa·s (already intensive)
                nu = mu / rho if rho else None
                Pr = float(mu*Cp/k) if (mu and Cp and k) else None
                return {'rho': rho, 'Cp': Cp, 'k': k, 'mu': mu, 'nu': nu, 'Pr': Pr, 'phase': 'g'}
            except Exception:
                # Fallback to Chemical which handles air properly
                try:
                    from thermo import Chemical
                    chem = Chemical('air', T=temperature, P=pressure)
                    return {
                        'rho': float(chem.rho) if getattr(chem, 'rho', None) is not None else None,
                        'Cp': float(chem.Cp) if getattr(chem, 'Cp', None) is not None else None,
                        'k': float(chem.k) if getattr(chem, 'k', None) is not None else None,
                        'mu': float(chem.mu) if getattr(chem, 'mu', None) is not None else None,
                        'nu': float(chem.nu) if getattr(chem, 'nu', None) is not None else None,
                        'Pr': float(chem.Pr) if getattr(chem, 'Pr', None) is not None else None,
                        'phase': getattr(chem, 'phase', 'g'),
                    }
                except Exception:
                    return None

        # Otherwise treat as pure chemical
        from thermo import Chemical
        chem = Chemical(fluid_name, T=temperature, P=pressure)
        # Only liquid or gas phases are supported for now
        if hasattr(chem, 'phase') and chem.phase not in ['l', 'g']:
            raise ValueError(f"Unsupported phase '{chem.phase}' for {fluid_name} at T={temperature} K, P={pressure} Pa")
        return {
            'rho': float(chem.rho) if getattr(chem, 'rho', None) is not None else None,
            'Cp': float(chem.Cp) if getattr(chem, 'Cp', None) is not None else None,
            'k': float(chem.k) if getattr(chem, 'k', None) is not None else None,
            'mu': float(chem.mu) if getattr(chem, 'mu', None) is not None else None,
            'nu': float(chem.nu) if getattr(chem, 'nu', None) is not None else None,
            'Pr': float(chem.Pr) if getattr(chem, 'Pr', None) is not None else None,
            'phase': getattr(chem, 'phase', 'unknown'),
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
        
        # Basic input validation
        try:
            T = float(temperature)
        except (TypeError, ValueError):
            return json.dumps({
                "error": "Temperature must be a numeric value in Kelvin."
            })
        try:
            P = float(pressure)
        except (TypeError, ValueError):
            return json.dumps({
                "error": "Pressure must be a numeric value in Pascals."
            })
        if T < 0.0:
            return json.dumps({
                "error": "Temperature cannot be below 0 K (absolute zero)."
            })
        if T > 1.0e4:
            return json.dumps({
                "error": "Temperature is unrealistically high. Please provide T < 10000 K."
            })
        if not math.isfinite(T) or not math.isfinite(P):
            return json.dumps({
                "error": "Temperature and pressure must be finite real numbers."
            })
        if P <= 0.0:
            return json.dumps({
                "error": "Pressure must be positive."
            })

        # Try CoolProp first for high-accuracy properties (if available)
        if COOLPROP_AVAILABLE:
            try:
                coolprop_props = get_fluid_properties_coolprop(fluid_name, T, P)
                if coolprop_props is not None:
                    result = {
                        "fluid_name": fluid_name,
                        "temperature_k": T,
                        "pressure_pa": P,
                        "density": coolprop_props.get("density"),
                        "specific_heat_cp": coolprop_props.get("specific_heat_cp"),
                        "thermal_conductivity": coolprop_props.get("thermal_conductivity"),
                        "dynamic_viscosity": coolprop_props.get("dynamic_viscosity"),
                        "kinematic_viscosity": coolprop_props.get("kinematic_viscosity"),
                        "prandtl_number": coolprop_props.get("prandtl_number"),
                        "phase": coolprop_props.get("phase", "unknown"),
                        "data_source": "CoolProp",
                        "accuracy": "high (reference EOS)",
                    }
                    # Add temperature in Celsius for convenience
                    result["temperature_c"] = round(T - DEG_C_to_K, 2)
                    return json.dumps(result)
            except Exception as e:
                logger.debug(f"CoolProp lookup failed for {fluid_name}, falling back to thermo: {e}")

        # Fall back to thermo for property calculations
        if THERMO_AVAILABLE:
            try:
                logger.info(f"Attempting to get properties for {fluid_name} using thermo")
                props = _cached_get_properties_thermo(fluid_name, T, P)
                if props is not None:
                    # Ensure all derived properties are present
                    rho = props.get('rho')
                    Cp = props.get('Cp')
                    k = props.get('k')
                    mu = props.get('mu')
                    nu = props.get('nu') if props.get('nu') is not None and props.get('nu') == props.get('nu') else (mu/rho if (mu and rho) else None)
                    Pr = props.get('Pr') if props.get('Pr') is not None and props.get('Pr') == props.get('Pr') else (mu*Cp/k if (mu and Cp and k) else None)

                    result = {
                        "fluid_name": fluid_name,
                        "temperature_k": T,
                        "pressure_pa": P,
                        "density": rho,
                        "specific_heat_cp": Cp,
                        "thermal_conductivity": k,
                        "dynamic_viscosity": mu,
                        "kinematic_viscosity": nu,
                        "prandtl_number": Pr,
                        "phase": props.get('phase', 'unknown'),
                        "data_source": "thermo_library",
                        "accuracy": "high",
                    }
                else:
                    if strict:
                        raise ValueError(f"Could not retrieve properties for '{fluid_name}' from thermo with strict=True")
                    logger.info(f"thermo could not resolve properties for {fluid_name}, using fallback")
                    fallback_props = get_fluid_properties_fallback(fluid_name, T, P)
                    if fallback_props:
                        result = {
                            "fluid_name": fluid_name,
                            "temperature_k": T,
                            "pressure_pa": P,
                            "density": fallback_props.get("density"),
                            "specific_heat_cp": fallback_props.get("specific_heat_cp"),
                            "thermal_conductivity": fallback_props.get("thermal_conductivity"),
                            "dynamic_viscosity": fallback_props.get("dynamic_viscosity"),
                            "kinematic_viscosity": fallback_props.get("kinematic_viscosity"),
                            "prandtl_number": fallback_props.get("prandtl_number"),
                            "data_source": "fallback_calculations",
                            "accuracy": "approximate",
                        }
                    else:
                        return json.dumps({
                            "error": f"Could not retrieve properties for '{fluid_name}'. The fluid may not be supported."
                        })
            except Exception as e:
                if strict:
                    raise
                logger.error(f"Error obtaining properties with thermo: {e}", exc_info=True)
                # Fall back to basic approximations
                fallback_props = get_fluid_properties_fallback(fluid_name, T, P)
                if fallback_props:
                    result = {
                        "fluid_name": fluid_name,
                        "temperature_k": T,
                        "pressure_pa": P,
                        "density": fallback_props.get("density"),
                        "specific_heat_cp": fallback_props.get("specific_heat_cp"),
                        "thermal_conductivity": fallback_props.get("thermal_conductivity"),
                        "dynamic_viscosity": fallback_props.get("dynamic_viscosity"),
                        "kinematic_viscosity": fallback_props.get("kinematic_viscosity"),
                        "prandtl_number": fallback_props.get("prandtl_number"),
                        "data_source": "fallback_calculations",
                        "accuracy": "approximate",
                    }
                else:
                    return json.dumps({
                        "error": f"Could not retrieve properties for '{fluid_name}' from thermo or fallback. Error: {str(e)}"
                    })
        else:
            # thermo not available: only use fallback if not strict
            if strict:
                raise ImportError("thermo library required with strict=True but not available")
            logger.info(f"Thermo library not available, using fallback for {fluid_name}")
            fallback_props = get_fluid_properties_fallback(fluid_name, T, P)
            if fallback_props:
                result = {
                    "fluid_name": fluid_name,
                    "temperature_k": T,
                    "pressure_pa": P,
                    "density": fallback_props.get("density"),
                    "specific_heat_cp": fallback_props.get("specific_heat_cp"),
                    "thermal_conductivity": fallback_props.get("thermal_conductivity"),
                    "dynamic_viscosity": fallback_props.get("dynamic_viscosity"),
                    "kinematic_viscosity": fallback_props.get("kinematic_viscosity"),
                    "prandtl_number": fallback_props.get("prandtl_number"),
                    "data_source": "fallback_calculations",
                    "accuracy": "approximate",
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
