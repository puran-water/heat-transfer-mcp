"""
Convection coefficient tool to calculate heat transfer coefficients for various geometries.

This module provides functionality to calculate convective heat transfer coefficients
using established correlations from the HT library or fallback implementations.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

from utils.constants import DEG_C_to_K
from utils.import_helpers import HT_AVAILABLE
from utils.helpers import (
    calculate_reynolds_number, 
    calculate_prandtl_number, 
    calculate_nusselt_number_external_flow
)

# Tools might need to call other tools 
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.convection_coefficient")

def calculate_convection_coefficient(
    geometry: str,
    characteristic_dimension: float,
    fluid_name: str,
    bulk_fluid_temperature: float,
    surface_temperature: float,
    pressure: float = 101325.0,
    flow_type: str = "forced",
    fluid_velocity: Optional[float] = None,
    roughness: Optional[float] = 0.0,
    pipe_length: Optional[float] = None,
    strict: bool = False,
) -> str:
    """Calculates convective heat transfer coefficient for various geometries and flow conditions.
    
    Args:
        geometry: Geometry type (e.g., 'flat_plate_external', 'pipe_internal')
        characteristic_dimension: Characteristic length/diameter in meters
        fluid_name: Name of the fluid
        bulk_fluid_temperature: Bulk temperature of the fluid in Kelvin
        surface_temperature: Temperature of the surface in Kelvin
        pressure: Pressure in Pascals
        flow_type: Flow regime ('natural' or 'forced')
        fluid_velocity: Fluid velocity in m/s (required for 'forced' flow)
        roughness: Surface roughness in meters (relevant for internal pipe flow)
        strict: If True, require ht library and fail if correlations are not available
        
    Returns:
        JSON string with the calculated convection coefficient and related parameters
    """
    # Validate inputs to avoid misleading errors and non-physical results
    try:
        if not isinstance(geometry, str) or not geometry.strip():
            return json.dumps({
                "error": "Geometry must be a non-empty string."
            })
        geometry_lower = geometry.lower()
        recognized_keywords = ['flat_plate', 'plate', 'pipe', 'cylinder', 'sphere', 'vertical', 'horizontal', 'wall']
        if not any(k in geometry_lower for k in recognized_keywords):
            return json.dumps({
                "error": f"Unsupported geometry: {geometry}."
            })
        if flow_type is None or flow_type.lower() not in {"forced", "natural"}:
            return json.dumps({
                "error": "flow_type must be 'forced' or 'natural'."
            })
        # Characteristic dimension
        try:
            characteristic_dimension = float(characteristic_dimension)
        except (TypeError, ValueError):
            return json.dumps({
                "error": "Characteristic dimension must be a numeric value in meters."
            })
        if not math.isfinite(characteristic_dimension) or characteristic_dimension <= 0.0:
            return json.dumps({
                "error": "Characteristic dimension must be a positive, finite value."
            })
        # Forced convection requires a valid velocity
        if flow_type.lower() == 'forced':
            if fluid_velocity is None:
                return json.dumps({
                    "error": "Fluid velocity is required for forced convection calculations."
                })
            try:
                fluid_velocity = float(fluid_velocity)
            except (TypeError, ValueError):
                return json.dumps({
                    "error": "Fluid velocity must be a numeric value in m/s."
                })
            if not math.isfinite(fluid_velocity) or fluid_velocity <= 0.0:
                return json.dumps({
                    "error": "Fluid velocity must be a positive, finite value for forced convection."
                })
        # Temperatures and pressure
        try:
            bulk_fluid_temperature = float(bulk_fluid_temperature)
            surface_temperature = float(surface_temperature)
            pressure = float(pressure)
        except (TypeError, ValueError):
            return json.dumps({
                "error": "Temperatures (K) and pressure (Pa) must be numeric values."
            })
        for T_val, label in [(bulk_fluid_temperature, 'bulk_fluid_temperature'), (surface_temperature, 'surface_temperature')]:
            if not math.isfinite(T_val):
                return json.dumps({
                    "error": f"{label} must be a finite real number."
                })
            if T_val < 0.0:
                return json.dumps({
                    "error": f"{label} cannot be below 0 K (absolute zero)."
                })
            if T_val > 1.0e4:
                return json.dumps({
                    "error": f"{label} is unrealistically high. Please provide < 10000 K."
                })
        if not math.isfinite(pressure) or pressure <= 0.0:
            return json.dumps({
                "error": "Pressure must be a positive, finite value."
            })
        if roughness is not None:
            try:
                roughness = float(roughness)
            except (TypeError, ValueError):
                return json.dumps({
                    "error": "Roughness must be a numeric value in meters."
                })
            if not math.isfinite(roughness) or roughness < 0.0:
                return json.dumps({
                    "error": "Roughness must be a non-negative, finite value."
                })

        # Calculate film temperature (average of bulk and surface)
        film_temperature = (bulk_fluid_temperature + surface_temperature) / 2.0
        
        # Get fluid properties at film temperature
        fluid_props_json = get_fluid_properties(fluid_name, film_temperature, pressure, strict=strict)
        fluid_props = json.loads(fluid_props_json)
        
        if "error" in fluid_props:
            return json.dumps({
                "error": f"Failed to get fluid properties: {fluid_props['error']}"
            })
        
        # Extract required properties
        density = fluid_props.get("density")
        dynamic_viscosity = fluid_props.get("dynamic_viscosity")
        thermal_conductivity = fluid_props.get("thermal_conductivity")
        specific_heat_cp = fluid_props.get("specific_heat_cp")
        prandtl_number = fluid_props.get("prandtl_number")
        
        # Check for required properties
        if None in [density, dynamic_viscosity, thermal_conductivity]:
            return json.dumps({
                "error": "Missing critical fluid properties for convection calculation."
            })
        
        # If Prandtl number not provided, calculate it
        if prandtl_number is None and specific_heat_cp is not None:
            prandtl_number = calculate_prandtl_number(
                dynamic_viscosity, specific_heat_cp, thermal_conductivity
            )
        
        # Calculate Reynolds number for forced flow
        reynolds_number = None
        if flow_type.lower() == 'forced' and fluid_velocity is not None:
            reynolds_number = calculate_reynolds_number(
                fluid_velocity, characteristic_dimension, density, dynamic_viscosity
            )
        
        # Calculate Nusselt number and convection coefficient based on geometry and flow regime
        result = {}
        geometry_lower = geometry.lower()
        nusselt_number = None
        convection_coefficient = None
        
        if strict and not HT_AVAILABLE:
            raise ImportError("ht library required with strict=True")
            
        if HT_AVAILABLE:
            import ht
            
            try:
                # Use HT library correlations directly instead of manual implementations
                if 'flat_plate_external' in geometry_lower and flow_type.lower() == 'forced':
                    # Use ht.conv_external meta-function for plates
                    from ht.conv_external import Nu_external_horizontal_plate
                    nusselt_number = Nu_external_horizontal_plate(reynolds_number, prandtl_number)
                
                elif 'pipe_internal' in geometry_lower and flow_type.lower() == 'forced':
                    from ht.conv_internal import Nu_conv_internal
                    eD = roughness / characteristic_dimension if roughness else 0.0
                    kwargs = {"eD": eD}
                    if pipe_length is not None and characteristic_dimension is not None:
                        kwargs.update({"Di": characteristic_dimension, "x": pipe_length})
                    nusselt_number = Nu_conv_internal(reynolds_number, prandtl_number, **kwargs)
                
                elif 'pipe_external' in geometry_lower or 'cylinder' in geometry_lower:
                    if flow_type.lower() == 'forced':
                        # Use ht.conv_external meta-function for cylinder; include wall Pr if available
                        from ht.conv_external import Nu_external_cylinder
                        prandtl_wall = None
                        try:
                            wall_props = json.loads(get_fluid_properties(fluid_name, surface_temperature, pressure, strict=strict))
                            if "error" not in wall_props:
                                mu_w = wall_props.get("dynamic_viscosity")
                                Cp_w = wall_props.get("specific_heat_cp")
                                k_w = wall_props.get("thermal_conductivity")
                                if mu_w and Cp_w and k_w:
                                    prandtl_wall = mu_w * Cp_w / k_w
                        except Exception:
                            pass
                        if prandtl_wall is not None:
                            nusselt_number = Nu_external_cylinder(reynolds_number, prandtl_number, Prw=prandtl_wall)
                        else:
                            nusselt_number = Nu_external_cylinder(reynolds_number, prandtl_number)
                    else:  # Natural convection
                        # Use ht.conv_free_immersed for natural convection around cylinder
                        from ht.conv_free_immersed import Nu_horizontal_cylinder
                        # Calculate Grashof number for ht function
                        g = 9.81  # m/s²
                        beta = 1.0 / film_temperature  # Thermal expansion coefficient (approximation)
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density
                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        nusselt_number = Nu_horizontal_cylinder(prandtl_number, grashof)
                
                elif 'sphere' in geometry_lower:
                    if flow_type.lower() == 'forced':
                        # Whitaker correlation for forced flow over sphere
                        nusselt_number = 2 + (0.4 * math.sqrt(reynolds_number) + 0.06 * reynolds_number**(2/3)) * prandtl_number**0.4
                    else:  # Natural convection
                        from ht.conv_free_immersed import Nu_sphere_Churchill
                        g = 9.81
                        beta = 1.0 / film_temperature
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density
                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        rayleigh = grashof * prandtl_number
                        nusselt_number = Nu_sphere_Churchill(prandtl_number, rayleigh)
                
                elif ('vertical' in geometry_lower or 'wall' in geometry_lower) and flow_type.lower() == 'natural':
                    from ht.conv_free_immersed import Nu_free_vertical_plate
                    g = 9.81
                    beta = 1.0 / film_temperature
                    delta_t = abs(surface_temperature - bulk_fluid_temperature)
                    kinematic_viscosity = dynamic_viscosity / density
                    grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                    nusselt_number = Nu_free_vertical_plate(prandtl_number, grashof)
                else:
                    # For other geometries, try a fallback approach
                    if flow_type.lower() == 'forced':
                        nusselt_number = calculate_nusselt_number_external_flow(
                            reynolds_number, prandtl_number, geometry)
                    else:  # Natural
                        # Basic natural convection for various geometries
                        g = 9.81
                        beta = 1.0 / film_temperature
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density
                        
                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        rayleigh = grashof * prandtl_number
                        
                        # Generic natural convection - use appropriate correlation based on geometry
                        if 'vertical' in geometry_lower:
                            if rayleigh < 1e9:  # Laminar
                                nusselt_number = 0.59 * rayleigh**(1/4)
                            else:  # Turbulent
                                nusselt_number = 0.1 * rayleigh**(1/3)
                        elif 'horizontal' in geometry_lower:
                            if 'plate_up' in geometry_lower:  # Upper surface of heated plate
                                if rayleigh < 1e7:  # Laminar
                                    nusselt_number = 0.54 * rayleigh**(1/4)
                                else:  # Turbulent
                                    nusselt_number = 0.15 * rayleigh**(1/3)
                            else:  # Lower surface of heated plate or upper surface of cooled plate
                                nusselt_number = 0.27 * rayleigh**(1/4)
                        else:
                            # Default approximation
                            nusselt_number = 0.5 * rayleigh**(1/4)
                
            except Exception as ht_error:
                if strict:
                    raise
                logger.warning(f"Error using HT library correlations: {ht_error}")
                # Fall back to basic correlations
                nusselt_number = None
        
        # If no nusselt_number was calculated using HT, use fallback correlations
        if nusselt_number is None:
            if strict:
                raise ValueError("Could not calculate Nusselt number with strict=True")
            if flow_type.lower() == 'forced':
                if 'pipe_internal' in geometry_lower:
                    # Basic Dittus-Boelter
                    if reynolds_number < 2300:  # Laminar
                        nusselt_number = 3.66
                    else:  # Turbulent
                        nusselt_number = 0.023 * reynolds_number**0.8 * prandtl_number**0.4
                else:
                    # Use helper function for external flows
                    nusselt_number = calculate_nusselt_number_external_flow(
                        reynolds_number, prandtl_number, geometry)
            else:  # Natural
                # Basic natural convection fallback
                g = 9.81
                beta = 1.0 / film_temperature
                delta_t = abs(surface_temperature - bulk_fluid_temperature)
                kinematic_viscosity = dynamic_viscosity / density
                
                grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                rayleigh = grashof * prandtl_number
                
                # Very basic correlation for natural convection
                nusselt_number = 0.54 * rayleigh**(1/4)
        
        # Calculate convection coefficient from Nusselt number
        if nusselt_number is not None:
            convection_coefficient = nusselt_number * thermal_conductivity / characteristic_dimension
        else:
            return json.dumps({
                "error": "Could not calculate Nusselt number for the given conditions."
            })
        
        # Prepare result
        result = {
            "convection_coefficient_h": convection_coefficient,
            "convection_coefficient_h_W_m2K": convection_coefficient,
            "nusselt_number": nusselt_number,
            "geometry": geometry,
            "flow_type": flow_type,
            "fluid_properties": {
                "name": fluid_name,
                "density": density,
                "dynamic_viscosity": dynamic_viscosity,
                "thermal_conductivity": thermal_conductivity,
                "prandtl_number": prandtl_number,
                "film_temperature": film_temperature
            },
            "calculation_details": {}
        }
        
        # Add flow-specific details
        if flow_type.lower() == 'forced':
            result["calculation_details"]["reynolds_number"] = reynolds_number
            result["calculation_details"]["fluid_velocity"] = fluid_velocity
        else:  # Natural convection
            g = 9.81
            beta = 1.0 / film_temperature
            delta_t = abs(surface_temperature - bulk_fluid_temperature)
            if density and dynamic_viscosity:
                kinematic_viscosity = dynamic_viscosity / density
                grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                rayleigh = grashof * prandtl_number if prandtl_number else None
                
                result["calculation_details"]["grashof_number"] = grashof
                result["calculation_details"]["rayleigh_number"] = rayleigh
                result["calculation_details"]["temperature_difference"] = delta_t
        
        # Add units for clarity
        result["unit"] = "W/(m²·K)"
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in calculate_convection_coefficient: {e}", exc_info=True)
        return json.dumps({
            "error": f"An unexpected error occurred: {str(e)}"
        })
