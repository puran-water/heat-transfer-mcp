"""
Surface heat transfer tool to calculate net heat transfer from external surfaces.

This module provides functionality to calculate heat loss/gain from surfaces
considering convection, radiation, and optional solar gain.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

from utils.constants import STEFAN_BOLTZMANN
from utils.import_helpers import HT_AVAILABLE
from utils.helpers import calculate_radiation_heat_transfer, estimate_sky_temperature
from tools.convection_coefficient import calculate_convection_coefficient
from tools.material_properties import get_material_properties
from tools.overall_heat_transfer import calculate_overall_heat_transfer_coefficient

logger = logging.getLogger("heat-transfer-mcp.surface_heat_transfer")

def calculate_surface_heat_transfer(
    geometry: str,
    dimensions: Dict[str, float],
    internal_temperature: float,
    ambient_air_temperature: float,
    wind_speed: float,
    surface_emissivity: float,
    overall_heat_transfer_coefficient_U: Optional[float] = None,
    wall_layers: Optional[List[Dict[str, Any]]] = None,
    fluid_name_internal: str = "water",
    fluid_name_external: str = "air",
    include_solar_gain: bool = False,
    incident_solar_radiation: Optional[float] = None,
    surface_absorptivity: float = 0.8,
    sky_temperature: Optional[float] = None,
) -> str:
    """Calculates net heat loss/gain from an external surface considering convection and radiation.
    
    Args:
        geometry: Surface geometry type (e.g., 'vertical_cylinder_tank', 'flat_surface')
        dimensions: Dictionary of dimensions in meters (e.g., {diameter: d, height: h})
        internal_temperature: Temperature of the fluid/contents inside in Kelvin
        ambient_air_temperature: Ambient air temperature in Kelvin
        wind_speed: Wind speed in m/s
        surface_emissivity: Emissivity of the outer surface (0 to 1)
        overall_heat_transfer_coefficient_U: Optional pre-calculated U-value (W/m²K)
        wall_layers: Optional list of wall material layers if U is not provided
        fluid_name_internal: Internal fluid name (default 'water')
        fluid_name_external: External fluid name (default 'air')
        include_solar_gain: Whether to include solar gain in the calculation
        incident_solar_radiation: Total incident solar radiation on surface (W/m²)
        surface_absorptivity: Absorptivity of outer surface for solar radiation (0 to 1)
        sky_temperature: Effective sky temperature for radiation calculation in Kelvin
        
    Returns:
        JSON string with calculated heat transfer results
    """
    # Validate inputs
    if overall_heat_transfer_coefficient_U is None and wall_layers is None:
        return json.dumps({
            "error": "Either 'overall_heat_transfer_coefficient_U' or 'wall_layers' must be provided."
        })
    
    if include_solar_gain and incident_solar_radiation is None:
        return json.dumps({
            "error": "Incident solar radiation is required when include_solar_gain is True."
        })
    
    try:
        # 1. Calculate Surface Area
        outer_surface_area = 0.0
        geometry_lower = geometry.lower()
        diameter = dimensions.get('diameter')
        height = dimensions.get('height')
        length = dimensions.get('length')
        width = dimensions.get('width')
        
        if ('cylinder_tank' in geometry_lower or 'tank' in geometry_lower) and diameter and height:
            if 'vertical' in geometry_lower:
                # Vertical cylinder
                outer_surface_area = math.pi * diameter * height + math.pi * (diameter/2)**2
            elif 'horizontal' in geometry_lower:
                # Horizontal cylinder
                outer_surface_area = math.pi * diameter * length + 2 * math.pi * (diameter/2)**2
            else:
                # Default to vertical if not specified
                outer_surface_area = math.pi * diameter * height + math.pi * (diameter/2)**2
        elif 'flat_surface' in geometry_lower or 'wall' in geometry_lower:
            if length and width:
                outer_surface_area = length * width
            else:
                return json.dumps({
                    "error": f"Missing dimensions for {geometry}. Need 'length' and 'width'."
                })
        elif 'pipe' in geometry_lower and diameter and length:
            outer_surface_area = math.pi * diameter * length
        else:
            return json.dumps({
                "error": f"Unsupported or incompletely defined geometry: {geometry}"
            })
        
        # 2. Estimate surface temperature and heat transfer iteratively
        # Start with an initial guess
        T_internal_K = internal_temperature
        T_ambient_K = ambient_air_temperature
        Ts_outer_K_guess = (T_internal_K + T_ambient_K) / 2.0  # Initial guess
        
        # Determine sky temperature if not provided
        T_sky_K = sky_temperature if sky_temperature else estimate_sky_temperature(T_ambient_K)
        
        # Iteration setup
        max_iterations = 50
        tolerance = 0.1  # Kelvin
        result_log = []  # To track convergence
        
        # Estimate external convection coefficient
        conv_geometry = ""
        if 'vertical' in geometry_lower and 'cylinder' in geometry_lower:
            conv_geometry = "vertical_cylinder_external"
        elif 'horizontal' in geometry_lower and 'cylinder' in geometry_lower:
            conv_geometry = "horizontal_cylinder_external"
        elif 'flat' in geometry_lower or 'wall' in geometry_lower:
            if 'vertical' in geometry_lower:
                conv_geometry = "vertical_flat_plate_external"
            else:
                conv_geometry = "flat_plate_external"
        elif 'pipe' in geometry_lower:
            conv_geometry = "pipe_external"
        else:
            conv_geometry = "flat_plate_external"  # Default
        
        # Get external convection coefficient
        conv_coeff_json = calculate_convection_coefficient(
            geometry=conv_geometry,
            characteristic_dimension=diameter if diameter else (length or width or 1.0),
            fluid_name=fluid_name_external,
            bulk_fluid_temperature=T_ambient_K,
            surface_temperature=Ts_outer_K_guess,  # Initial guess
            pressure=101325.0,  # Standard pressure
            flow_type='forced',
            fluid_velocity=wind_speed
        )
        conv_coeff_data = json.loads(conv_coeff_json)
        
        if "error" in conv_coeff_data:
            return json.dumps({
                "error": f"Failed to calculate external convection coefficient: {conv_coeff_data['error']}"
            })
        
        h_outer = conv_coeff_data.get("convection_coefficient_h")
        
        # Calculate internal thermal resistance if wall_layers provided
        if wall_layers is not None:
            # Need to determine internal h and wall resistance
            # For simplicity, use typical values for internal convection
            if fluid_name_internal.lower() == "water":
                h_inner_assumed = 1000.0  # W/m²K (typical for water)
            else:
                h_inner_assumed = 100.0  # W/m²K (moderate value for other fluids)
            
            # Calculate thermal resistance between internal fluid and outer surface
            if 'cylinder' in geometry_lower:
                # Use overall_heat_transfer tool for cylindrical calculation
                # Need inner diameter
                inner_diameter = dimensions.get('inner_diameter')
                if inner_diameter is None:
                    # Estimate inner diameter from outer diameter and wall thickness
                    total_wall_thickness = sum(layer.get('thickness', 0) for layer in wall_layers)
                    inner_diameter = diameter - 2 * total_wall_thickness
                
                u_value_json = calculate_overall_heat_transfer_coefficient(
                    geometry='cylinder',
                    layers=wall_layers,
                    inner_convection_coefficient_h=h_inner_assumed,
                    outer_convection_coefficient_h=h_outer,
                    inner_diameter=inner_diameter,
                    outer_diameter=diameter
                )
                u_value_data = json.loads(u_value_json)
                
                if "error" in u_value_data:
                    return json.dumps({
                        "error": f"Failed to calculate overall heat transfer coefficient: {u_value_data['error']}"
                    })
                
                # Use the calculated U-value
                R_internal_plus_wall = 1.0 / u_value_data.get("overall_heat_transfer_coefficient_U_outer", 0)
                
            else:  # Flat wall
                # Calculate flat wall resistance
                R_wall_total = 0.0
                for layer in wall_layers:
                    thickness = layer.get('thickness')
                    k = layer.get('thermal_conductivity_k')
                    material_name = layer.get('material_name')
                    
                    if k is None and material_name:
                        # Look up material properties
                        material_props_json = get_material_properties(material_name)
                        material_props = json.loads(material_props_json)
                        
                        if "error" in material_props:
                            return json.dumps({
                                "error": f"Failed to get material properties: {material_props['error']}"
                            })
                        
                        k = material_props.get("thermal_conductivity_k")
                    
                    if thickness is None or k is None:
                        return json.dumps({
                            "error": f"Missing thickness or thermal conductivity for wall layer."
                        })
                    
                    R_wall_total += thickness / k
                
                # Total resistance is sum of internal convection and wall conduction
                R_internal_plus_wall = 1.0 / h_inner_assumed + R_wall_total
        else:
            # Use the provided U-value directly
            R_internal_plus_wall = 1.0 / overall_heat_transfer_coefficient_U
        
        # Iteration to find Ts_outer and heat transfer rates
        for i in range(max_iterations):
            # Heat flow FROM internal fluid TO outer surface
            Q_cond = (T_internal_K - Ts_outer_K_guess) / (R_internal_plus_wall / outer_surface_area)
            
            # Heat flow FROM outer surface TO ambient
            Q_conv_out = h_outer * outer_surface_area * (Ts_outer_K_guess - T_ambient_K)
            Q_rad_out = calculate_radiation_heat_transfer(
                surface_emissivity, outer_surface_area, Ts_outer_K_guess, T_sky_K
            )
            
            # Solar gain if included
            Q_solar_in = incident_solar_radiation * surface_absorptivity * outer_surface_area if include_solar_gain else 0.0
            
            # Net heat flow out from surface
            Q_net_out = Q_conv_out + Q_rad_out - Q_solar_in
            
            # Energy balance difference
            balance_diff = Q_cond - Q_net_out
            
            # Check for convergence
            if abs(balance_diff) < tolerance * outer_surface_area:
                logger.info(f"Surface temperature converged after {i+1} iterations: {Ts_outer_K_guess:.2f} K")
                break
            
            # Update surface temperature estimate
            # Use dampening factor to prevent oscillations
            adjustment_factor = 0.1
            # Approximate derivatives for Newton's method
            dQ_cond_dTs = -outer_surface_area / R_internal_plus_wall
            dQ_out_dTs = h_outer * outer_surface_area + 4 * surface_emissivity * STEFAN_BOLTZMANN * outer_surface_area * Ts_outer_K_guess**3
            
            # Update using Newton's method with dampening
            Ts_outer_K_guess -= adjustment_factor * balance_diff / (dQ_cond_dTs - dQ_out_dTs)
            
            # Clamp temperature to reasonable bounds
            Ts_outer_K_guess = max(T_ambient_K - 50, min(T_internal_K + 50, Ts_outer_K_guess))
            
            # Log iteration results
            result_log.append({
                "iteration": i+1,
                "surface_temp_K": Ts_outer_K_guess,
                "Q_cond_W": Q_cond,
                "Q_conv_W": Q_conv_out,
                "Q_rad_W": Q_rad_out,
                "Q_solar_W": Q_solar_in,
                "balance_diff_W": balance_diff
            })
        
        # Final calculation with converged (or last) surface temperature
        Ts_outer_final = Ts_outer_K_guess
        Q_cond_final = (T_internal_K - Ts_outer_final) / (R_internal_plus_wall / outer_surface_area)
        Q_conv_final = h_outer * outer_surface_area * (Ts_outer_final - T_ambient_K)
        Q_rad_final = calculate_radiation_heat_transfer(
            surface_emissivity, outer_surface_area, Ts_outer_final, T_sky_K
        )
        Q_solar_final = incident_solar_radiation * surface_absorptivity * outer_surface_area if include_solar_gain else 0.0
        Q_total_final = Q_conv_final + Q_rad_final - Q_solar_final
        
        # Create result
        result = {
            "total_heat_rate_loss_w": Q_total_final,
            "convective_heat_rate_w": Q_conv_final,
            "radiative_heat_rate_w": Q_rad_final,
            "solar_gain_rate_w": Q_solar_final,
            "conduction_heat_rate_w": Q_cond_final,
            "estimated_outer_surface_temp_k": Ts_outer_final,
            "estimated_outer_surface_temp_c": Ts_outer_final - 273.15,
            "outer_surface_area_m2": outer_surface_area,
            "external_convection_coefficient_w_m2k": h_outer,
            "internal_plus_wall_resistance_k_w": R_internal_plus_wall / outer_surface_area,
            "geometry": geometry,
            "converged": i < max_iterations - 1,
            "iterations_required": i + 1,
            "sky_temperature_k": T_sky_K,
            "iteration_log": result_log[-5:] if len(result_log) > 5 else result_log  # Show last 5 iterations
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in calculate_surface_heat_transfer: {e}", exc_info=True)
        return json.dumps({
            "error": f"An unexpected error occurred: {str(e)}"
        })
