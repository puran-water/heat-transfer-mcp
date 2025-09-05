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
from utils.validation import (
    ValidationError,
    require_non_negative,
    validate_geometry_dimensions,
)

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
    internal_convection_coefficient_h_override: Optional[float] = None,
    view_factor_sky_vertical: float = 0.5,  # Vertical walls see 50% sky
    view_factor_sky_horizontal: float = 1.0,  # Roof sees 100% sky
    ground_temperature: Optional[float] = None,  # Default to ambient
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
    # Basic input validation (allow no layers/U => internal convection only)
    try:
        require_non_negative(float(wind_speed), "wind_speed")
        validate_geometry_dimensions(geometry, dimensions)
    except ValidationError as ve:
        return json.dumps({"error": str(ve)})

    if include_solar_gain and incident_solar_radiation is None:
        return json.dumps({
            "error": "Incident solar radiation is required when include_solar_gain is True."
        })
    
    try:
        # 1. Calculate Surface Area
        outer_surface_area = 0.0
        lateral_area = 0.0
        top_area = 0.0
        geometry_lower = geometry.lower()
        diameter = dimensions.get('diameter')
        height = dimensions.get('height')
        length = dimensions.get('length')
        width = dimensions.get('width')
        
        if ('cylinder_tank' in geometry_lower or 'tank' in geometry_lower) and diameter and height:
            if 'vertical' in geometry_lower:
                # Vertical cylinder tanks: walls + top cap only (bottom is ground-contact)
                lateral_area = math.pi * diameter * height
                top_area = math.pi * (diameter/2)**2
                outer_surface_area = lateral_area + top_area
            elif 'horizontal' in geometry_lower:
                # Horizontal cylinder
                outer_surface_area = math.pi * diameter * length + 2 * math.pi * (diameter/2)**2
            else:
                # Default to vertical if not specified - includes both endcaps
                outer_surface_area = math.pi * diameter * height + 2 * math.pi * (diameter/2)**2
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
        tolerance_W = 1.0  # Watts - dimensionally consistent tolerance
        damping_factor = 0.5  # Adaptive damping
        result_log = []  # To track convergence
        
        # Determine convection geometry for external surface
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
        
        # Initial calculation of external convection coefficient
        # Will be updated during iteration
        
        # Determine internal convection estimate regardless of wall specification
        if internal_convection_coefficient_h_override is not None:
            h_inner_assumed = float(internal_convection_coefficient_h_override)
        else:
            if fluid_name_internal.lower() == "water":
                h_inner_assumed = 1000.0  # W/m²K (typical for water)
            else:
                h_inner_assumed = 100.0  # W/m²K (moderate value for other fluids)

        # Calculate internal thermal resistance path
        if overall_heat_transfer_coefficient_U is not None:
            # Use the provided U-value directly
            R_internal_plus_wall = 1.0 / float(overall_heat_transfer_coefficient_U)
        else:
            # wall_layers may be None or empty -> treat as no wall conduction
            if wall_layers and len(wall_layers) > 0:
                if 'cylinder' in geometry_lower or 'pipe' in geometry_lower:
                    # Use overall_heat_transfer tool for cylindrical calculation excluding external convection
                    inner_diameter = dimensions.get('inner_diameter')
                    if inner_diameter is None:
                        total_wall_thickness = sum(layer.get('thickness', 0) for layer in wall_layers)
                        inner_diameter = diameter - 2 * total_wall_thickness

                    u_value_json = calculate_overall_heat_transfer_coefficient(
                        geometry='cylinder',
                        layers=wall_layers,
                        inner_convection_coefficient_h=h_inner_assumed,
                        outer_convection_coefficient_h=1e10,  # exclude external resistance
                        inner_diameter=inner_diameter,
                        outer_diameter=diameter
                    )
                    u_value_data = json.loads(u_value_json)
                    if "error" in u_value_data:
                        return json.dumps({
                            "error": f"Failed to calculate overall heat transfer coefficient: {u_value_data['error']}"
                        })
                    R_conv_inner_per_L = u_value_data.get('convection_resistance_inner_per_length_mk_w', 0)
                    R_cond_total_per_L = u_value_data.get('conduction_resistance_total_per_length_mk_w', 0)
                    r_outer = diameter / 2
                    A_out_per_L = 2 * math.pi * r_outer
                    # Convert per-length resistance (K·m/W) to area-based (K·m²/W) by multiplying by outer area per unit length
                    R_internal_plus_wall = (R_conv_inner_per_L + R_cond_total_per_L) * A_out_per_L
                else:
                    # Flat wall path
                    R_wall_total = 0.0
                    for layer in wall_layers:
                        thickness = layer.get('thickness')
                        k = layer.get('thermal_conductivity_k')
                        material_name = layer.get('material_name')
                        if k is None and material_name:
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
                    R_internal_plus_wall = 1.0 / h_inner_assumed + R_wall_total
            else:
                # No layers provided -> internal convection only
                R_internal_plus_wall = 1.0 / h_inner_assumed
        
        # Iteration to find Ts_outer and heat transfer rates
        prev_balance_diff = float('inf')
        for i in range(max_iterations):
            # Calculate external convection coefficient with current surface temperature
            # Use natural convection for very low wind speeds
            flow_type = 'natural' if wind_speed <= 0.2 else 'forced'
            
            # Use film temperature for better accuracy
            T_film = 0.5 * (Ts_outer_K_guess + T_ambient_K)
            
            conv_coeff_json = calculate_convection_coefficient(
                geometry=conv_geometry,
                characteristic_dimension=diameter if diameter else (length or width or 1.0),
                fluid_name=fluid_name_external,
                bulk_fluid_temperature=T_film,  # Use film temperature
                surface_temperature=Ts_outer_K_guess,
                pressure=101325.0,  # Standard pressure
                flow_type=flow_type,
                fluid_velocity=wind_speed if flow_type == 'forced' else None
            )
            conv_coeff_data = json.loads(conv_coeff_json)
            
            if "error" in conv_coeff_data:
                logger.warning(f"Failed to update h: {conv_coeff_data.get('error')}. Using previous value.")
                if i == 0:
                    h_outer = 10.0  # Default fallback value
            else:
                h_outer = conv_coeff_data.get("convection_coefficient_h", 10.0)
            # Heat flow FROM internal fluid TO outer surface
            Q_cond = (T_internal_K - Ts_outer_K_guess) / (R_internal_plus_wall / outer_surface_area)
            
            # Heat flow FROM outer surface TO ambient
            Q_conv_out = h_outer * outer_surface_area * (Ts_outer_K_guess - T_ambient_K)
            
            # Calculate effective radiation temperature with view factors
            T_ground_K = ground_temperature if ground_temperature is not None else T_ambient_K
            
            if 'vertical' in geometry_lower and 'cylinder' in geometry_lower and lateral_area > 0 and top_area > 0:
                # Separate areas for different view factors
                # Effective temps for each surface
                T_eff4_lateral = (view_factor_sky_vertical * T_sky_K**4 + 
                                 (1-view_factor_sky_vertical) * T_ground_K**4)
                T_eff4_top = (view_factor_sky_horizontal * T_sky_K**4 + 
                             (1-view_factor_sky_horizontal) * T_ground_K**4)
                
                # Combined radiation
                Q_rad_lateral = surface_emissivity * STEFAN_BOLTZMANN * lateral_area * (Ts_outer_K_guess**4 - T_eff4_lateral)
                Q_rad_top = surface_emissivity * STEFAN_BOLTZMANN * top_area * (Ts_outer_K_guess**4 - T_eff4_top)
                Q_rad_out = Q_rad_lateral + Q_rad_top
                
                # Store effective temperature for reporting
                T_eff4_combined = (lateral_area * T_eff4_lateral + top_area * T_eff4_top) / outer_surface_area
                T_eff_radiation = T_eff4_combined ** 0.25
            else:
                # Simple case - single view factor
                view_factor_sky = view_factor_sky_horizontal if 'horizontal' in geometry_lower else view_factor_sky_vertical
                T_eff4 = view_factor_sky * T_sky_K**4 + (1-view_factor_sky) * T_ground_K**4
                T_eff_radiation = T_eff4 ** 0.25
                Q_rad_out = surface_emissivity * STEFAN_BOLTZMANN * outer_surface_area * (Ts_outer_K_guess**4 - T_eff4)
            
            # Solar gain if included
            Q_solar_in = incident_solar_radiation * surface_absorptivity * outer_surface_area if include_solar_gain else 0.0
            
            # Net heat flow out from surface
            Q_net_out = Q_conv_out + Q_rad_out - Q_solar_in
            
            # Energy balance difference
            balance_diff = Q_cond - Q_net_out
            
            # Check for convergence (dimensionally consistent)
            if abs(balance_diff) < tolerance_W:
                logger.info(f"Surface temperature converged after {i+1} iterations: {Ts_outer_K_guess:.2f} K")
                break
            
            # Adaptive damping - reduce if diverging
            if abs(balance_diff) > abs(prev_balance_diff):
                damping_factor *= 0.5  # Reduce damping if diverging
                damping_factor = max(0.01, damping_factor)  # Keep minimum damping
            
            # Update surface temperature estimate
            # Approximate derivatives for Newton's method
            dQ_cond_dTs = -outer_surface_area / R_internal_plus_wall
            
            # Linearize radiation for stability with view factors
            if 'T_eff_radiation' in locals():
                T_mean_rad = 0.5 * (Ts_outer_K_guess + T_eff_radiation)
            else:
                T_mean_rad = 0.5 * (Ts_outer_K_guess + T_sky_K)
            h_rad_linear = 4 * surface_emissivity * STEFAN_BOLTZMANN * T_mean_rad**3
            
            dQ_out_dTs = h_outer * outer_surface_area + h_rad_linear * outer_surface_area
            
            # Update using Newton's method with adaptive damping
            Ts_outer_K_new = Ts_outer_K_guess - damping_factor * balance_diff / (dQ_cond_dTs - dQ_out_dTs)
            
            # Apply damped update
            Ts_outer_K_guess = damping_factor * Ts_outer_K_new + (1 - damping_factor) * Ts_outer_K_guess
            
            prev_balance_diff = balance_diff
            
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
        
        # Recalculate radiation with view factors for final result
        T_ground_K = ground_temperature if ground_temperature is not None else T_ambient_K
        if 'vertical' in geometry_lower and 'cylinder' in geometry_lower and lateral_area > 0 and top_area > 0:
            T_eff4_lateral = (view_factor_sky_vertical * T_sky_K**4 + 
                             (1-view_factor_sky_vertical) * T_ground_K**4)
            T_eff4_top = (view_factor_sky_horizontal * T_sky_K**4 + 
                         (1-view_factor_sky_horizontal) * T_ground_K**4)
            Q_rad_lateral = surface_emissivity * STEFAN_BOLTZMANN * lateral_area * (Ts_outer_final**4 - T_eff4_lateral)
            Q_rad_top = surface_emissivity * STEFAN_BOLTZMANN * top_area * (Ts_outer_final**4 - T_eff4_top)
            Q_rad_final = Q_rad_lateral + Q_rad_top
            T_eff4_combined = (lateral_area * T_eff4_lateral + top_area * T_eff4_top) / outer_surface_area
            T_eff_radiation_final = T_eff4_combined ** 0.25
        else:
            view_factor_sky = view_factor_sky_horizontal if 'horizontal' in geometry_lower else view_factor_sky_vertical
            T_eff4 = view_factor_sky * T_sky_K**4 + (1-view_factor_sky) * T_ground_K**4
            T_eff_radiation_final = T_eff4 ** 0.25
            Q_rad_final = surface_emissivity * STEFAN_BOLTZMANN * outer_surface_area * (Ts_outer_final**4 - T_eff4)
        
        Q_solar_final = incident_solar_radiation * surface_absorptivity * outer_surface_area if include_solar_gain else 0.0
        Q_total_final = Q_conv_final + Q_rad_final - Q_solar_final
        
        # Create result with transparency fields
        converged = i < max_iterations - 1
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
            "converged": converged,
            "iterations_required": i + 1,
            "sky_temperature_k": T_sky_K,
            "ground_temperature_k": T_ground_K,
            "radiation_model": {
                "view_factors": {
                    "sky_vertical": view_factor_sky_vertical,
                    "sky_horizontal": view_factor_sky_horizontal,
                    "ground_vertical": 1.0 - view_factor_sky_vertical,
                    "ground_horizontal": 1.0 - view_factor_sky_horizontal
                },
                "effective_env_temp_k": T_eff_radiation_final if 'T_eff_radiation_final' in locals() else T_sky_K
            },
            "iteration_log": result_log[-5:] if len(result_log) > 5 else result_log  # Show last 5 iterations
        }
        # Add warnings
        warnings = []
        if not converged:
            warnings.append(f"Surface temperature solver did not converge within {max_iterations} iterations. Results may be approximate.")
        if Ts_outer_final < T_ambient_K:
            warnings.append(f"Surface temperature ({Ts_outer_final-273.15:.1f}°C) is colder than ambient ({T_ambient_K-273.15:.1f}°C) due to radiative cooling")
        if warnings:
            result["warnings"] = warnings
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in calculate_surface_heat_transfer: {e}", exc_info=True)
        return json.dumps({
            "error": f"An unexpected error occurred: {str(e)}"
        })
