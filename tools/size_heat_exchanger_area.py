"""
Heat exchanger sizing tool.

This module provides functionality to calculate the required heat transfer area (A), 
overall U-value, and LMTD for a heat exchanger given thermal duty and fluid conditions.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

from utils.import_helpers import HT_AVAILABLE
from utils.helpers import calculate_lmtd
from tools.fluid_properties import get_fluid_properties
from tools.convection_coefficient import calculate_convection_coefficient
from tools.hx_shell_side_h_kern import calculate_hx_shell_side_h_kern

logger = logging.getLogger("heat-transfer-mcp.size_heat_exchanger")

def size_heat_exchanger_area(
    required_heat_duty_q: float,
    hot_fluid_name: str,
    hot_fluid_flow_rate: float,
    hot_fluid_inlet_temp: float,
    hot_fluid_outlet_temp: Optional[float] = None,
    cold_fluid_name: str = None,
    cold_fluid_flow_rate: float = None,
    cold_fluid_inlet_temp: float = None,
    cold_fluid_outlet_temp: Optional[float] = None,
    flow_arrangement: str = None,
    shells: int = 1,
    tube_side_h: Optional[float] = None,
    shell_side_h: Optional[float] = None,
    tube_outer_diameter: Optional[float] = None,
    tube_inner_diameter: Optional[float] = None,
    tube_material_conductivity: Optional[float] = None,
    fouling_factor_inner: float = 0.0,
    fouling_factor_outer: float = 0.0,
    hot_fluid_pressure: float = 101325.0,
    cold_fluid_pressure: float = 101325.0,
    tube_roughness: float = 0.0,
    hx_length_per_pass: Optional[float] = None,
    shell_geometry_params: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> str:
    """Calculates required HX Area, U, LMTD based on Q and fluid conditions.
    
    Args:
        required_heat_duty_q: The target heat transfer rate (Q) in Watts
        hot_fluid_name: Name of the hot fluid
        hot_fluid_flow_rate: Mass flow rate (kg/s)
        hot_fluid_inlet_temp: Inlet temperature (K)
        hot_fluid_outlet_temp: Outlet temperature (K), calculated if not provided
        cold_fluid_name: Name of the cold fluid
        cold_fluid_flow_rate: Mass flow rate (kg/s)
        cold_fluid_inlet_temp: Inlet temperature (K)
        cold_fluid_outlet_temp: Outlet temperature (K), calculated if not provided
        flow_arrangement: Flow arrangement ('counterflow', 'parallelflow', etc.)
        shells: Number of shell passes for LMTD correction factor (default: 1)
        tube_side_h: Tube-side convection coefficient (hi) in W/m²K
        shell_side_h: Shell-side convection coefficient (ho) in W/m²K
        tube_outer_diameter: Tube outer diameter (m)
        tube_inner_diameter: Tube inner diameter (m)
        tube_material_conductivity: Thermal conductivity of tube material (kW) in W/mK
        fouling_factor_inner: Fouling resistance on the inner tube surface (Rfi) in m²K/W
        fouling_factor_outer: Fouling resistance on the outer tube surface (Rfo) in m²K/W
        hot_fluid_pressure: Pressure (Pa)
        cold_fluid_pressure: Pressure (Pa)
        tube_roughness: Tube inner surface roughness (m) for hi calculation
        hx_length_per_pass: Approximate length per pass (m) for internal h calculation
        shell_geometry_params: Parameters for shell-side h calculation if needed
        
    Returns:
        JSON string with heat exchanger sizing results
    """
    try:
        # Validate required inputs
        required_base_inputs = [
            required_heat_duty_q, hot_fluid_name, hot_fluid_flow_rate, hot_fluid_inlet_temp,
            cold_fluid_name, cold_fluid_flow_rate, cold_fluid_inlet_temp, flow_arrangement
        ]
        
        if None in required_base_inputs:
            return json.dumps({
                "error": "Basic inputs (heat duty, fluid names, flow rates, inlet temps, flow arrangement) are required."
            })
            
        if required_heat_duty_q == 0:
            return json.dumps({
                "error": "Required heat duty Q must be non-zero."
            })

        # Dictionary to track calculated temperatures
        calculated_temps = {}
        
        # 1. Determine all four temperatures
        # Get fluid properties for specific heat calculation
        hot_fluid_props_json = get_fluid_properties(hot_fluid_name, hot_fluid_inlet_temp, hot_fluid_pressure, strict=strict)
        hot_fluid_props = json.loads(hot_fluid_props_json)
        
        if "error" in hot_fluid_props:
            return json.dumps({
                "error": f"Failed to get hot fluid properties: {hot_fluid_props['error']}"
            })
            
        cold_fluid_props_json = get_fluid_properties(cold_fluid_name, cold_fluid_inlet_temp, cold_fluid_pressure, strict=strict)
        cold_fluid_props = json.loads(cold_fluid_props_json)
        
        if "error" in cold_fluid_props:
            return json.dumps({
                "error": f"Failed to get cold fluid properties: {cold_fluid_props['error']}"
            })
            
        # Extract specific heat capacities
        hot_fluid_cp = hot_fluid_props.get("specific_heat_cp")
        cold_fluid_cp = cold_fluid_props.get("specific_heat_cp")
        
        if hot_fluid_cp is None or cold_fluid_cp is None:
            return json.dumps({
                "error": "Could not determine specific heat capacity for one or both fluids."
            })
            
        # Calculate missing outlet temps if needed
        if hot_fluid_outlet_temp is None:
            hot_fluid_outlet_temp = hot_fluid_inlet_temp - required_heat_duty_q / (hot_fluid_flow_rate * hot_fluid_cp)
            calculated_temps["hot_outlet_temp_k"] = hot_fluid_outlet_temp
            
        if cold_fluid_outlet_temp is None:
            cold_fluid_outlet_temp = cold_fluid_inlet_temp + required_heat_duty_q / (cold_fluid_flow_rate * cold_fluid_cp)
            calculated_temps["cold_outlet_temp_k"] = cold_fluid_outlet_temp
            
        # Verify energy balance
        Q_check_hot = hot_fluid_flow_rate * hot_fluid_cp * (hot_fluid_inlet_temp - hot_fluid_outlet_temp)
        Q_check_cold = cold_fluid_flow_rate * cold_fluid_cp * (cold_fluid_outlet_temp - cold_fluid_inlet_temp)
        
        if not math.isclose(Q_check_hot, required_heat_duty_q, rel_tol=0.01) or \
           not math.isclose(Q_check_cold, required_heat_duty_q, rel_tol=0.01):
            logger.warning(f"Energy balance check: Q_req={required_heat_duty_q:.1f}, "
                          f"Q_hot={Q_check_hot:.1f}, Q_cold={Q_check_cold:.1f}")
            
        # 2. Calculate LMTD
        try:
            lmtd = calculate_lmtd(
                hot_fluid_inlet_temp, hot_fluid_outlet_temp,
                cold_fluid_inlet_temp, cold_fluid_outlet_temp,
                flow_arrangement
            )
        except Exception as e:
            return json.dumps({
                "error": f"Could not calculate LMTD: {str(e)}. Check inlet/outlet temperatures for crossover."
            })
            
        # 3. Calculate LMTD Correction Factor (Ft) using ht library
        Ft = 1.0  # Default for counterflow/parallel
        
        # For shell-and-tube heat exchangers, use ht library for accurate Ft calculation
        if flow_arrangement.lower() not in ['counterflow', 'parallelflow']:
            try:
                # Use ht.hx.F_LMTD_Fakheri for shell-and-tube LMTD correction factor
                if HT_AVAILABLE and 'shell_tube' in flow_arrangement.lower():
                    from ht.hx import F_LMTD_Fakheri
                    
                    # F_LMTD_Fakheri(Thi, Tho, Tci, Tco, shells)
                    Ft = F_LMTD_Fakheri(
                        Thi=hot_fluid_inlet_temp,
                        Tho=hot_fluid_outlet_temp,
                        Tci=cold_fluid_inlet_temp,
                        Tco=cold_fluid_outlet_temp,
                        shells=shells
                    )
                    logger.info(f"Used ht.F_LMTD_Fakheri: Ft={Ft:.4f} for {flow_arrangement} with {shells} shell(s)")
                    
                else:
                    # Fallback to simplified approximation if ht not available
                    # Calculate P and R parameters
                    P = abs(cold_fluid_outlet_temp - cold_fluid_inlet_temp) / \
                        abs(hot_fluid_inlet_temp - cold_fluid_inlet_temp)
                    R = abs(hot_fluid_inlet_temp - hot_fluid_outlet_temp) / \
                        abs(cold_fluid_outlet_temp - cold_fluid_inlet_temp)
                    
                    # Simple F-factor approximation for shell & tube with 1 shell pass
                    if 'shell_tube' in flow_arrangement.lower():
                        # Very approximate formula for 1-shell pass, 2 or more tube passes
                        Z = (P - 1) / (P * R - 1) if (P * R - 1) != 0 else 0
                        Ft = math.sqrt(R**2 + 1) / (R - 1) * math.log((1 - P) / (1 - P * R)) / math.log((2 - P * (R + 1 - math.sqrt(R**2 + 1))) / (2 - P * (R + 1 + math.sqrt(R**2 + 1)))) if Z != 0 else 1.0
                        
                        # Constrain to reasonable values
                        Ft = max(0.75, min(Ft, 1.0))
                    
                    logger.info(f"Used fallback approximation: Ft={Ft:.3f} for {flow_arrangement}")
                    
            except Exception as e:
                logger.warning(f"Failed to calculate Ft with ht library: {e}. Using fallback approximation.")
                # Fallback calculation (same as above)
                P = abs(cold_fluid_outlet_temp - cold_fluid_inlet_temp) / \
                    abs(hot_fluid_inlet_temp - cold_fluid_inlet_temp)
                R = abs(hot_fluid_inlet_temp - hot_fluid_outlet_temp) / \
                    abs(cold_fluid_outlet_temp - cold_fluid_inlet_temp)
                
                if 'shell_tube' in flow_arrangement.lower():
                    Z = (P - 1) / (P * R - 1) if (P * R - 1) != 0 else 0
                    Ft = math.sqrt(R**2 + 1) / (R - 1) * math.log((1 - P) / (1 - P * R)) / math.log((2 - P * (R + 1 - math.sqrt(R**2 + 1))) / (2 - P * (R + 1 + math.sqrt(R**2 + 1)))) if Z != 0 else 1.0
                    Ft = max(0.75, min(Ft, 1.0))
            
        # 4. Calculate convection coefficients if not provided
        h_calc_details = {}
        
        # Calculate tube-side h (internal) if not provided
        if tube_side_h is None:
            if tube_inner_diameter is None:
                return json.dumps({
                    "error": "tube_inner_diameter is required to calculate tube-side h"
                })
                
            # Determine which fluid is inside tubes (typically the one with higher pressure or 
            # that is more corrosive or needs higher velocity)
            # For simplicity, assuming cold fluid in tubes for now
            tube_fluid_name = cold_fluid_name
            tube_fluid_flow_rate = cold_fluid_flow_rate
            tube_fluid_bulk_temp = (cold_fluid_inlet_temp + cold_fluid_outlet_temp) / 2.0
            tube_fluid_pressure = cold_fluid_pressure
            
            # Estimate wall temperature
            hot_fluid_bulk_temp = (hot_fluid_inlet_temp + hot_fluid_outlet_temp) / 2.0
            tube_wall_temp_est = (tube_fluid_bulk_temp + hot_fluid_bulk_temp) / 2.0
            
            # Calculate velocity if possible
            tube_fluid_props = json.loads(get_fluid_properties(tube_fluid_name, tube_fluid_bulk_temp, tube_fluid_pressure, strict=strict))
            tube_fluid_density = tube_fluid_props.get("density")
            
            if tube_fluid_density is None:
                return json.dumps({
                    "error": "Could not get tube fluid density for velocity calculation"
                })
                
            tube_cross_area = math.pi * (tube_inner_diameter**2) / 4.0
            tube_fluid_velocity = tube_fluid_flow_rate / (tube_fluid_density * tube_cross_area)
            
            # Use convection coefficient calculator
            tube_side_h_json = calculate_convection_coefficient(
                geometry='pipe_internal_forced',
                characteristic_dimension=tube_inner_diameter,
                fluid_name=tube_fluid_name,
                bulk_fluid_temperature=tube_fluid_bulk_temp,
                surface_temperature=tube_wall_temp_est,
                pressure=tube_fluid_pressure,
                flow_type='forced',
                fluid_velocity=tube_fluid_velocity,
                roughness=tube_roughness,
                pipe_length=hx_length_per_pass,
                strict=strict
            )
            
            tube_side_h_result = json.loads(tube_side_h_json)
            
            if "error" in tube_side_h_result:
                return json.dumps({
                    "error": f"Failed to calculate tube-side h: {tube_side_h_result['error']}"
                })
                
            tube_side_h = tube_side_h_result.get("convection_coefficient_h_W_m2K", tube_side_h_result.get("convection_coefficient_h"))
            h_calc_details["calculated_hi"] = tube_side_h
            h_calc_details["hi_details"] = tube_side_h_result
            
        # Calculate shell-side h (external) if not provided
        if shell_side_h is None:
            if tube_outer_diameter is None:
                return json.dumps({
                    "error": "tube_outer_diameter is required to calculate shell-side h"
                })
                
            if shell_geometry_params is None:
                return json.dumps({
                    "error": "shell_geometry_params required to calculate shell-side h"
                })
                
            # Setup and calculate shell-side h using Kern method
            shell_fluid_name = hot_fluid_name
            shell_fluid_flow_rate = hot_fluid_flow_rate
            shell_fluid_bulk_temp = (hot_fluid_inlet_temp + hot_fluid_outlet_temp) / 2.0
            shell_fluid_pressure = hot_fluid_pressure
            
            # Estimate tube wall temperature
            tube_wall_temp_est = (shell_fluid_bulk_temp + (cold_fluid_inlet_temp + cold_fluid_outlet_temp) / 2.0) / 2.0
            
            # Get required shell geometry params
            shell_inner_diameter = shell_geometry_params.get("shell_inner_diameter")
            tube_pitch = shell_geometry_params.get("tube_pitch")
            tube_layout_angle = shell_geometry_params.get("tube_layout_angle")
            baffle_spacing = shell_geometry_params.get("baffle_spacing")
            baffle_cut_percent = shell_geometry_params.get("baffle_cut_percent")
            tube_rows = shell_geometry_params.get("tube_rows")
            pitch_parallel = shell_geometry_params.get("pitch_parallel")
            pitch_normal = shell_geometry_params.get("pitch_normal")
            
            if None in [shell_inner_diameter, tube_pitch, tube_layout_angle, baffle_spacing, baffle_cut_percent]:
                return json.dumps({
                    "error": "Incomplete shell geometry parameters provided"
                })
                
            shell_side_h_json = calculate_hx_shell_side_h_kern(
                shell_inner_diameter=shell_inner_diameter,
                tube_outer_diameter=tube_outer_diameter,
                tube_pitch=tube_pitch,
                tube_layout_angle=tube_layout_angle,
                baffle_spacing=baffle_spacing,
                baffle_cut_percent=baffle_cut_percent,
                shell_fluid_name=shell_fluid_name,
                shell_fluid_flow_rate=shell_fluid_flow_rate,
                shell_fluid_bulk_temp=shell_fluid_bulk_temp,
                shell_fluid_pressure=shell_fluid_pressure,
                tube_wall_temp=tube_wall_temp_est,
                tube_rows=tube_rows,
                pitch_parallel=pitch_parallel,
                pitch_normal=pitch_normal
            )
            
            shell_side_h_result = json.loads(shell_side_h_json)
            
            if "error" in shell_side_h_result:
                return json.dumps({
                    "error": f"Failed to calculate shell-side h: {shell_side_h_result['error']}"
                })
                
            shell_side_h = shell_side_h_result.get("shell_side_h_W_m2K")
            h_calc_details["calculated_ho"] = shell_side_h
            h_calc_details["ho_details"] = shell_side_h_result
            
        # 5. Calculate overall heat transfer coefficient (U)
        # Need to verify all required parameters are available
        if None in [tube_side_h, shell_side_h, tube_outer_diameter, tube_inner_diameter, tube_material_conductivity]:
            return json.dumps({
                "error": "Cannot calculate U: Missing hi, ho, tube diameters, or tube conductivity."
            })
            
        # Calculate U based on outer area (Ao)
        term1 = tube_outer_diameter / (tube_inner_diameter * tube_side_h) if tube_inner_diameter * tube_side_h != 0 else float('inf')
        term2 = fouling_factor_inner * tube_outer_diameter / tube_inner_diameter if tube_inner_diameter != 0 else float('inf')  # scale Rfi
        term3 = tube_outer_diameter * math.log(tube_outer_diameter / tube_inner_diameter) / (2.0 * tube_material_conductivity) if tube_material_conductivity != 0 and tube_inner_diameter != 0 else float('inf')
        term4 = fouling_factor_outer
        term5 = 1.0 / shell_side_h if shell_side_h != 0 else float('inf')
        
        R_total_o = term1 + term2 + term3 + term4 + term5
        
        if R_total_o <= 0:
            return json.dumps({
                "error": "Calculated total thermal resistance is zero or negative."
            })
            
        U = 1.0 / R_total_o
        
        # 6. Calculate required area
        if U <= 0 or Ft <= 0 or lmtd <= 0:
            return json.dumps({
                "error": f"Cannot calculate area with invalid parameters: U={U:.3f}, Ft={Ft:.3f}, LMTD={lmtd:.3f}"
            })
            
        Area = abs(required_heat_duty_q) / (U * Ft * lmtd)
        
        # Prepare resistance details
        resistance_details = {
            "R_conv_i_scaled": term1,
            "R_foul_i_scaled": term2,
            "R_wall": term3,
            "R_foul_o": term4,
            "R_conv_o": term5,
            "R_total_o": R_total_o
        }
        
        # Create final result
        result = {
            "required_area_m2": Area,
            "calculated_overall_U_W_m2K": U,
            "log_mean_temp_diff_k": lmtd,
            "lmtd_correction_factor_Ft": Ft,
            "shells": shells,
            "flow_arrangement": flow_arrangement,
            "hot_fluid": {
                "name": hot_fluid_name,
                "flow_rate_kg_s": hot_fluid_flow_rate,
                "inlet_temp_k": hot_fluid_inlet_temp,
                "outlet_temp_k": hot_fluid_outlet_temp,
                "specific_heat_j_kgk": hot_fluid_cp
            },
            "cold_fluid": {
                "name": cold_fluid_name,
                "flow_rate_kg_s": cold_fluid_flow_rate,
                "inlet_temp_k": cold_fluid_inlet_temp,
                "outlet_temp_k": cold_fluid_outlet_temp,
                "specific_heat_j_kgk": cold_fluid_cp
            },
            "calculated_outlet_temps_k": calculated_temps,
            "convection_coefficient_details": h_calc_details,
            "total_resistance_details_Ao_basis": resistance_details,
            "heat_duty_q_watts": required_heat_duty_q
        }
        
        # Add Celsius temperatures for convenience
        result["hot_fluid"]["inlet_temp_c"] = hot_fluid_inlet_temp - 273.15
        result["hot_fluid"]["outlet_temp_c"] = hot_fluid_outlet_temp - 273.15
        result["cold_fluid"]["inlet_temp_c"] = cold_fluid_inlet_temp - 273.15
        result["cold_fluid"]["outlet_temp_c"] = cold_fluid_outlet_temp - 273.15
        
        if "hot_outlet_temp_k" in calculated_temps:
            calculated_temps["hot_outlet_temp_c"] = hot_fluid_outlet_temp - 273.15
        
        if "cold_outlet_temp_k" in calculated_temps:
            calculated_temps["cold_outlet_temp_c"] = cold_fluid_outlet_temp - 273.15

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in size_heat_exchanger_area: {e}", exc_info=True)
        return json.dumps({
            "error": f"An unexpected error occurred: {str(e)}"
        })
