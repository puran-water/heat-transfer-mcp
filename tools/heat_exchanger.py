"""
Heat exchanger performance tool to analyze heat exchanger scenarios.

This module provides functionality to analyze heat exchangers using 
the NTU-effectiveness and LMTD methods.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

from utils.import_helpers import HT_AVAILABLE
from utils.helpers import calculate_lmtd, calculate_ntu_effectiveness
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.heat_exchanger")

def calculate_heat_exchanger_performance(
    hx_type: str = "shell_tube",
    flow_arrangement: str = None,
    hot_fluid_name: str = None,
    hot_fluid_flow_rate: float = None,
    hot_fluid_inlet_temp: float = None,
    cold_fluid_name: str = None,
    cold_fluid_flow_rate: float = None,
    cold_fluid_inlet_temp: float = None,
    overall_heat_transfer_coefficient_U: float = None,
    heat_transfer_area: float = None,
    hot_fluid_pressure: float = 101325.0,
    cold_fluid_pressure: float = 101325.0,
) -> str:
    """Analyzes simple heat exchanger scenarios using NTU or LMTD methods.
    
    Args:
        hx_type: Type of heat exchanger (e.g., 'shell_tube', 'double_pipe', 'plate')
        flow_arrangement: Flow arrangement (e.g., 'counterflow', 'parallelflow')
        hot_fluid_name: Name of the hot fluid
        hot_fluid_flow_rate: Mass flow rate of the hot fluid (kg/s)
        hot_fluid_inlet_temp: Inlet temperature of the hot fluid (K)
        cold_fluid_name: Name of the cold fluid
        cold_fluid_flow_rate: Mass flow rate of the cold fluid (kg/s)
        cold_fluid_inlet_temp: Inlet temperature of the cold fluid (K)
        overall_heat_transfer_coefficient_U: Overall heat transfer coefficient (W/m²K)
        heat_transfer_area: Total heat transfer area (m²)
        hot_fluid_pressure: Pressure of hot fluid (Pa)
        cold_fluid_pressure: Pressure of cold fluid (Pa)
        
    Returns:
        JSON string with heat exchanger performance results
    """
    try:
        # Validate required inputs
        required_inputs = [
            flow_arrangement, hot_fluid_name, hot_fluid_flow_rate, hot_fluid_inlet_temp,
            cold_fluid_name, cold_fluid_flow_rate, cold_fluid_inlet_temp,
            overall_heat_transfer_coefficient_U, heat_transfer_area
        ]
        
        if None in required_inputs:
            return json.dumps({
                "error": "All required inputs must be provided for heat exchanger analysis."
            })
        
        # 1. Get fluid properties for hot fluid
        hot_fluid_props_json = get_fluid_properties(hot_fluid_name, hot_fluid_inlet_temp, hot_fluid_pressure)
        hot_fluid_props = json.loads(hot_fluid_props_json)
        
        if "error" in hot_fluid_props:
            return json.dumps({
                "error": f"Failed to get hot fluid properties: {hot_fluid_props['error']}"
            })
        
        # 2. Get fluid properties for cold fluid
        cold_fluid_props_json = get_fluid_properties(cold_fluid_name, cold_fluid_inlet_temp, cold_fluid_pressure)
        cold_fluid_props = json.loads(cold_fluid_props_json)
        
        if "error" in cold_fluid_props:
            return json.dumps({
                "error": f"Failed to get cold fluid properties: {cold_fluid_props['error']}"
            })
        
        # 3. Extract specific heat capacities
        hot_fluid_cp = hot_fluid_props.get("specific_heat_cp")
        cold_fluid_cp = cold_fluid_props.get("specific_heat_cp")
        
        if hot_fluid_cp is None or cold_fluid_cp is None:
            return json.dumps({
                "error": "Could not determine specific heat capacity for one or both fluids."
            })
        
        # 4. Calculate heat capacity rates
        C_hot = hot_fluid_flow_rate * hot_fluid_cp
        C_cold = cold_fluid_flow_rate * cold_fluid_cp
        
        # 5. Determine C_min and C_max
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_r = C_min / C_max if C_max > 0 else 0  # Capacity ratio
        
        # 6. Calculate NTU (Number of Transfer Units)
        NTU = overall_heat_transfer_coefficient_U * heat_transfer_area / C_min if C_min > 0 else 0
        
        # 7. Calculate effectiveness based on flow arrangement
        effectiveness = calculate_ntu_effectiveness(NTU, C_r, flow_arrangement)
        
        # 8. Calculate heat duty
        max_possible_heat_transfer = C_min * (hot_fluid_inlet_temp - cold_fluid_inlet_temp)
        heat_duty = effectiveness * max_possible_heat_transfer
        
        # 9. Calculate outlet temperatures
        hot_fluid_outlet_temp = hot_fluid_inlet_temp - heat_duty / C_hot if C_hot > 0 else hot_fluid_inlet_temp
        cold_fluid_outlet_temp = cold_fluid_inlet_temp + heat_duty / C_cold if C_cold > 0 else cold_fluid_inlet_temp
        
        # 10. Calculate LMTD for verification/alternative calculation
        lmtd = None
        try:
            lmtd = calculate_lmtd(
                hot_fluid_inlet_temp, hot_fluid_outlet_temp,
                cold_fluid_inlet_temp, cold_fluid_outlet_temp,
                flow_arrangement
            )
        except Exception as lmtd_error:
            logger.warning(f"Could not calculate LMTD: {lmtd_error}")
        
        # Create result
        result = {
            "heat_duty_w": heat_duty,
            "hot_fluid_outlet_temp_k": hot_fluid_outlet_temp,
            "cold_fluid_outlet_temp_k": cold_fluid_outlet_temp,
            "effectiveness": effectiveness,
            "ntu": NTU,
            "capacity_ratio_cr": C_r,
            "c_hot_w_k": C_hot,
            "c_cold_w_k": C_cold,
            "c_min_w_k": C_min,
            "c_max_w_k": C_max,
            "log_mean_temperature_difference_k": lmtd,
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
            "heat_exchanger": {
                "type": hx_type,
                "flow_arrangement": flow_arrangement,
                "u_value_w_m2k": overall_heat_transfer_coefficient_U,
                "area_m2": heat_transfer_area
            }
        }
        
        # Add Celsius temperatures for convenience
        result["hot_fluid"]["inlet_temp_c"] = hot_fluid_inlet_temp - 273.15
        result["hot_fluid"]["outlet_temp_c"] = hot_fluid_outlet_temp - 273.15
        result["cold_fluid"]["inlet_temp_c"] = cold_fluid_inlet_temp - 273.15
        result["cold_fluid"]["outlet_temp_c"] = cold_fluid_outlet_temp - 273.15
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in calculate_heat_exchanger_performance: {e}", exc_info=True)
        return json.dumps({
            "error": f"An unexpected error occurred: {str(e)}"
        })
