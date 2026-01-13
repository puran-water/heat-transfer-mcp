"""
Heat duty calculation tool to determine heat transfer rates.

This module provides functionality to calculate heat duty based on
sensible heat change or overall heat transfer methods.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

from utils.import_helpers import HT_AVAILABLE
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.heat_duty")


def calculate_heat_duty(
    calculation_method: str,
    fluid_name: Optional[str] = None,
    flow_rate: Optional[float] = None,
    inlet_temp: Optional[float] = None,
    outlet_temp: Optional[float] = None,
    specific_heat_cp: Optional[float] = None,
    fluid_pressure: float = 101325.0,
    overall_heat_transfer_coefficient_U: Optional[float] = None,
    heat_transfer_area: Optional[float] = None,
    mean_temperature_difference: Optional[float] = None,
) -> str:
    """Calculates heat duty based on specified method.

    Args:
        calculation_method: Method to use ('sensible_heat' or 'ua_deltaT')
        fluid_name: Fluid name (for 'sensible_heat')
        flow_rate: Mass flow rate in kg/s (for 'sensible_heat')
        inlet_temp: Inlet temperature in K (for 'sensible_heat')
        outlet_temp: Outlet temperature in K (for 'sensible_heat')
        specific_heat_cp: Specific heat capacity in J/kg·K (for 'sensible_heat')
        fluid_pressure: Fluid pressure in Pa (for property lookup)
        overall_heat_transfer_coefficient_U: U-value in W/m²K (for 'ua_deltaT')
        heat_transfer_area: Heat transfer area in m² (for 'ua_deltaT')
        mean_temperature_difference: LMTD or mean temperature difference in K (for 'ua_deltaT')

    Returns:
        JSON string with calculated heat duty and details
    """
    try:
        calculation_details = {}
        heat_duty = None

        # Process based on calculation method
        if calculation_method.lower() == "sensible_heat":
            # Validate required inputs for sensible heat
            if None in [flow_rate, inlet_temp, outlet_temp]:
                return json.dumps(
                    {"error": "For 'sensible_heat' method, 'flow_rate', 'inlet_temp', and 'outlet_temp' are required."}
                )

            # Get specific_heat_cp if not provided
            if specific_heat_cp is None:
                if fluid_name is None:
                    return json.dumps(
                        {"error": "For 'sensible_heat' method, provide either 'specific_heat_cp' or 'fluid_name' for lookup."}
                    )

                # Calculate average temperature for property lookup
                avg_temp = (inlet_temp + outlet_temp) / 2.0

                # Get fluid properties at average temperature
                fluid_props_json = get_fluid_properties(fluid_name, avg_temp, fluid_pressure)
                fluid_props = json.loads(fluid_props_json)

                if "error" in fluid_props:
                    return json.dumps({"error": f"Failed to get fluid properties: {fluid_props['error']}"})

                specific_heat_cp = fluid_props.get("specific_heat_cp")
                if specific_heat_cp is None:
                    return json.dumps({"error": f"Could not determine specific heat capacity for '{fluid_name}'."})

                # Record that Cp was looked up
                calculation_details["specific_heat_source"] = "Looked up for fluid"
                calculation_details["specific_heat_lookup_temp_k"] = avg_temp
            else:
                calculation_details["specific_heat_source"] = "User provided"

            # Calculate temperature difference
            delta_T = outlet_temp - inlet_temp

            # Calculate heat duty: Q = m * Cp * ΔT
            heat_duty = flow_rate * specific_heat_cp * delta_T

            # Record calculation details
            calculation_details["flow_rate_kg_s"] = flow_rate
            calculation_details["inlet_temp_k"] = inlet_temp
            calculation_details["outlet_temp_k"] = outlet_temp
            calculation_details["specific_heat_cp_j_kgk"] = specific_heat_cp
            calculation_details["temperature_difference_k"] = delta_T
            calculation_details["calculation_formula"] = "Q = m * Cp * ΔT"

        elif calculation_method.lower() == "ua_deltat":
            # Validate required inputs for UA-deltaT method
            if None in [overall_heat_transfer_coefficient_U, heat_transfer_area, mean_temperature_difference]:
                return json.dumps(
                    {
                        "error": "For 'ua_deltaT' method, 'overall_heat_transfer_coefficient_U', 'heat_transfer_area', and 'mean_temperature_difference' are required."
                    }
                )

            # Calculate heat duty: Q = U * A * ΔT
            heat_duty = overall_heat_transfer_coefficient_U * heat_transfer_area * mean_temperature_difference

            # Record calculation details
            calculation_details["overall_heat_transfer_coefficient_u_w_m2k"] = overall_heat_transfer_coefficient_U
            calculation_details["heat_transfer_area_m2"] = heat_transfer_area
            calculation_details["mean_temperature_difference_k"] = mean_temperature_difference
            calculation_details["calculation_formula"] = "Q = U * A * ΔT"

        else:
            return json.dumps(
                {"error": f"Invalid calculation_method: '{calculation_method}'. Choose 'sensible_heat' or 'ua_deltaT'."}
            )

        # Check if heat duty was calculated
        if heat_duty is None:
            return json.dumps({"error": "Could not calculate heat duty with the provided inputs."})

        # Create result
        result = {
            "heat_duty_watts": heat_duty,
            "heat_duty_kilowatts": heat_duty / 1000.0,
            "calculation_method": calculation_method,
            "calculation_details": calculation_details,
        }

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in calculate_heat_duty: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
