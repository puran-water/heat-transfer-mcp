"""
Heat duty calculation tool to determine heat transfer rates.

This module provides functionality to calculate heat duty based on
sensible heat change or overall heat transfer methods.
"""

import json
import logging
import math
from typing import Any, Dict, Optional

from heat_transfer_mcp.tools.fluid_properties import PropertyLookupRefused, calculation_properties, calculation_property_source, property_refusal

logger = logging.getLogger("heat-transfer-mcp.heat_duty")


def _json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, allow_nan=False)


def _require_finite(value: Optional[float], name: str, *, positive: bool = False) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return numeric


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
    property_source: str = "CoolProp",
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
        property_source: The branch every property lookup reads, 'CoolProp' or 'thermo' (CALCULATION_PROPERTY_RULE)

    Returns:
        JSON string with calculated heat duty and details
    """
    try:
        fluid_pressure = _require_finite(fluid_pressure, "fluid_pressure", positive=True)
        property_source = calculation_property_source(property_source)
        calculation_details = {}
        heat_duty = None

        # Process based on calculation method
        if calculation_method.lower() == "sensible_heat":
            # Validate required inputs for sensible heat
            if None in [flow_rate, inlet_temp, outlet_temp]:
                return _json(
                    {"error": "For 'sensible_heat' method, 'flow_rate', 'inlet_temp', and 'outlet_temp' are required."}
                )
            flow_rate = _require_finite(flow_rate, "flow_rate", positive=True)
            inlet_temp = _require_finite(inlet_temp, "inlet_temp", positive=True)
            outlet_temp = _require_finite(outlet_temp, "outlet_temp", positive=True)
            specific_heat_cp = _require_finite(specific_heat_cp, "specific_heat_cp", positive=True)

            phase = None
            if fluid_name:
                for temperature in (inlet_temp, outlet_temp):
                    endpoint = calculation_properties(fluid_name, temperature, fluid_pressure, property_source)
                    current_phase = endpoint.get("phase")
                    if current_phase not in {"l", "g", "liquid", "gas"}:
                        raise ValueError("sensible_heat requires a qualified liquid or gas phase")
                    current_phase = {"liquid": "l", "gas": "g"}.get(current_phase, current_phase)
                    if phase is not None and current_phase != phase:
                        raise ValueError("sensible_heat cannot cross a phase transition; latent heat requires a phase-change model")
                    phase = current_phase

            # Get specific_heat_cp if not provided
            if specific_heat_cp is None:
                if fluid_name is None:
                    return _json(
                        {"error": "For 'sensible_heat' method, provide either 'specific_heat_cp' or 'fluid_name' for lookup."}
                    )

                # Calculate average temperature for property lookup
                avg_temp = (inlet_temp + outlet_temp) / 2.0

                # Get fluid properties at average temperature
                fluid_props = calculation_properties(fluid_name, avg_temp, fluid_pressure, property_source)

                specific_heat_cp = fluid_props.get("specific_heat_cp")
                if specific_heat_cp is None:
                    return _json({"error": f"Could not determine specific heat capacity for '{fluid_name}'."})
                specific_heat_cp = _require_finite(specific_heat_cp, "specific_heat_cp", positive=True)

                from scipy.integrate import quad

                def cp_at(temperature):
                    props = calculation_properties(fluid_name, temperature, fluid_pressure, property_source)
                    return _require_finite(props.get("specific_heat_cp"), "specific_heat_cp", positive=True)

                if inlet_temp != outlet_temp:
                    delta_h, _ = quad(cp_at, inlet_temp, outlet_temp)
                    specific_heat_cp = delta_h / (outlet_temp - inlet_temp)
                calculation_details["specific_heat_source"] = "Temperature integral of property-backend mass cp at supplied pressure"
                calculation_details["data_source"] = fluid_props.get("data_source")
            else:
                calculation_details["specific_heat_source"] = "User provided"
                calculation_details["assumption"] = "Supplied constant cp; sensible single-phase duty only"

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
                return _json(
                    {
                        "error": "For 'ua_deltaT' method, 'overall_heat_transfer_coefficient_U', 'heat_transfer_area', and 'mean_temperature_difference' are required."
                    }
                )
            overall_heat_transfer_coefficient_U = _require_finite(
                overall_heat_transfer_coefficient_U,
                "overall_heat_transfer_coefficient_U",
                positive=True,
            )
            heat_transfer_area = _require_finite(heat_transfer_area, "heat_transfer_area", positive=True)
            mean_temperature_difference = _require_finite(
                mean_temperature_difference,
                "mean_temperature_difference",
                positive=True,
            )

            # Calculate heat duty: Q = U * A * ΔT
            heat_duty = overall_heat_transfer_coefficient_U * heat_transfer_area * mean_temperature_difference

            # Record calculation details
            calculation_details["overall_heat_transfer_coefficient_u_w_m2k"] = overall_heat_transfer_coefficient_U
            calculation_details["heat_transfer_area_m2"] = heat_transfer_area
            calculation_details["mean_temperature_difference_k"] = mean_temperature_difference
            calculation_details["calculation_formula"] = "Q = U * A * ΔT"

        else:
            return _json(
                {"error": f"Invalid calculation_method: '{calculation_method}'. Choose 'sensible_heat' or 'ua_deltaT'."}
            )

        # Check if heat duty was calculated
        if heat_duty is None:
            return _json({"error": "Could not calculate heat duty with the provided inputs."})

        # Create result
        result = {
            "heat_duty_watts": heat_duty,
            "heat_duty_kilowatts": heat_duty / 1000.0,
            "calculation_method": calculation_method,
            "calculation_details": calculation_details,
        }

        return _json(result)

    except PropertyLookupRefused as refused:
        return property_refusal(refused, "Failed to get fluid properties")
    except Exception as e:
        logger.error(f"Unexpected error in calculate_heat_duty: {e}", exc_info=True)
        return _json({"error": f"An unexpected error occurred: {str(e)}"})
