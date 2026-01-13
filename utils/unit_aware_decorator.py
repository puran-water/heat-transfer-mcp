"""Universal unit-aware decorator for heat transfer MCP tools.

This decorator automatically converts imperial units to SI units for tool parameters.
It's designed to be applied at the server registration level for all tools.
"""

import logging
from functools import wraps
from typing import Dict, Any, Union, Optional, List, Tuple
import re

from utils.unit_converter import parse_and_convert, convert_units

logger = logging.getLogger("heat-transfer-mcp.unit_decorator")

# Parameter type definitions with their target SI units
PARAMETER_TYPES = {
    # Temperature conversions
    "temperature": {
        "target_unit": "kelvin",
        "common_units": ["degF", "degC", "K", "fahrenheit", "celsius"],
        "default_imperial": "degF",
    },
    # Length conversions
    "length": {
        "target_unit": "meter",
        "common_units": ["ft", "feet", "in", "inch", "inches", "m", "meter"],
        "default_imperial": "feet",
    },
    # Small length (typically inches)
    "length_small": {
        "target_unit": "meter",
        "common_units": ["in", "inch", "inches", "mm", "cm", "m"],
        "default_imperial": "inch",
    },
    # Area conversions
    "area": {
        "target_unit": "meter^2",
        "common_units": ["ft2", "ft^2", "sqft", "m2", "m^2", "sqm"],
        "default_imperial": "foot^2",
    },
    # Mass flow rate conversions
    "mass_flow": {
        "target_unit": "kg/s",
        "common_units": ["lb/hr", "lbm/hr", "kg/s", "kg/hr"],
        "default_imperial": "pound/hour",
    },
    # Volumetric flow rate conversions (needs density)
    "volume_flow": {
        "target_unit": "kg/s",  # Convert to mass flow
        "common_units": ["gpm", "GPM", "mgd", "MGD", "L/s", "m3/s", "cfm", "SCFM"],
        "default_imperial": "gpm",
        "needs_density": True,
    },
    # Pressure conversions
    "pressure": {
        "target_unit": "pascal",
        "common_units": ["psi", "psig", "psia", "bar", "Pa", "kPa", "atm"],
        "default_imperial": "psi",
    },
    # Velocity conversions
    "velocity": {"target_unit": "m/s", "common_units": ["ft/s", "fps", "mph", "m/s", "km/hr"], "default_imperial": "ft/s"},
    # Power/heat conversions
    "power": {"target_unit": "watt", "common_units": ["BTU/hr", "hp", "HP", "W", "kW", "MW"], "default_imperial": "BTU/hour"},
    # Heat transfer coefficient
    "htc": {
        "target_unit": "W/(m^2*K)",
        "common_units": ["BTU/(hr*ft^2*F)", "W/(m^2*K)"],
        "default_imperial": "BTU/(hour*foot^2*degF)",
    },
    # Thermal conductivity
    "thermal_conductivity": {
        "target_unit": "W/(m*K)",
        "common_units": ["BTU/(hr*ft*F)", "W/(m*K)"],
        "default_imperial": "BTU/(hour*foot*degF)",
    },
    # Fouling factor
    "fouling": {
        "target_unit": "m^2*K/W",
        "common_units": ["hr*ft^2*F/BTU", "m^2*K/W"],
        "default_imperial": "hour*foot^2*degF/BTU",
    },
}


def detect_unit_type(value: str) -> Optional[str]:
    """Detect the unit type from a string value.

    Args:
        value: String that may contain a unit

    Returns:
        Unit type key from PARAMETER_TYPES or None
    """
    if not isinstance(value, str):
        return None

    value_lower = value.lower()

    # Check each parameter type's common units
    for param_type, config in PARAMETER_TYPES.items():
        for unit in config["common_units"]:
            if unit.lower() in value_lower:
                return param_type

    return None


def convert_value(value: Any, param_type: str, param_name: str = None, fluid_density: Optional[float] = None) -> Any:
    """Convert a single value based on parameter type.

    Args:
        value: Value to convert (may be number or string with unit)
        param_type: Type of parameter from PARAMETER_TYPES
        param_name: Optional parameter name for logging
        fluid_density: Density for volumetric to mass flow conversions

    Returns:
        Converted value in SI units
    """
    if param_type not in PARAMETER_TYPES:
        return value

    config = PARAMETER_TYPES[param_type]
    target_unit = config["target_unit"]

    # Handle string values with potential units
    if isinstance(value, str):
        try:
            # Special handling for volumetric flow
            if config.get("needs_density"):
                return parse_and_convert(value, target_unit, param_type, fluid_density)
            else:
                return parse_and_convert(value, target_unit, param_type)
        except Exception as e:
            logger.debug(f"Could not parse '{value}' as unit string: {e}")
            # Try to convert as plain number
            try:
                numeric_value = float(value)
                # Assume imperial unit if just a number
                if config.get("default_imperial"):
                    try:
                        return convert_units(numeric_value, config["default_imperial"], target_unit)
                    except:
                        pass
                return numeric_value
            except ValueError:
                logger.warning(f"Could not convert {param_name}='{value}' to number")
                raise

    # Handle numeric values (already in SI or need assumption)
    elif isinstance(value, (int, float)):
        # For now, assume it's already in SI units
        # Could add heuristics here (e.g., temp > 200 likely °F not K)
        return float(value)

    return value


def convert_dict_values(data: Dict[str, Any], mappings: Dict[str, str]) -> Dict[str, Any]:
    """Convert values in a dictionary based on parameter mappings.

    Args:
        data: Dictionary of parameter values
        mappings: Dictionary mapping parameter names to types

    Returns:
        Dictionary with converted values
    """
    converted = {}

    # First pass: convert all non-volume_flow parameters
    for key, value in data.items():
        if value is None:
            converted[key] = None
            continue

        # Check if this parameter has a mapping
        if key in mappings:
            param_type = mappings[key]

            # Skip volume_flow for now - need to handle with fluid density
            if param_type == "volume_flow":
                converted[key] = value  # Keep original for now
                continue

            # Handle nested dictionaries (like dimensions)
            if param_type == "nested" and isinstance(value, dict):
                # Assume all values in nested dict are lengths
                nested_converted = {}
                for nested_key, nested_val in value.items():
                    nested_converted[nested_key] = convert_value(nested_val, "length", f"{key}.{nested_key}")
                converted[key] = nested_converted

            # Handle lists (like wall_layers)
            elif param_type == "list_of_dicts" and isinstance(value, list):
                converted_list = []
                for item in value:
                    if isinstance(item, dict) and "thickness" in item:
                        item_copy = item.copy()
                        item_copy["thickness"] = convert_value(item["thickness"], "length", f"{key}.thickness")
                        converted_list.append(item_copy)
                    else:
                        converted_list.append(item)
                converted[key] = converted_list

            else:
                # Single value conversion
                converted[key] = convert_value(value, param_type, key)
        else:
            # No mapping, pass through
            converted[key] = value

    # Second pass: handle volume_flow conversions with fluid density
    for key, value in data.items():
        if key in mappings and mappings[key] == "volume_flow" and value is not None:
            # Try to get fluid density for conversion
            fluid_density = _get_fluid_density_for_conversion(converted)
            if fluid_density:
                try:
                    converted[key] = convert_value(value, "volume_flow", key, fluid_density)
                except Exception as e:
                    logger.warning(f"Could not convert volume flow {key}: {e}. Using original value.")
                    converted[key] = value
            else:
                logger.warning(f"No fluid density available for volume flow conversion of {key}. Using original value.")
                converted[key] = value

    return converted


def _get_fluid_density_for_conversion(converted_params: Dict[str, Any]) -> Optional[float]:
    """Try to determine fluid density from converted parameters.

    This function attempts to get fluid density by looking for fluid_name and temperature
    in the parameters and calling get_fluid_properties if available.
    """
    try:
        # Check if we have fluid_name and temperature to get density
        fluid_name = converted_params.get("fluid_name")
        temperature = (
            converted_params.get("temperature")
            or converted_params.get("inlet_temp")
            or converted_params.get("bulk_fluid_temperature")
        )
        pressure = converted_params.get("pressure", 101325.0)  # Standard pressure default

        if fluid_name and temperature:
            # Import here to avoid circular imports
            from tools.fluid_properties import get_fluid_properties
            import json

            props_json = get_fluid_properties(fluid_name, temperature, pressure)
            props = json.loads(props_json)

            if "error" not in props and "density" in props:
                return props["density"]

    except Exception as e:
        logger.debug(f"Could not determine fluid density: {e}")

    return None


def unit_aware(param_mappings: Dict[str, str]):
    """Decorator to make a function unit-aware.

    Args:
        param_mappings: Dictionary mapping parameter names to their unit types

    Example:
        @unit_aware({
            'temperature': 'temperature',
            'length': 'length',
            'flow_rate': 'mass_flow'
        })
        def my_function(temperature, length, flow_rate):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(**kwargs):
            # Log the conversion attempt
            logger.debug(f"Unit conversion for {func.__name__} with params: {list(kwargs.keys())}")

            try:
                # Convert the parameters
                converted_kwargs = convert_dict_values(kwargs, param_mappings)

                # Log successful conversions
                for key in kwargs:
                    if key in converted_kwargs and kwargs[key] != converted_kwargs[key]:
                        logger.info(f"Converted {key}: {kwargs[key]} → {converted_kwargs[key]}")

                # Call the original function with converted values
                return func(**converted_kwargs)

            except Exception as e:
                logger.error(f"Unit conversion error in {func.__name__}: {e}")
                # Fall back to original values
                return func(**kwargs)

        # Add attribute to indicate this function is unit-aware
        wrapper._unit_aware = True
        wrapper._param_mappings = param_mappings

        return wrapper

    return decorator


# Tool-specific parameter mappings
TOOL_MAPPINGS = {
    "get_fluid_properties": {"temperature": "temperature", "pressure": "pressure"},
    "get_material_properties": {"temperature": "temperature"},
    "calculate_convection_coefficient": {
        "temperature": "temperature",
        "velocity": "velocity",
        "characteristic_length": "length",
        "surface_area": "area",
        "surface_temperature": "temperature",
        "bulk_temperature": "temperature",
        "hydraulic_diameter": "length",
        "pipe_diameter": "length",
        "flow_area": "area",
    },
    "calculate_overall_heat_transfer_coefficient": {
        "inside_htc": "htc",
        "outside_htc": "htc",
        "wall_thickness": "length",
        "wall_thermal_conductivity": "thermal_conductivity",
        "inside_fouling_factor": "fouling",
        "outside_fouling_factor": "fouling",
        "inside_diameter": "length",
        "outside_diameter": "length",
    },
    "calculate_surface_heat_transfer": {
        "dimensions": "nested",  # Special handling for nested dict
        "internal_temperature": "temperature",
        "ambient_air_temperature": "temperature",
        "wind_speed": "velocity",
        "wall_layers": "list_of_dicts",  # Special handling for list
    },
    "calculate_heat_exchanger_performance": {
        "hot_inlet_temp": "temperature",
        "hot_outlet_temp": "temperature",
        "cold_inlet_temp": "temperature",
        "cold_outlet_temp": "temperature",
        "hot_flow_rate": "mass_flow",
        "cold_flow_rate": "mass_flow",
        "heat_transfer_area": "area",
        "overall_htc": "htc",
    },
    "calculate_heat_duty": {
        "flow_rate": "volume_flow",  # Can be GPM, SCFM, etc.
        "inlet_temp": "temperature",
        "outlet_temp": "temperature",
        "fluid_pressure": "pressure",
        "overall_heat_transfer_coefficient_U": "htc",
        "heat_transfer_area": "area",
        "mean_temperature_difference": "temperature",
    },
    "calculate_solar_radiation_on_surface": {"surface_area": "area", "ambient_temperature": "temperature"},
    "calculate_ground_heat_loss": {
        "foundation_length": "length",
        "foundation_width": "length",
        "foundation_depth": "length",
        "wall_thickness": "length",
        "floor_thickness": "length",
        "inside_temperature": "temperature",
        "outside_temperature": "temperature",
        "ground_temperature": "temperature",
        "wall_thermal_conductivity": "thermal_conductivity",
        "floor_thermal_conductivity": "thermal_conductivity",
        "soil_thermal_conductivity": "thermal_conductivity",
    },
    "calculate_buried_object_heat_loss": {
        "diameter": "length",
        "width": "length",
        "height": "length",
        "length": "length",
        "burial_depth": "length",
        "object_temperature": "temperature",
        "ground_surface_temperature": "temperature",
        "soil_conductivity": "thermal_conductivity",
    },
    "calculate_hx_shell_side_h_kern": {
        "mass_flow_rate": "mass_flow",
        "tube_outer_diameter": "length_small",
        "shell_inner_diameter": "length_small",
        "tube_pitch": "length_small",
        "baffle_spacing": "length_small",
        "bulk_temperature": "temperature",
        "wall_temperature": "temperature",
    },
    "size_heat_exchanger_area": {
        "required_heat_duty_q": "power",
        "hot_fluid_flow_rate": "volume_flow",
        "hot_fluid_inlet_temp": "temperature",
        "cold_fluid_flow_rate": "volume_flow",
        "cold_fluid_inlet_temp": "temperature",
        "tube_outer_diameter": "length_small",
        "tube_inner_diameter": "length_small",
        "tube_material_conductivity": "thermal_conductivity",
        "fouling_factor_inner": "fouling",
        "fouling_factor_outer": "fouling",
    },
    "estimate_hx_physical_dims": {
        "required_area": "area",
        "tube_outer_diameter": "length_small",
        "tube_inner_diameter": "length_small",
        "shell_inner_diameter": "length_small",
        "baffle_spacing": "length_small",
        "tube_pitch": "length_small",
        "tube_length_options": "list",  # List of lengths
    },
    # Omnibus consolidated tools
    "tank_heat_loss": {
        "dimensions": "nested",
        "contents_temperature": "temperature",
        "ambient_air_temperature": "temperature",
        "wind_speed": "velocity",
        "sky_temperature": "temperature",
        "wall_layers": "list_of_dicts",
    },
    "heat_exchanger_design": {
        "process_mass_flow_kg_s": "mass_flow",
        "process_inlet_temp_K": "temperature",
        "process_target_temp_K": "temperature",
        "heating_inlet_temp_K": "temperature",
        "heating_outlet_temp_K": "temperature",
        "required_total_duty_W": "power",
        "overall_U_W_m2K": "htc",
    },
    "pipe_heat_management": {
        "outer_diameter_m": "length",
        "length_m": "length",
        "internal_temperature_K": "temperature",
        "ambient_air_temperature_K": "temperature",
        "wind_speed_m_s": "velocity",
        "burial_depth_m": "length",
        "ground_surface_temperature_K": "temperature",
        "pipe_inner_diameter_m": "length",
        "heat_trace_w_per_m": "power",
    },
}


def get_tool_mapping(tool_name: str) -> Dict[str, str]:
    """Get parameter mappings for a specific tool.

    Args:
        tool_name: Name of the tool function

    Returns:
        Dictionary of parameter mappings
    """
    return TOOL_MAPPINGS.get(tool_name, {})


def make_tool_unit_aware(tool_func):
    """Make a tool function unit-aware using its predefined mappings.

    Args:
        tool_func: The tool function to wrap

    Returns:
        Unit-aware version of the function
    """
    tool_name = tool_func.__name__
    mappings = get_tool_mapping(tool_name)

    if not mappings:
        logger.debug(f"No unit mappings defined for {tool_name}")
        return tool_func

    return unit_aware(mappings)(tool_func)
