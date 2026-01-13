"""Unit conversion utilities using Pint for the Heat Transfer MCP server.

This module provides unit conversion functions to handle common imperial to SI conversions
needed for wastewater and HVAC engineering calculations.
"""

from pint import UnitRegistry
import logging
from typing import Union, Optional

logger = logging.getLogger("heat-transfer-mcp.units")

# Initialize unit registry
ureg = UnitRegistry()

# Add common wastewater and HVAC units
ureg.define("SCFM = 0.000594 * kg/s")  # Standard cubic feet per minute to kg/s (for air at standard conditions)
ureg.define("MGD = 0.0438126 * m^3/s")  # Million gallons per day
ureg.define("gpm = gallon/minute")
ureg.define("GPM = gallon/minute")  # Capital version
ureg.define("cfm = foot^3/minute")  # Cubic feet per minute


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between units using Pint.

    Args:
        value: Numerical value to convert
        from_unit: Source unit string (e.g., 'degF', 'feet', 'SCFM', 'psi')
        to_unit: Target unit string (e.g., 'kelvin', 'meter', 'kg/s', 'pascal')

    Returns:
        Converted value as float

    Raises:
        ValueError: If conversion fails
    """
    try:
        quantity = ureg.Quantity(value, from_unit)
        converted = quantity.to(to_unit)
        return float(converted.magnitude)
    except Exception as e:
        logger.error(f"Unit conversion failed: {e}")
        raise ValueError(f"Cannot convert {value} {from_unit} to {to_unit}: {e}")


# Temperature conversions
def fahrenheit_to_kelvin(temp_f: float) -> float:
    """Convert temperature from Fahrenheit to Kelvin."""
    return convert_units(temp_f, "degF", "kelvin")


def celsius_to_kelvin(temp_c: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return convert_units(temp_c, "degC", "kelvin")


def kelvin_to_fahrenheit(temp_k: float) -> float:
    """Convert temperature from Kelvin to Fahrenheit."""
    return convert_units(temp_k, "kelvin", "degF")


def kelvin_to_celsius(temp_k: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return convert_units(temp_k, "kelvin", "degC")


# Length conversions
def feet_to_meters(length_ft: float) -> float:
    """Convert length from feet to meters."""
    return convert_units(length_ft, "feet", "meter")


def inches_to_meters(length_in: float) -> float:
    """Convert length from inches to meters."""
    return convert_units(length_in, "inch", "meter")


def meters_to_feet(length_m: float) -> float:
    """Convert length from meters to feet."""
    return convert_units(length_m, "meter", "feet")


def meters_to_inches(length_m: float) -> float:
    """Convert length from meters to inches."""
    return convert_units(length_m, "meter", "inch")


# Flow rate conversions
def scfm_to_kg_per_s(flow_scfm: float) -> float:
    """Convert flow rate from SCFM (Standard Cubic Feet per Minute) to kg/s.

    Note: This assumes air at standard conditions (14.7 psia, 60°F).
    """
    return convert_units(flow_scfm, "SCFM", "kg/s")


def cfm_to_m3_per_s(flow_cfm: float) -> float:
    """Convert flow rate from CFM to m³/s."""
    return convert_units(flow_cfm, "cfm", "m^3/s")


def gpm_to_m3_per_s(flow_gpm: float) -> float:
    """Convert flow rate from GPM (Gallons Per Minute) to m³/s."""
    return convert_units(flow_gpm, "gpm", "m^3/s")


def mgd_to_m3_per_s(flow_mgd: float) -> float:
    """Convert flow rate from MGD (Million Gallons per Day) to m³/s."""
    return convert_units(flow_mgd, "MGD", "m^3/s")


def m3_per_s_to_gpm(flow_m3s: float) -> float:
    """Convert flow rate from m³/s to GPM."""
    return convert_units(flow_m3s, "m^3/s", "gpm")


def kg_per_s_to_scfm(flow_kgs: float) -> float:
    """Convert flow rate from kg/s to SCFM (for air)."""
    return convert_units(flow_kgs, "kg/s", "SCFM")


# Pressure conversions
def psi_to_pascal(pressure_psi: float) -> float:
    """Convert pressure from PSI to Pascal."""
    return convert_units(pressure_psi, "psi", "pascal")


def bar_to_pascal(pressure_bar: float) -> float:
    """Convert pressure from bar to Pascal."""
    return convert_units(pressure_bar, "bar", "pascal")


def pascal_to_psi(pressure_pa: float) -> float:
    """Convert pressure from Pascal to PSI."""
    return convert_units(pressure_pa, "pascal", "psi")


def pascal_to_bar(pressure_pa: float) -> float:
    """Convert pressure from Pascal to bar."""
    return convert_units(pressure_pa, "pascal", "bar")


# Area conversions
def ft2_to_m2(area_ft2: float) -> float:
    """Convert area from ft² to m²."""
    return convert_units(area_ft2, "foot^2", "meter^2")


def m2_to_ft2(area_m2: float) -> float:
    """Convert area from m² to ft²."""
    return convert_units(area_m2, "meter^2", "foot^2")


# Power conversions
def hp_to_watts(power_hp: float) -> float:
    """Convert power from horsepower to watts."""
    return convert_units(power_hp, "horsepower", "watt")


def watts_to_hp(power_w: float) -> float:
    """Convert power from watts to horsepower."""
    return convert_units(power_w, "watt", "horsepower")


def btu_per_hr_to_watts(power_btu_hr: float) -> float:
    """Convert power from BTU/hr to watts."""
    return convert_units(power_btu_hr, "Btu/hour", "watt")


def watts_to_btu_per_hr(power_w: float) -> float:
    """Convert power from watts to BTU/hr."""
    return convert_units(power_w, "watt", "Btu/hour")


# Mass flow conversions
def lb_per_hr_to_kg_per_s(flow_lb_hr: float) -> float:
    """Convert mass flow from lb/hr to kg/s."""
    return convert_units(flow_lb_hr, "pound/hour", "kg/s")


def kg_per_s_to_lb_per_hr(flow_kg_s: float) -> float:
    """Convert mass flow from kg/s to lb/hr."""
    return convert_units(flow_kg_s, "kg/s", "pound/hour")


# Utility function to parse and convert user input
def parse_and_convert(
    value_str: Union[str, float, int],
    target_unit: str,
    param_type: Optional[str] = None,
    fluid_density: Optional[float] = None,
) -> float:
    """Parse a value with optional unit and convert to target unit.

    Args:
        value_str: String that may contain value and unit (e.g., "70 degF", "100 feet", "50")
        target_unit: Target unit to convert to
        param_type: Optional parameter type hint (e.g., 'temperature', 'length', 'pressure')
        fluid_density: Density in kg/m³ for volumetric to mass flow conversions (required for volume_flow)

    Returns:
        Converted value in target units

    Examples:
        >>> parse_and_convert("70 degF", "kelvin")
        294.26...
        >>> parse_and_convert("100", "meter", param_type="length")  # Assumes feet for length
        30.48
    """
    # Handle numeric input
    if isinstance(value_str, (int, float)):
        # Numeric values are assumed to be in SI units already
        return float(value_str)

    # Try to parse as "value unit" format
    parts = value_str.strip().split(maxsplit=1)

    if len(parts) == 2:
        # Has explicit unit
        try:
            value = float(parts[0])
            unit = parts[1]

            # Special handling for volumetric to mass flow conversion
            if target_unit == "kg/s" and unit.upper() in ["GPM", "MGD", "L/S", "M3/S"]:
                if fluid_density is None:
                    raise ValueError(f"Fluid density is required for volumetric to mass flow conversion of {value_str}")

                # First convert to m³/s
                if unit.upper() == "GPM":
                    vol_flow_m3s = value * 0.0000631
                elif unit.upper() == "MGD":
                    vol_flow_m3s = value * 0.0438
                elif unit.upper() == "L/S":
                    vol_flow_m3s = value * 0.001
                elif unit.upper() == "M3/S":
                    vol_flow_m3s = value
                else:
                    vol_flow_m3s = convert_units(value, unit, "m^3/s")

                # Then convert to kg/s using actual fluid density
                return vol_flow_m3s * fluid_density

            return convert_units(value, unit, target_unit)
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to parse '{value_str}' as value+unit: {e}")

    # Try as plain number
    try:
        value = float(value_str.strip())

        # If param_type hint provided, use common defaults
        if param_type:
            default_units = {
                "temperature": "degF",
                "length": "feet",
                "pressure": "psi",
                "flow_rate": "gpm",
                "area": "foot^2",
                "power": "horsepower",
            }

            if param_type in default_units:
                from_unit = default_units[param_type]
                logger.info(f"Assuming {from_unit} for {param_type} value {value}")
                return convert_units(value, from_unit, target_unit)

        # No conversion needed - already in target units
        return value

    except ValueError:
        raise ValueError(f"Could not parse '{value_str}' as a numeric value")


if __name__ == "__main__":
    # Test conversions
    print("Testing unit conversions:")
    print(f"70°F = {fahrenheit_to_kelvin(70):.2f} K")
    print(f"100 feet = {feet_to_meters(100):.2f} m")
    print(f"5000 SCFM = {scfm_to_kg_per_s(5000):.3f} kg/s")
    print(f"500 GPM = {gpm_to_m3_per_s(500):.5f} m³/s")
    print(f"50 PSI = {psi_to_pascal(50):.0f} Pa")

    # Test parse_and_convert
    print("\nTesting parse_and_convert:")
    print(f"'70 degF' -> {parse_and_convert('70 degF', 'kelvin'):.2f} K")
    print(f"'100 feet' -> {parse_and_convert('100 feet', 'meter'):.2f} m")
    print(f"'50' (temp) -> {parse_and_convert('50', 'kelvin', 'temperature'):.2f} K")
