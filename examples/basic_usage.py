"""Simple example of using the Heat Transfer MCP server."""

import sys
import os

# Add the parent directory to the path so we can import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fluid_properties import get_fluid_properties
from tools.heat_duty import calculate_heat_duty
from tools.surface_heat_transfer import calculate_surface_heat_transfer


def main():
    """Run basic usage examples."""

    # Example 1: Get fluid properties
    print("Example 1: Fluid Properties")
    result = get_fluid_properties(
        fluid_name="water",
        temperature="95 degF",  # Automatically converts to 308.15 K
        pressure="14.7 psi",  # Automatically converts to 101325 Pa
    )
    print(f"Density: {result['density']:.1f} kg/m³")
    print(f"Viscosity: {result['dynamic_viscosity']:.2e} Pa·s")
    print()

    # Example 2: Calculate heat duty
    print("Example 2: Heat Duty Calculation")
    result = calculate_heat_duty(
        calculation_method="sensible_heat",
        fluid_name="water",
        flow_rate="200 GPM",  # Automatically converts to kg/s
        inlet_temp="60 degF",
        outlet_temp="95 degF",
    )
    print(f"Heat duty: {result['heat_duty_w']:.0f} W")
    print(f"Heat duty: {result['heat_duty_btu_hr']:.0f} BTU/hr")
    print()

    # Example 3: Digester heat loss
    print("Example 3: Surface Heat Transfer")
    result = calculate_surface_heat_transfer(
        geometry="vertical_cylinder_tank",
        dimensions={"diameter": "50 ft", "height": "30 ft"},
        internal_temperature="95 degF",
        ambient_air_temperature="32 degF",
        wind_speed="15 mph",
        wall_layers=[
            {"material_name": "concrete", "thickness": "12 in"},
            {"material_name": "polyurethane foam", "thickness": "4 in"},
        ],
    )
    print(f"Heat loss: {result['heat_loss_w']:.0f} W")
    print(f"Heat loss: {result['heat_loss_btu_hr']:.0f} BTU/hr")
    print()


if __name__ == "__main__":
    main()
