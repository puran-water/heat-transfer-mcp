"""Simple example of using the Heat Transfer MCP server."""

# Example 1: Get fluid properties
result = get_fluid_properties(
    fluid_name="water",
    temperature="95 degF",  # Automatically converts to 308.15 K
    pressure="14.7 psi"     # Automatically converts to 101325 Pa
)

# Example 2: Calculate heat duty
result = calculate_heat_duty(
    calculation_method="sensible_heat",
    fluid_name="water",
    flow_rate="200 GPM",    # Automatically converts to kg/s
    inlet_temp="60 degF",
    outlet_temp="95 degF"
)

# Example 3: Digester heat loss
result = calculate_surface_heat_transfer(
    geometry="vertical_cylinder_tank",
    dimensions={"diameter": "50 ft", "height": "30 ft"},
    internal_temperature="95 degF",
    ambient_air_temperature="32 degF",
    wind_speed="15 mph",
    wall_layers=[
        {"material_name": "concrete", "thickness": "12 in"},
        {"material_name": "polyurethane foam", "thickness": "4 in"}
    ]
)
