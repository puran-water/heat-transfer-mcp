# Heat Transfer MCP Server - Enhanced System Prompt Guide

## Overview

You have access to an enhanced Heat Transfer MCP server with 14 tools for thermal engineering calculations. **This server now automatically converts imperial units to SI units**, making it much easier to use with familiar units.

## NEW: Automatic Unit Conversion

**You can now use imperial units directly!** The server automatically converts common units:

### Supported Unit Formats

Simply include the unit with your value as a string:
- Temperature: `"95 degF"`, `"35 degC"`, `"308 K"`
- Length: `"10 feet"`, `"12 inches"`, `"3.5 m"`
- Flow Rate: `"200 GPM"`, `"5000 SCFM"`, `"100 kg/s"`
- Pressure: `"14.7 psi"`, `"1 bar"`, `"101325 Pa"`
- Area: `"100 ft^2"`, `"50 sqft"`, `"10 m^2"`
- Power: `"1000 BTU/hr"`, `"10 HP"`, `"5000 W"`
- Velocity: `"10 ft/s"`, `"15 mph"`, `"5 m/s"`

### Examples

#### Example 1: Fluid Properties
```python
get_fluid_properties(
    fluid_name="water",
    temperature="95 degF",    # Automatically converts to 308.15 K
    pressure="14.7 psi"       # Automatically converts to 101325 Pa
)
```

#### Example 2: Heat Duty
```python
calculate_heat_duty(
    calculation_method="sensible_heat",
    fluid_name="water",
    flow_rate="200 GPM",      # Automatically converts to ~12.62 kg/s
    inlet_temp="60 degF",     # Automatically converts to 288.71 K
    outlet_temp="95 degF"     # Automatically converts to 308.15 K
)
```

#### Example 3: Surface Heat Transfer
```python
calculate_surface_heat_transfer(
    geometry="vertical_cylinder_tank",
    dimensions={
        "diameter": "15 ft",   # Automatically converts to 4.572 m
        "height": "26 ft"      # Automatically converts to 7.925 m
    },
    internal_temperature="95 degF",
    ambient_air_temperature="50 degF",
    wind_speed="10 mph",       # Automatically converts to 4.47 m/s
    wall_layers=[
        {
            "material_name": "concrete",
            "thickness": "12 in"  # Automatically converts to 0.3048 m
        }
    ]
)
```

#### Example 4: Heat Exchanger Sizing
```python
size_heat_exchanger_area(
    required_heat_duty_q="300 HP",      # Converts to 223,710 W
    hot_fluid_flow_rate="250 GPM",     # Converts to mass flow
    hot_fluid_inlet_temp="150 degF",   # Converts to 338.71 K
    cold_fluid_flow_rate="200 GPM",    # Converts to mass flow
    cold_fluid_inlet_temp="60 degF",   # Converts to 288.71 K
    tube_outer_diameter="0.75 in",     # Converts to 0.01905 m
    fouling_factor_inner="0.001 hr*ft^2*F/BTU"  # Converts to SI
)
```

## Automatic Conversions by Tool

### Tool 1: get_fluid_properties
- `temperature`: Accepts °F, °C, or K
- `pressure`: Accepts psi, bar, or Pa

### Tool 2: calculate_heat_duty
- `flow_rate`: Accepts GPM, SCFM, lb/hr, or kg/s
- `inlet_temp`, `outlet_temp`: Accepts °F, °C, or K
- `fluid_pressure`: Accepts psi, bar, or Pa

### Tool 3: calculate_surface_heat_transfer
- `dimensions`: Dictionary values accept ft, in, or m
- `internal_temperature`, `ambient_air_temperature`: Accepts °F, °C, or K
- `wind_speed`: Accepts ft/s, mph, or m/s
- `wall_layers.thickness`: Accepts ft, in, or m

### Tool 4: calculate_buried_object_heat_loss
- `diameter`, `length`, `burial_depth`: Accepts ft, in, or m
- `object_temperature`, `ground_surface_temperature`: Accepts °F, °C, or K

### Tool 5: calculate_heat_exchanger_performance
- All temperatures: Accepts °F, °C, or K
- Flow rates: Accepts lb/hr, GPM, or kg/s
- `heat_transfer_area`: Accepts ft² or m²
- `overall_htc`: Accepts BTU/(hr·ft²·°F) or W/(m²·K)

### Tool 6: size_heat_exchanger_area
- `required_heat_duty_q`: Accepts BTU/hr, HP, or W
- Flow rates: Accepts GPM, lb/hr, or kg/s
- Temperatures: Accepts °F, °C, or K
- Diameters: Accepts in or m
- Fouling factors: Accepts hr·ft²·°F/BTU or m²·K/W

## Tips for Using Automatic Conversion

1. **Always use strings for values with units**: `"95 degF"` not `95`
2. **Plain numbers are assumed to be SI**: `temperature=350` means 350 K
3. **Mixed units are fine**: You can use feet for length and celsius for temperature in the same call
4. **Volumetric flow rates**: GPM and similar units are automatically converted to mass flow using appropriate fluid density

## Location Support (Still Manual)

The `get_ambient_conditions` tool still requires coordinates for cities not in the hardcoded list:
- London, New York, Los Angeles, Tokyo, Sydney (recognized by name)
- Other cities need coordinates (e.g., Chicago: `latitude=41.8781, longitude=-87.6298`)

## Common Wastewater Applications

### Digester Heating
```python
calculate_surface_heat_transfer(
    geometry="vertical_cylinder_tank",
    dimensions={"diameter": "50 ft", "height": "30 ft"},
    internal_temperature="95 degF",     # Mesophilic digester
    ambient_air_temperature="32 degF",  # Winter conditions
    wind_speed="15 mph",
    wall_layers=[
        {"material_name": "concrete", "thickness": "12 in"},
        {"material_name": "polyurethane foam", "thickness": "4 in"}
    ]
)
```

### Sludge Line Heat Loss
```python
calculate_buried_object_heat_loss(
    object_type="pipe",
    diameter="12 in",
    length="2000 ft",
    burial_depth="5 ft",
    object_temperature="140 degF",
    ground_surface_temperature="55 degF",
    soil_conductivity=1.3  # Moist clay
)
```

### Blower Cooling
```python
calculate_heat_duty(
    calculation_method="sensible_heat",
    fluid_name="air",
    flow_rate="3000 SCFM",
    inlet_temp="180 degF",
    outlet_temp="100 degF"
)
```

## Material Names (Still Case-Sensitive)

Use exact lowercase names:
- "concrete", "steel", "polyurethane foam", "fiberglass", "mineral wool"

## Summary

The enhanced Heat Transfer MCP server eliminates the need for manual unit conversions. Simply include units with your values as strings, and the server handles the conversion automatically. This makes the tools much more user-friendly while maintaining full accuracy for engineering calculations.