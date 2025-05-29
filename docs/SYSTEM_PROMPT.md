# Heat Transfer MCP Server - System Prompt Guide

## Overview

You have access to a Heat Transfer MCP server with 14 tools for thermal engineering calculations. This guide helps you use them effectively, especially for wastewater treatment applications.

## CRITICAL: Unit Conversions

**The MCP tools require SI units.** You MUST convert all imperial units before using the tools.

**Note**: While we have created a unit conversion utility (`utils/unit_converter.py`) that could enable automatic unit handling, the current tool implementations still require manual conversion. Future updates may integrate automatic unit parsing, which would allow inputs like "95 degF" or "200 GPM" directly.

### Temperature Conversions
- **°F to K**: `(temp_F + 459.67) * 5/9`
- **°C to K**: `temp_C + 273.15`
- **Example**: 95°F = 308.15 K, 35°C = 308.15 K

### Length Conversions
- **Feet to meters**: Multiply by 0.3048
- **Inches to meters**: Multiply by 0.0254
- **Miles to meters**: Multiply by 1609.34
- **Example**: 10 feet = 3.048 m, 12 inches = 0.3048 m

### Flow Rate Conversions
- **SCFM to kg/s**: Multiply by 0.000594 (for air at standard conditions)
- **CFM to m³/s**: Multiply by 0.000472
- **GPM to m³/s**: Multiply by 0.0000631
- **MGD to m³/s**: Multiply by 0.0438
- **Example**: 1000 SCFM = 0.594 kg/s, 100 GPM = 0.00631 m³/s

### Pressure Conversions
- **PSI to Pa**: Multiply by 6895
- **Bar to Pa**: Multiply by 100000
- **Example**: 14.7 PSI = 101325 Pa

### Area Conversions
- **ft² to m²**: Multiply by 0.0929
- **Example**: 100 ft² = 9.29 m²

### Power Conversions
- **HP to W**: Multiply by 745.7
- **BTU/hr to W**: Multiply by 0.293
- **Example**: 10 HP = 7457 W, 1000 BTU/hr = 293 W

## Location Support for Weather Data

The `get_ambient_conditions` tool recognizes ONLY these cities by name:
- London
- New York
- Los Angeles
- Tokyo
- Sydney

**For ANY other location, you MUST provide coordinates:**
- Chicago: `latitude=41.8781, longitude=-87.6298`
- Houston: `latitude=29.7604, longitude=-95.3698`
- Denver: `latitude=39.7392, longitude=-104.9903`
- Phoenix: `latitude=33.4484, longitude=-112.0740`
- Seattle: `latitude=47.6062, longitude=-122.3321`
- Miami: `latitude=25.7617, longitude=-80.1918`
- Boston: `latitude=42.3601, longitude=-71.0589`
- Atlanta: `latitude=33.7490, longitude=-84.3880`

## Wastewater-Specific Parameters

### Sludge Properties
Approximate sludge as water with density adjustments:
- **Primary sludge (3-6% solids)**: Water properties × 1.02 density factor
- **Thickened sludge (6-8% solids)**: Water properties × 1.05 density factor
- **Digested sludge (3-5% solids)**: Water properties × 1.03 density factor
- **Waste activated sludge (1-2% solids)**: Water properties × 1.01 density factor

### Typical Operating Temperatures
- **Mesophilic digesters**: 35°C (95°F) = 308.15 K
- **Thermophilic digesters**: 55°C (131°F) = 328.15 K
- **Activated sludge**: 10-25°C (50-77°F) = 283-298 K
- **Hot water for heating**: 80-90°C (176-194°F) = 353-363 K
- **Raw wastewater**: 10-20°C (50-68°F) = 283-293 K

### Fouling Factors (m²K/W)
Always include fouling factors for accurate heat exchanger sizing:
- **Clean water**: 0.0001
- **Treated wastewater**: 0.0002
- **Raw wastewater**: 0.0003-0.0005
- **Primary sludge**: 0.0005
- **Digested sludge**: 0.0004
- **Thickened sludge**: 0.0006-0.001
- **Biogas**: 0.0001

### Typical Heat Transfer Coefficients (W/m²K)
- **Water in pipes**: 1000-3000
- **Sludge in pipes**: 500-1500
- **Biogas in pipes**: 50-150
- **Air in pipes**: 50-200
- **Tank walls to air (natural convection)**: 5-15
- **Tank walls to air (forced convection/wind)**: 10-30
- **Jacketed vessels (water)**: 300-800
- **Jacketed vessels (sludge)**: 200-500

### Soil Properties
Common soil thermal conductivities around wastewater plants:
- **Dry soil**: k = 0.5 W/mK
- **Moist soil**: k = 1.0-1.5 W/mK
- **Saturated clay**: k = 1.3 W/mK
- **Saturated sand**: k = 2.4 W/mK
- **Average moist soil**: k = 1.5 W/mK

### Common Pipe Sizes
Standard pipe outer diameters:
- 2" = 0.051 m
- 4" = 0.102 m
- 6" = 0.152 m
- 8" = 0.203 m
- 10" = 0.254 m
- 12" = 0.305 m
- 16" = 0.406 m
- 24" = 0.610 m

## Tool Usage Tips

### 1. For Quick Heat Exchanger Estimates
When `size_heat_exchanger_area` requires tube dimensions, use typical values:
- **Tube OD**: 0.019 m (3/4")
- **Tube ID**: 0.016 m
- **Tube material**: Stainless steel (k = 15 W/mK)
- **Shell-side h**: Use 1000-2000 W/m²K for water, 500-1000 for sludge
- **Tube-side h**: Use 1500-3000 W/m²K for water, 800-1500 for sludge

### 2. For Digester Heat Loss
Use `calculate_surface_heat_transfer` with:
- **geometry**: "vertical_cylinder_tank"
- **wall_layers**: Include both concrete and insulation layers
- **insulation**: Typically 3-4 inches (0.076-0.102 m) polyurethane foam
- **wind_speed**: Use 5 m/s if unknown
- **Note**: If convergence warnings appear, results are still usable

### 3. For Buried Pipes
Use `calculate_buried_object_heat_loss` with:
- **soil_conductivity**: 1.3 W/mK for moist clay (common)
- **burial_depth**: Typically 1.2-1.8 m (4-6 ft)
- **Note**: Include pipe insulation thickness in diameter if insulated

### 4. Material Names
Use exact lowercase names for materials:
- "concrete" (not "Concrete")
- "polyurethane foam" (not "foam insulation" or "PU foam")
- "steel" or "stainless steel"
- "fiberglass"
- "mineral wool"

### 5. Fluid Names
The tools recognize:
- "water"
- "air" (uses nitrogen properties as proxy)
- "ethanol"
- "methanol"
- Other chemicals by their standard names

## Common Calculations

### 1. Digester Heating Load
```
Total Load = Heat Loss + Sludge Heating + Mixing Energy Loss

Typical values:
- Heat loss: 10-20 W/m² (insulated)
- Sludge heating: 0.5-1.0 kW per m³/day feed
- Mixing losses: 5-10% of total
```

### 2. Blower Cooling
```
Heat duty = Flow_rate × Cp × ΔT

For air:
- Use actual temperature for density calculation
- Cp ≈ 1005 J/(kg·K) (relatively constant)
- Remember SCFM is at standard conditions
```

### 3. Heat Recovery Potential
```
Maximum recovery = Flow × Cp × (T_hot - T_ambient)
Practical recovery = 60-80% of maximum
```

## Common Pitfalls to Avoid

1. **Temperature**: Always convert to Kelvin - the tools will NOT accept °F or °C
2. **Flow rates**: SCFM and GPM must be converted to kg/s or m³/s
3. **Locations**: Use coordinates for cities not in the hardcoded list
4. **Air density**: Changes significantly with temperature - don't use fixed values
5. **Fouling**: Always include fouling factors for wastewater applications
6. **Materials**: Use exact lowercase names from the materials database

## Error Handling

### "Fluid not recognized"
- Check spelling and capitalization
- Use "water" instead of specific sludge types
- Air might need special handling

### "Material not found"
- Use lowercase names
- Try simpler names (e.g., "foam" instead of "polyurethane foam")
- Check the fallback materials list

### "Did not converge"
- Results are often still usable
- Check if temperature differences are realistic
- Very high wind speeds can cause issues

## Example Workflow

```python
# 1. Convert units first
temp_k = (95 + 459.67) * 5/9  # 95°F to K
flow_kgs = 1000 * 0.000594     # 1000 SCFM to kg/s
diameter_m = 12 * 0.0254       # 12 inches to m

# 2. Get fluid properties with temperature
props = get_fluid_properties("water", temp_k, 101325)

# 3. Include all required parameters
result = calculate_heat_duty(
    calculation_method="sensible_heat",
    fluid_name="water",
    flow_rate=flow_kgs,
    inlet_temp=temp_k,
    outlet_temp=outlet_temp_k
)
```

## Quick Reference Card

| From | To | Multiply by |
|------|----|-----------:|
| °F | K | Use formula: (°F + 459.67) × 5/9 |
| feet | meters | 0.3048 |
| inches | meters | 0.0254 |
| SCFM | kg/s | 0.000594 |
| GPM | m³/s | 0.0000631 |
| PSI | Pa | 6895 |
| HP | W | 745.7 |
| BTU/hr | W | 0.293 |

Remember: When in doubt, check units first!