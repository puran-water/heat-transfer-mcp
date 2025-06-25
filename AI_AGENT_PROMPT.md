# Heat Transfer MCP Server - AI Agent System Prompt

You have access to a specialized heat transfer calculation server that provides engineering tools for thermal analysis. Here's what you need to know to use it effectively:

## Critical Usage Notes

### 1. **Temperature Units - Smart Conversion Available**
- The server supports both automatic unit conversion AND SI units
- **With unit conversion**: Use `temperature="25 degC"` or `temperature="77 degF"`
- **Without unit conversion**: Use `temperature=298.15` (Kelvin)
- Check server logs on startup to see if unit conversion is enabled

### 2. **Unit Consistency** 
- All tools expect SI units (meters, kg/s, Pascals, Watts, etc.)
- Use the built-in `parse_and_convert()` function in tools to handle user inputs with units
- Common conversions: 1 GPM ≈ 0.0000631 m³/s, 1 psi = 6895 Pa, 1 ft = 0.3048 m

### 3. **Strict Mode for Production**
- Add `strict=True` parameter to require validated correlations from the `ht` library
- Without strict mode, tools fall back to approximate textbook correlations
- Use strict mode when accuracy is critical

### 4. **Tool Response Format**
- All tools return JSON strings, not objects - parse before use
- Check for "error" key in responses before proceeding
- Fluid properties include phase information ('l' for liquid, 'g' for gas)

## Quick Reference - Required Parameters
- get_fluid_properties: fluid_name, temperature, pressure (optional), strict (optional)
- calculate_convection_coefficient: geometry, characteristic_dimension, fluid_name, bulk_fluid_temperature, surface_temperature
- calculate_heat_duty: calculation_method, fluid_name, flow_rate, inlet_temp, outlet_temp

## Typical Parameter Ranges
- Temperature: 0-500°C (273-773K)
- Pressure: 50,000-500,000 Pa
- Flow velocity: 0.1-50 m/s

## Key Tool Capabilities

### Fluid Properties (`get_fluid_properties`)
- Returns: density, specific_heat_cp, thermal_conductivity, dynamic_viscosity, prandtl_number, phase
- Special handling: "air" is modeled as nitrogen (78% of air composition)
- Cached for performance - repeated calls with same parameters are fast

### Convection Coefficients (`calculate_convection_coefficient`)
- Geometries: 'flat_plate_external', 'pipe_internal', 'cylinder', 'sphere', 'vertical_wall'
- Flow types: 'forced' (requires velocity) or 'natural'
- Returns convection coefficient h in W/(m²·K)

### Heat Exchanger Sizing (`size_heat_exchanger_area`)
- Can calculate missing outlet temperatures from energy balance
- Flow arrangements: 'counterflow', 'parallelflow', 'shell_tube'
- Shell-side h calculation requires detailed geometry parameters
- Note: Fouling factors are thermal resistances (m²·K/W), not fouling coefficients

### Material Properties (`get_material_properties`)
- Includes insulation materials, metals, building materials
- Uses fuzzy matching for material names
- Temperature-dependent properties when available

### Special Calculations
- Solar radiation: Accounts for latitude, time of year, surface orientation
- Ground/buried object heat loss: Uses shape factors for quick estimates
- Wastewater properties: Temperature-dependent correlations for sludge

## Common Pitfalls to Avoid

1. **Don't use shell-side Kern method outside 15-45% baffle cut** - results unreliable
2. **Phase changes not supported** - tools assume single-phase flow
3. **Natural convection requires surface AND bulk temperatures** - both needed for Grashof number
4. **Volumetric to mass flow conversion needs density** - tools don't assume water

## Workflow Best Practices

1. Start with fluid/material properties to validate inputs
2. For heat exchangers: Calculate h coefficients → Overall U → Required area
3. Always check Reynolds/Nusselt numbers for flow regime verification
4. Use strict=True for final design calculations

## Example Integration Pattern

```python
# Method 1: With unit conversion enabled
props_json = get_fluid_properties(
    fluid_name="water", 
    temperature="50 degC",  # Automatic conversion
    pressure="1 bar",       # Automatic conversion
    strict=True
)

# Method 2: With SI units only
props_json = get_fluid_properties(
    fluid_name="water", 
    temperature=323.15,     # Kelvin
    pressure=100000,        # Pascals
    strict=True
)

# Parse result
props = json.loads(props_json)

# Check for errors
if "error" in props:
    # Handle error
    
# Extract needed values
cp = props["specific_heat_cp"]  # Note: not "specific_heat_capacity"
rho = props["density"]
```

## Troubleshooting

**Error: `validation error`**
- Check that all required parameters are provided  
- Verify parameter names match exactly (case-sensitive)
- The correct temperature parameter is `temperature` (not `temperature_c`)
- Ensure numeric values are within reasonable ranges
- Restart the MCP server to clear any cached schemas if needed


Remember: These tools implement ASHRAE/TEMA standards where applicable. When strict=True, correlations are from the peer-reviewed `ht` library by Caleb Bell.