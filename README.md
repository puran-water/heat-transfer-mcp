# Heat Transfer MCP Server

A comprehensive Model Context Protocol (MCP) server for thermal engineering calculations, featuring 14 specialized tools for heat transfer analysis with automatic imperial-to-SI unit conversion.

## Features

- **14 Heat Transfer Tools**: Complete suite for thermal engineering calculations
- **Automatic Unit Conversion**: Accepts imperial units (°F, GPM, PSI, ft, etc.) and converts to SI automatically
- **390+ Material Properties**: Comprehensive database from VDI and ASHRAE handbooks
- **Temperature-Dependent Properties**: Accurate fluid properties using thermo library
- **Weather Data Integration**: Real-time ambient conditions for any location
- **Wastewater Engineering Focus**: Specialized parameters for treatment plant applications

## Installation

1. Clone the repository:
```bash
git clone https://github.com/puran-water/heat-transfer-mcp.git
cd heat-transfer-mcp
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the MCP server:
```bash
python server.py
```

The server will automatically detect and enable unit conversion support. For detailed setup instructions, see the [Setup Guide](SETUP.md).

## Available Tools

### 1. **get_ambient_conditions**
Retrieve weather data for any location using coordinates or city names.

### 2. **get_fluid_properties**
Get temperature-dependent properties (density, viscosity, Cp, etc.) for fluids.

### 3. **get_material_properties**
Access thermal conductivity for 390+ materials with fuzzy name matching.

### 4. **calculate_convection_coefficient**
Calculate convection heat transfer coefficients for various geometries.

### 5. **calculate_overall_heat_transfer_coefficient**
Determine overall U-values including fouling factors.

### 6. **calculate_surface_heat_transfer**
Analyze heat loss from tanks, pipes, and buildings.

### 7. **calculate_heat_exchanger_performance**
Evaluate heat exchanger effectiveness and outlet temperatures.

### 8. **calculate_heat_duty**
Calculate heating/cooling requirements for processes.

### 9. **calculate_solar_radiation_on_surface**
Determine solar heat gain on surfaces.

### 10. **calculate_ground_heat_loss**
Analyze heat loss through foundations and slabs.

### 11. **calculate_buried_object_heat_loss**
Calculate heat loss from buried pipes and ducts.

### 12. **calculate_hx_shell_side_h_kern**
Determine shell-side heat transfer coefficients using Kern method.

### 13. **size_heat_exchanger_area**
Size heat exchangers based on duty requirements.

### 14. **estimate_hx_physical_dims**
Estimate tube counts and bundle dimensions.

## Unit Conversion Examples

The server automatically converts imperial units to SI:

```python
# Temperature
"95 degF" → 308.15 K
"35 degC" → 308.15 K

# Flow Rate  
"200 GPM" → 12.62 kg/s
"5000 SCFM" → 2.97 kg/s

# Length
"10 feet" → 3.048 m
"12 inches" → 0.3048 m

# Pressure
"14.7 psi" → 101325 Pa
"1 bar" → 100000 Pa

# Power
"1000 BTU/hr" → 293 W
"10 HP" → 7457 W
```

## Example Usage

### Digester Heat Loss
```python
calculate_surface_heat_transfer(
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
```

### Heat Duty Calculation
```python
calculate_heat_duty(
    calculation_method="sensible_heat",
    fluid_name="water",
    flow_rate="200 GPM",
    inlet_temp="60 degF",
    outlet_temp="95 degF"
)
```

## Wastewater Engineering Features

- Fouling factors for wastewater applications
- Sludge property approximations
- Typical operating temperatures for digesters
- Soil thermal conductivity values
- Common pipe sizes and insulation thicknesses

## Claude Desktop Integration

For use with Claude Desktop, update your configuration:

```json
{
  "mcpServers": {
    "heat-transfer": {
      "command": "C:\\path\\to\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\heat-transfer-mcp\\server.py"]
    }
  }
}
```

## Documentation

- [Setup Guide](SETUP.md) - Installation and configuration instructions
- [System Prompt Guide](docs/SYSTEM_PROMPT_ENHANCED.md) - Comprehensive usage guide
- [Tool Parameter Analysis](docs/TOOL_PARAMETER_ANALYSIS.md) - Detailed parameter documentation

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built on the [ht](https://github.com/CalebBell/ht) library by Caleb Bell
- Material properties from VDI Heat Atlas and ASHRAE Handbook
- Developed as an MCP server for thermal engineering calculations