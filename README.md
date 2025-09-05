# Heat Transfer MCP Server

[![MCP](https://img.shields.io/badge/MCP-1.0-blue)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![HT](https://img.shields.io/badge/HT-1.0.2-green)](https://github.com/CalebBell/ht)
[![Thermo](https://img.shields.io/badge/Thermo-0.3.0-green)](https://github.com/CalebBell/thermo)
[![Fluids](https://img.shields.io/badge/Fluids-1.0.26-green)](https://github.com/CalebBell/fluids)
[![Meteostat](https://img.shields.io/badge/Meteostat-1.6.8-orange)](https://meteostat.net)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success)](https://github.com/puran-water/heat-transfer-mcp)

Model Context Protocol server providing comprehensive heat transfer and thermal system analysis for industrial process engineering.

## Overview

This MCP server provides 5 consolidated omnitools that encapsulate thermal engineering calculations:

1. **tank_heat_loss**: Comprehensive tank/vessel heat loss with iterative surface temperature solver
2. **heat_exchanger_design**: Integrated HX sizing with tank heat loss calculations
3. **pipe_heat_management**: Pipe insulation, heat trace, and freeze protection analysis
4. **parameter_optimization**: Grid-search optimization for thermal system design
5. **extreme_conditions**: Historical weather extremes and percentile-based design conditions

## Technical Capabilities

### Heat Transfer Analysis
- **Iterative Surface Temperature Solver**: Newton-Raphson with adaptive damping for convergence
- **Multi-Layer Walls**: Composite wall thermal resistance with automatic material lookup
- **Convection Correlations**: Natural/forced convection switching based on flow conditions
- **Radiation Effects**: Stefan-Boltzmann radiation with linearization for stability
- **Headspace Modeling**: Two-zone model for tanks with gas space above liquid level

### Thermodynamic Properties
- **HT Library Integration**: NIST-validated correlations for convection coefficients
- **Thermo/Fluids Integration**: Temperature-dependent fluid properties
- **Material Database**: 390+ materials with temperature-dependent thermal conductivity
- **Mixture Support**: Composition-based property calculation for gas mixtures
- **Unit Safety**: Explicit SI units throughout with automatic conversion

### Weather Integration
- **Meteostat API**: Historical weather data for any global location
- **Percentile Analysis**: Design conditions based on 90th, 95th, 99th percentiles
- **Cold Design Logic**: Proper lower-tail percentiles for heating applications
- **Concurrent Extremes**: Wind speed during cold periods for conservative design

### Optimization Capabilities
- **Parameter Sweeps**: Systematic variation of design parameters
- **Diminishing Returns Analysis**: Identify optimal insulation thickness
- **Constraint-Based Design**: Meet target heat loss with minimum insulation
- **Multi-Variable Analysis**: Cartesian product of parameter combinations

## Recent Improvements (December 2024)

### Critical Validation Framework
- **New**: Comprehensive input validation module preventing physically invalid parameters
- **Fixed**: Negative dimensions, flow rates, and R-values now properly rejected
- **Fixed**: Temperature crossing detection for heat exchangers with detailed error messages
- **Fixed**: Geographic coordinate bounds validation (latitude ±90°, longitude ±180°)
- **Fixed**: Date range validation ensuring start_date ≤ end_date

### Area/Resistance Calculation Fixes
- **Fixed**: Cylindrical resistance calculation error causing 1000x heat loss overestimation
- **Fixed**: Vertical tank bottom now properly excluded from air-exposed area (ground contact)
- **Fixed**: Double-counting of gas-liquid interface area in headspace calculations
- **Fixed**: Automatic ground heat loss calculation for vertical cylindrical tanks

### Heat Trace Sizing Improvements
- **New**: Steady-state heat trace mode returns actual loss at maintenance temperature
- **New**: Dedicated `freeze_protection_w_per_m` mode with automatic safety factors
- **New**: Catalog rounding to standard heat trace ratings (5, 10, 15, 20, 25 W/m)
- **New**: Clear output fields distinguishing steady-state vs delta power requirements
- **Fixed**: Heat trace now correctly sized as steady-state loss at target temperature

### Headspace Modeling
- **New**: Two-zone model for digesters and tanks with gas headspace
- **New**: Separate heat loss calculation for wetted walls vs gas-exposed surfaces
- **New**: Configurable inner convection coefficient for gas spaces (default 5 W/m²K)
- **Fixed**: Headspace convection coefficient now properly applied via override parameter

## Installation

### Prerequisites
- Python 3.10+ (3.12 tested)
- Virtual environment
- MCP client (Claude Desktop or compatible)

### Setup
```bash
git clone https://github.com/puran-water/heat-transfer-mcp.git
cd heat-transfer-mcp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## MCP Configuration

### Claude Desktop Integration

Add to your configuration file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "heat-transfer-mcp": {
      "command": "python",
      "args": ["/absolute/path/to/heat-transfer-mcp/server.py"],
      "env": {
        "HEAT_TRANSFER_MCP_ROOT": "/absolute/path/to/heat-transfer-mcp"
      }
    }
  }
}
```

## API Reference

### Registered Omnitools

#### tank_heat_loss
Comprehensive tank heat loss analysis with iterative surface temperature solver.

**Parameters:**
- `geometry`: Tank geometry ("vertical_cylinder_tank", "horizontal_cylinder_tank", "sphere")
- `dimensions`: Dictionary with diameter, height/length in meters
- `contents_temperature`: Internal temperature (K)
- `headspace_height_m`: Height of gas space above liquid (m) - for digesters/tanks
- `headspace_fluid`: Gas in headspace ("air", "biogas", etc.)
- `insulation_R_value_si`: Thermal resistance (m²K/W)
- `latitude`, `longitude`: Location for weather data
- `design_percentile`: Weather percentile (0.90, 0.95, 0.99)
- `solve_for`: Optional ("R_value" to find required insulation)

#### heat_exchanger_design
Size heat exchangers integrated with tank heat loss calculations.

**Parameters:**
- `include_tank_loss`: Calculate tank losses and add to duty
- `tank_params`: Parameters for tank_heat_loss tool
- `process_fluid`: Fluid being heated/cooled
- `process_mass_flow_kg_s`: Flow rate (kg/s)
- `heating_inlet_temp_K`: Hot fluid inlet temperature
- `overall_U_W_m2K`: Overall heat transfer coefficient

#### pipe_heat_management
Pipe insulation and heat trace calculations with freeze protection sizing.

**Parameters:**
- `outer_diameter_m`: Pipe outer diameter including insulation (m)
- `length_m`: Pipe length (m)
- `internal_temperature_K`: Fluid temperature (K)
- `wall_layers`: Insulation layers with thickness and conductivity
- `solve_for`: Calculation modes:
  - `"heat_trace_w_per_m"`: Steady-state heat trace at target temperature
  - `"freeze_protection_w_per_m"`: Heat trace with safety factor and catalog rounding
  - `"heat_trace_delta_w_per_m"`: Delta power between temperatures (informational)
  - `"freeze_time_h"`: Time to freeze without heat trace
- `target_temperature_K`: Maintenance temperature for heat trace sizing
- `heat_trace_safety_factor`: Safety factor for recommendations (default 1.25)
- `installation`: "above_ground" or "buried"

#### parameter_optimization
Grid-search optimization for thermal systems.

**Parameters:**
- `tool_name`: Tool to optimize ("tank_heat_loss", etc.)
- `base_params`: Baseline parameters
- `sweep`: Dictionary of parameter→values to sweep
- `objective_key`: Result key to optimize
- `direction`: "minimize" or "maximize"

#### extreme_conditions
Weather extremes and design day selection.

**Parameters:**
- `latitude`, `longitude`: Location coordinates
- `start_date`, `end_date`: Date range (YYYY-MM-DD)
- `percentiles`: List of percentiles [0.90, 0.95, 0.99]
- `include_wind`: Include wind analysis
- `time_resolution`: "daily" or "hourly"

## Validation Examples

### Anaerobic Digester with Ground Contact
```python
tank_heat_loss(
    geometry="vertical_cylinder_tank",
    dimensions={"diameter": 24.69, "height": 10.06},  # 81ft diameter
    contents_temperature=310.93,  # 100°F
    headspace_height_m=1.83,  # 6ft biogas space
    headspace_fluid="biogas",
    headspace_h_inner_override_w_m2k=5.0,  # Stagnant gas
    insulation_R_value_si=1.76,  # R-10 imperial
    ambient_air_temperature=264.15,  # -9°C design
    wind_speed=6.63,
    average_external_air_temperature=283.15  # Annual average
)
# Result: 32 kW total (30.9 kW above-ground, 1.5 kW ground)
```

### Pipe Freeze Protection Sizing
```python
pipe_heat_management(
    outer_diameter_m=0.1651,  # 4" pipe + 1" insulation
    length_m=30,
    internal_temperature_K=288.71,  # 60°F normal
    ambient_air_temperature_K=258.15,  # 5°F extreme
    wind_speed_m_s=8.14,
    wall_layers=[{"thickness": 0.0254, "thermal_conductivity_k": 0.043}],
    solve_for="freeze_protection_w_per_m"
)
# Result: 16.4 W/m required, 25 W/m catalog selection with safety factor
```

### Insulation Optimization
```python
parameter_optimization(
    tool_name="tank_heat_loss",
    base_params={
        "geometry": "vertical_cylinder_tank",
        "dimensions": {"diameter": 10, "height": 8},
        "contents_temperature": 350,
        "ambient_air_temperature": 270
    },
    sweep={"insulation_R_value_si": [0, 0.5, 1, 2, 3, 4, 5, 10]},
    objective_key="total_heat_loss_w",
    direction="minimize"
)
# Result: Optimal at R=3-4, diminishing returns above (△Q < 10% per R-unit)
```

## Performance Characteristics

### Convergence Behavior
- Surface temperature typically converges in 10-25 iterations
- Adaptive damping prevents oscillation
- Tolerance: 1.0 W energy balance
- Temperature bounds: [T_ambient-50K, T_internal+50K]

### Computational Efficiency
- Single tank evaluation: ~100ms
- Parameter sweep (15 points): ~2s
- Weather percentile analysis: ~500ms

### Accuracy Validation
- Headspace model reduces heat loss estimates by 15-30% vs fully-wetted assumption
- Matches engineering references within ±5%
- Proper handling of natural convection at low wind speeds

## Dependencies

Core libraries:
- `mcp>=1.1.0`: Model Context Protocol
- `ht>=1.0.2`: Heat transfer correlations
- `thermo>=0.3.0`: Thermodynamic properties
- `fluids>=1.0.26`: Fluid mechanics
- `meteostat>=1.6.8`: Weather data
- `pandas>=2.2.0`: Data analysis

## Testing

Run validation suite:
```bash
python -m pytest tests/
```

Key test cases:
- Convergence for extreme R-values (0.1 to 20)
- Headspace modeling validation
- Weather percentile calculations
- Cylindrical resistance calculations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built on [ht](https://github.com/CalebBell/ht), [thermo](https://github.com/CalebBell/thermo), and [fluids](https://github.com/CalebBell/fluids) by Caleb Bell
- Weather data from [Meteostat](https://meteostat.net)
- Developed for industrial thermal system design