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

## Critical Bug Fixes (December 2024)

### Surface Heat Transfer Solver
- **Fixed**: Cylindrical resistance double-counting causing convergence failures
- **Fixed**: Vertical cylinder area calculation now includes both endcaps
- **Fixed**: Dimensional consistency in convergence tolerance (Watts instead of Kelvin)
- **Fixed**: Natural convection switching for wind speeds ≤ 0.2 m/s
- **Fixed**: Pipe geometry now uses cylindrical conduction model

### Weather and Percentiles
- **Fixed**: Date parsing for Meteostat API (string to datetime conversion)
- **Fixed**: Cold percentile logic (p99 cold design uses 0.01 quantile)
- **Fixed**: Concurrent extremes calculation for cold+windy conditions

### Headspace Modeling
- **New**: Two-zone model for digesters and tanks with gas headspace
- **New**: Separate heat loss calculation for wetted walls vs gas-exposed surfaces
- **New**: Configurable inner convection coefficient for gas spaces (default 5 W/m²K)

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
Pipe insulation and heat trace calculations.

**Parameters:**
- `outer_diameter_m`: Pipe outer diameter including insulation
- `length_m`: Pipe length
- `internal_temperature_K`: Fluid temperature
- `wall_layers`: Insulation layers
- `solve_for`: Optional ("heat_trace_w_per_m", "freeze_time_h")
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

### Anaerobic Digester with Headspace
```python
tank_heat_loss(
    geometry="vertical_cylinder_tank",
    dimensions={"diameter": 15, "height": 10},
    contents_temperature=311.15,  # 38°C
    headspace_height_m=1.5,  # Biogas space
    headspace_fluid="biogas",
    insulation_R_value_si=3.0,
    latitude=40.7128,
    longitude=-74.0060,
    design_percentile=0.99
)
# Result: 508 kW total loss (362 kW wetted, 146 kW headspace)
```

### Insulation Optimization
```python
parameter_optimization(
    tool_name="tank_heat_loss",
    base_params={...},
    sweep={"insulation_R_value_si": [0, 1, 2, 3, 4, 5, 10, 20]},
    objective_key="total_heat_loss_w",
    direction="minimize"
)
# Result: Diminishing returns above R=4 (6-7 kW saved per R-unit)
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