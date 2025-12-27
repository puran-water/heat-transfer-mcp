# Heat Transfer MCP Server

[![MCP](https://img.shields.io/badge/MCP-1.0-blue)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![HT](https://img.shields.io/badge/HT-1.2.0-green)](https://github.com/CalebBell/ht)
[![Thermo](https://img.shields.io/badge/Thermo-0.6.0-green)](https://github.com/CalebBell/thermo)
[![Fluids](https://img.shields.io/badge/Fluids-1.3.0-green)](https://github.com/CalebBell/fluids)
[![CoolProp](https://img.shields.io/badge/CoolProp-Optional-lightgrey)](http://www.coolprop.org/)
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
- **Primary Data Source**: Meteostat API for historical weather observations
- **Secondary Data Source**: Open-Meteo ERA5 reanalysis for missing atmospheric parameters
- **Percentile Analysis**: Statistical design conditions at 90th, 95th, 99th percentiles
- **Cold Design Methodology**: Lower-tail percentile selection for heating load calculations
- **Concurrent Analysis**: Joint probability assessment of temperature and wind speed extremes
- **Context Optimization**: Compressed data transmission preserving essential design values

### Optimization Capabilities
- **Parameter Sweeps**: Systematic variation of design parameters
- **Diminishing Returns Analysis**: Identify optimal insulation thickness
- **Constraint-Based Design**: Meet target heat loss with minimum insulation
- **Multi-Variable Analysis**: Cartesian product of parameter combinations

## Technical Enhancements

### Radiation Heat Transfer Corrections
- Corrected sky temperature estimation formula to use fourth-root relationship with atmospheric emissivity
- Implemented view factor methodology for radiation exchange between surfaces and environment
- Incorporated ground temperature effects in radiation balance for vertical cylindrical geometries
- Applied separate view factors: 0.5 for vertical surfaces, 1.0 for horizontal surfaces
- Achieved 19% reduction in calculated radiation heat loss through improved physical modeling

### Weather Data Integration Architecture
- Implemented hierarchical data source fallback: primary (Meteostat), secondary (ERA5 reanalysis), tertiary (empirical correlations)
- Integrated Open-Meteo ERA5 reanalysis for dew point temperature when station observations unavailable
- Optimized data retrieval with single-fetch architecture for multiple percentile calculations
- Implemented context-aware data compression returning essential design parameters only
- Cached weather queries to minimize external API calls and improve response time

### Enhanced Output Transparency
- Added comprehensive weather data provenance tracking in output structure
- Incorporated radiation model parameters including view factors and effective environment temperature
- Implemented diagnostic warnings for non-physical results (e.g., surface temperature below ambient)
- Restructured extreme conditions output to report data availability without transmitting raw datasets

### Critical Validation Framework
- Implemented comprehensive input validation module preventing physically invalid parameters
- Added dimension validation rejecting negative values for physical quantities
- Incorporated temperature crossing detection for heat exchangers with detailed diagnostics
- Validated geographic coordinates within physical bounds (latitude ±90°, longitude ±180°)
- Enforced temporal consistency in date range specifications

### Heat Transfer Calculation Improvements
- Corrected cylindrical thermal resistance calculation methodology
- Excluded tank bottom surface from convective area for ground-coupled configurations
- Eliminated double-counting of interfacial area in two-zone headspace model
- Implemented automatic ground heat loss calculation for vertical cylindrical vessels

### Heat Trace System Design
- Implemented steady-state heat trace sizing at maintenance temperature conditions
- Added dedicated freeze protection mode with configurable safety factors
- Incorporated catalog-based sizing to standard ratings (5, 10, 15, 20, 25 W/m)
- Differentiated output fields for steady-state versus transient power requirements

### Two-Zone Tank Modeling
- Developed separate heat transfer models for liquid-wetted and gas-exposed surfaces
- Implemented zone-specific convection coefficients for improved accuracy
- Added configurable internal heat transfer coefficient for gas spaces (default 5 W/m²K)
- Validated headspace convection coefficient application through parameter override system

### TEMA Heat Exchanger Standards
- Integrated `ht.hx` TEMA functions for professional-grade shell-and-tube design
- Added tube validation with `check_tubing_TEMA` for standard NPS/BWG combinations
- Implemented `DBundle_min` for minimum bundle diameter sizing
- Added `shell_clearance` lookup for TEMA-compliant shell-bundle clearances
- Baffle thickness calculation per TEMA standards for various service classes
- Multi-shell F_LMTD correction factor using Fakheri correlation

### Two-Phase Heat Transfer
- Added `calculate_two_phase_h` tool for boiling/condensation heat transfer
- Supports 9+ correlations: Shah, Chen, Kandlikar, Liu-Winterton, Thome, Sun-Mishima, etc.
- Automatic method selection based on available parameters
- CoolProp integration for accurate vapor properties when available

### High-Accuracy Fluid Properties (CoolProp)
- Optional CoolProp integration for REFPROP-quality thermodynamic properties
- Automatic fallback to thermo library when CoolProp unavailable
- Reference equations of state for common refrigerants and industrial fluids
- Install with: `pip install CoolProp`

### Tank Geometry Integration
- Integrated `fluids.geometry.TANK` for accurate tank volume and surface area
- Supports ellipsoidal, torispherical, conical, and spherical heads
- Partial fill calculations with `V_from_h` and `SA_from_h` methods
- ASME F&D and DIN 28011 standard head configurations

### Meteostat Data Quality Assurance
- Documented known upstream data quality issues (#202, #201, #148, #171)
- Implemented minimum data point validation (warns if < 2 years of data)
- Data coverage analysis with warnings for gaps > 20%
- Automatic ERA5 fallback for missing dew point observations

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
- `mcp>=0.5.0`: Model Context Protocol
- `ht>=1.2.0`: Heat transfer correlations (TEMA, two-phase, Numba optimizations)
- `thermo>=0.6.0`: Thermodynamic properties
- `fluids>=1.3.0`: Fluid mechanics and tank geometry
- `chemicals>=1.4.0`: Chemical property database
- `meteostat>=1.6.8`: Weather data
- `pandas>=2.0.0`: Data analysis
- `cachetools>=5.3.0`: Weather data caching

Optional (high-accuracy properties):
- `CoolProp>=6.4.0`: REFPROP-quality equations of state (install separately)

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