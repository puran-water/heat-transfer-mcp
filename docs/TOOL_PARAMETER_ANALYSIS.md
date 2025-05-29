# Heat Transfer Tool Parameter Analysis for Unit Conversion

## Overview
Analysis of all 14 tools to identify parameters that commonly need unit conversion from imperial to SI units.

## Tool Analysis

### 1. get_ambient_conditions
**Parameters needing conversion:**
- `latitude`, `longitude`: degrees (no conversion needed)
- `start_date`, `end_date`: dates (no conversion needed)
- **No unit conversion needed**

### 2. get_fluid_properties
**Parameters needing conversion:**
- `temperature`: K (from °F, °C)
- `pressure`: Pa (from psi, bar, atm)

### 3. get_material_properties
**Parameters needing conversion:**
- `temperature`: K (from °F, °C) [optional]
- **Mostly material names, minimal conversion needed**

### 4. calculate_convection_coefficient
**Parameters needing conversion:**
- `temperature`: K (from °F, °C)
- `velocity`: m/s (from ft/s, mph)
- `characteristic_length`: m (from ft, in)
- `surface_area`: m² (from ft²)
- `surface_temperature`: K (from °F, °C)
- `bulk_temperature`: K (from °F, °C)
- `hydraulic_diameter`: m (from ft, in)
- `pipe_diameter`: m (from ft, in)
- `flow_area`: m² (from ft²)

### 5. calculate_overall_heat_transfer_coefficient
**Parameters needing conversion:**
- `inside_htc`: W/m²K (from BTU/hr·ft²·°F)
- `outside_htc`: W/m²K (from BTU/hr·ft²·°F)
- `wall_thickness`: m (from ft, in)
- `wall_thermal_conductivity`: W/mK (from BTU/hr·ft·°F)
- `inside_fouling_factor`: m²K/W
- `outside_fouling_factor`: m²K/W
- `inside_diameter`: m (from ft, in)
- `outside_diameter`: m (from ft, in)

### 6. calculate_surface_heat_transfer
**Parameters needing conversion:**
- `dimensions`: dict with lengths in m (from ft, in)
  - `length`, `width`, `height`, `diameter`, `radius`
- `internal_temperature`: K (from °F, °C)
- `ambient_air_temperature`: K (from °F, °C)
- `wind_speed`: m/s (from ft/s, mph)
- `wall_layers`: list of dicts with `thickness` in m (from ft, in)

### 7. calculate_heat_exchanger_performance
**Parameters needing conversion:**
- `hot_inlet_temp`: K (from °F, °C)
- `hot_outlet_temp`: K (from °F, °C)
- `cold_inlet_temp`: K (from °F, °C)
- `cold_outlet_temp`: K (from °F, °C)
- `hot_flow_rate`: kg/s (from lb/hr, GPM for liquids)
- `cold_flow_rate`: kg/s (from lb/hr, GPM for liquids)
- `heat_transfer_area`: m² (from ft²)
- `overall_htc`: W/m²K (from BTU/hr·ft²·°F)

### 8. calculate_heat_duty
**Parameters needing conversion:**
- `flow_rate`: kg/s (from lb/hr, SCFM, GPM)
- `inlet_temp`: K (from °F, °C)
- `outlet_temp`: K (from °F, °C)
- `fluid_pressure`: Pa (from psi, bar)
- `overall_heat_transfer_coefficient_U`: W/m²K
- `heat_transfer_area`: m² (from ft²)
- `mean_temperature_difference`: K (from °F, °C)

### 9. calculate_solar_radiation_on_surface
**Parameters needing conversion:**
- `surface_area`: m² (from ft²)
- `ambient_temperature`: K (from °F, °C)
- **Other parameters are ratios or angles**

### 10. calculate_ground_heat_loss
**Parameters needing conversion:**
- `foundation_length`: m (from ft)
- `foundation_width`: m (from ft)
- `foundation_depth`: m (from ft)
- `wall_thickness`: m (from ft, in)
- `floor_thickness`: m (from ft, in)
- `inside_temperature`: K (from °F, °C)
- `outside_temperature`: K (from °F, °C)
- `ground_temperature`: K (from °F, °C)
- `wall_thermal_conductivity`: W/mK
- `floor_thermal_conductivity`: W/mK
- `soil_thermal_conductivity`: W/mK

### 11. calculate_buried_object_heat_loss
**Parameters needing conversion:**
- `diameter`: m (from ft, in) [for pipes]
- `width`: m (from ft) [for ducts]
- `height`: m (from ft) [for ducts]
- `length`: m (from ft)
- `burial_depth`: m (from ft)
- `object_temperature`: K (from °F, °C)
- `ground_surface_temperature`: K (from °F, °C)
- `soil_conductivity`: W/mK

### 12. calculate_hx_shell_side_h_kern
**Parameters needing conversion:**
- `mass_flow_rate`: kg/s (from lb/hr, GPM)
- `tube_outer_diameter`: m (from in)
- `shell_inner_diameter`: m (from in)
- `tube_pitch`: m (from in)
- `baffle_spacing`: m (from in)
- `baffle_cut`: fraction (no conversion)
- `bulk_temperature`: K (from °F, °C)
- `wall_temperature`: K (from °F, °C)

### 13. size_heat_exchanger_area
**Parameters needing conversion:**
- `required_heat_duty_q`: W (from BTU/hr, HP)
- `hot_fluid_flow_rate`: kg/s (from lb/hr, GPM)
- `hot_fluid_inlet_temp`: K (from °F, °C)
- `cold_fluid_flow_rate`: kg/s (from lb/hr, GPM)
- `cold_fluid_inlet_temp`: K (from °F, °C)
- `tube_outer_diameter`: m (from in)
- `tube_inner_diameter`: m (from in)
- `tube_material_conductivity`: W/mK
- `fouling_factor_inner`: m²K/W
- `fouling_factor_outer`: m²K/W

### 14. estimate_hx_physical_dims
**Parameters needing conversion:**
- `required_area`: m² (from ft²)
- `tube_outer_diameter`: m (from in)
- `tube_inner_diameter`: m (from in)
- `shell_inner_diameter`: m (from in)
- `baffle_spacing`: m (from in)
- `tube_pitch`: m (from in)
- `tube_length_options`: list of m (from ft)

## Common Parameter Types

### Temperature Parameters (°F/°C → K)
- temperature, inlet_temp, outlet_temp, hot_inlet_temp, cold_inlet_temp
- internal_temperature, ambient_temperature, surface_temperature
- bulk_temperature, wall_temperature, inside_temperature, outside_temperature
- ground_temperature, object_temperature, ground_surface_temperature

### Length Parameters (ft/in → m)
- characteristic_length, wall_thickness, diameter, radius
- length, width, height, hydraulic_diameter, pipe_diameter
- tube_outer_diameter, tube_inner_diameter, shell_inner_diameter
- baffle_spacing, tube_pitch, burial_depth, foundation_depth

### Area Parameters (ft² → m²)
- surface_area, flow_area, heat_transfer_area, required_area

### Flow Rate Parameters (GPM/SCFM/lb/hr → kg/s or m³/s)
- flow_rate, hot_flow_rate, cold_flow_rate, mass_flow_rate
- hot_fluid_flow_rate, cold_fluid_flow_rate

### Pressure Parameters (psi/bar → Pa)
- pressure, fluid_pressure

### Velocity Parameters (ft/s/mph → m/s)
- velocity, wind_speed

### Power/Heat Parameters (BTU/hr/HP → W)
- required_heat_duty_q

### Heat Transfer Coefficient (BTU/hr·ft²·°F → W/m²K)
- inside_htc, outside_htc, overall_htc
- overall_heat_transfer_coefficient_U

### Thermal Conductivity (BTU/hr·ft·°F → W/mK)
- wall_thermal_conductivity, tube_material_conductivity
- floor_thermal_conductivity, soil_thermal_conductivity