"""
Constants used across the Heat Transfer MCP server.

This module defines unit conversion factors and other constants used throughout the server.
"""

# Conversion factors for unit flexibility
GPM_to_M3S = 0.0000630902    # US GPM to m³/s
INCH_to_M = 0.0254           # inch to meter
PSI_to_PA = 6894.76          # psi to Pascal
CENTIPOISE_to_PAS = 0.001    # centipoise to Pa·s
FT_to_M = 0.3048             # foot to meter
LBFT3_to_KGM3 = 16.0185      # lb/ft³ to kg/m³
DEG_C_to_K = 273.15          # Celsius to Kelvin (offset)
DEG_F_to_C = 5.0/9.0         # Fahrenheit to Celsius conversion factor
DEG_F_to_K = 5.0/9.0         # Fahrenheit to Kelvin conversion factor (after offset)

# Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.67e-8   # Stefan-Boltzmann constant, W/(m²·K⁴)

# Standard atmospheric conditions
P_ATM = 101325.0             # Standard atmospheric pressure, Pa (1 atm)
T_STD = 288.15               # Standard temperature, K (15°C)

# Default values for calculations
DEFAULT_ROUGHNESS = 1.5e-5   # Default pipe roughness, m
DEFAULT_EMISSIVITY = 0.9     # Default surface emissivity (dimensionless)
DEFAULT_ABSORPTIVITY = 0.7   # Default surface absorptivity for solar radiation (dimensionless)
DEFAULT_GROUND_ALBEDO = 0.2  # Default ground reflectivity (dimensionless)

# Physical constants
GRAVITY = 9.80665            # Standard gravity, m/s²
AIR_THERMAL_CONDUCTIVITY = 0.026  # Approximate thermal conductivity of air at 300K, W/(m·K)
