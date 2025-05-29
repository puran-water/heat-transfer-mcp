"""Common engineering values for wastewater treatment applications.

This module provides typical values and constants used in wastewater treatment
heat transfer calculations, including temperatures, fouling factors, heat transfer
coefficients, and material properties.
"""

import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger("heat-transfer-mcp.wastewater")

# Typical operating temperatures (K)
TEMPERATURES = {
    'mesophilic_digester': 308.15,  # 35°C (95°F)
    'thermophilic_digester': 328.15,  # 55°C (131°F)
    'activated_sludge_summer': 298.15,  # 25°C (77°F)
    'activated_sludge_winter': 283.15,  # 10°C (50°F)
    'hot_water_heating': 363.15,  # 90°C (194°F)
    'warm_water_heating': 333.15,  # 60°C (140°F)
    'primary_clarifier': 288.15,  # 15°C (59°F)
    'raw_wastewater_summer': 293.15,  # 20°C (68°F)
    'raw_wastewater_winter': 283.15,  # 10°C (50°F)
}

# Fouling factors (m²K/W)
FOULING_FACTORS = {
    'clean_water': 0.0001,
    'treated_effluent': 0.0002,
    'raw_wastewater': 0.0003,
    'primary_sludge': 0.0005,
    'digested_sludge': 0.0004,
    'thickened_sludge': 0.0006,
    'biogas': 0.0001,
    'cooling_water': 0.00035,
    'river_water': 0.0004,
    'boiler_water': 0.0001,
}

# Typical heat transfer coefficients (W/m²K) - (min, max)
HEAT_TRANSFER_COEFFICIENTS = {
    'water_in_tubes': (1000, 3000),
    'sludge_in_tubes': (500, 1500),
    'biogas_in_tubes': (50, 150),
    'air_in_tubes': (50, 200),
    'steam_condensing': (5000, 10000),
    'tank_to_air_natural': (5, 15),
    'tank_to_air_forced': (10, 30),
    'jacketed_vessel_water': (300, 800),
    'jacketed_vessel_sludge': (200, 500),
}

# Soil thermal conductivity (W/mK)
SOIL_CONDUCTIVITY = {
    'dry_sand': 0.3,
    'dry_clay': 0.5,
    'moist_sand': 2.0,
    'moist_clay': 1.3,
    'saturated_sand': 2.4,
    'saturated_clay': 1.5,
    'average_moist': 1.5,
    'frozen_soil': 2.2,
    'rock': 3.5,
}

# Sludge properties multipliers (relative to water)
SLUDGE_PROPERTIES = {
    'primary_3_percent': {
        'density_factor': 1.02,
        'viscosity_factor': 2.0,
        'cp_factor': 0.98,
        'k_factor': 0.95,
    },
    'thickened_6_percent': {
        'density_factor': 1.05,
        'viscosity_factor': 5.0,
        'cp_factor': 0.95,
        'k_factor': 0.90,
    },
    'digested_4_percent': {
        'density_factor': 1.03,
        'viscosity_factor': 3.0,
        'cp_factor': 0.97,
        'k_factor': 0.93,
    },
    'waste_activated_2_percent': {
        'density_factor': 1.01,
        'viscosity_factor': 1.5,
        'cp_factor': 0.99,
        'k_factor': 0.97,
    },
}

# Common pipe sizes (inches to meters)
PIPE_SIZES = {
    '2_inch': 0.0508,
    '3_inch': 0.0762,
    '4_inch': 0.1016,
    '6_inch': 0.1524,
    '8_inch': 0.2032,
    '10_inch': 0.254,
    '12_inch': 0.3048,
    '14_inch': 0.3556,
    '16_inch': 0.4064,
    '18_inch': 0.4572,
    '20_inch': 0.508,
    '24_inch': 0.6096,
    '30_inch': 0.762,
    '36_inch': 0.9144,
}

# Typical insulation thicknesses (inches)
INSULATION_THICKNESS = {
    'digester_roof': 4,  # inches
    'digester_wall': 3,  # inches
    'hot_water_pipe_small': 1.5,  # inches (< 4")
    'hot_water_pipe_large': 2,  # inches (>= 4")
    'sludge_pipe': 2,  # inches
    'tank_wall': 3,  # inches
}

# Heat loss factors (W/m² for tanks, W/m for pipes)
TYPICAL_HEAT_LOSSES = {
    'uninsulated_digester': 150,  # W/m² at ΔT = 25°C
    'insulated_digester': 15,  # W/m² at ΔT = 25°C
    'uninsulated_pipe_4in': 100,  # W/m at ΔT = 50°C
    'insulated_pipe_4in': 20,  # W/m at ΔT = 50°C
    'buried_pipe_12in': 50,  # W/m at ΔT = 40°C
}

# Process heat requirements (kW)
TYPICAL_HEAT_DUTIES = {
    'digester_heating_per_m3': 0.5,  # kW/m³ digester volume
    'sludge_heating_per_kg': 0.15,  # kW per kg/s sludge flow
    'building_heating_per_m2': 0.1,  # kW/m² floor area
}

def get_sludge_properties(base_fluid_props: Dict, sludge_type: str = 'primary_3_percent') -> Dict:
    """Adjust water properties for sludge based on solids content.
    
    Args:
        base_fluid_props: Dictionary of water properties
        sludge_type: Type of sludge (see SLUDGE_PROPERTIES keys)
        
    Returns:
        Dictionary of adjusted properties for sludge
    """
    if sludge_type not in SLUDGE_PROPERTIES:
        logger.warning(f"Unknown sludge type '{sludge_type}', using primary_3_percent")
        sludge_type = 'primary_3_percent'
        
    factors = SLUDGE_PROPERTIES[sludge_type]
    
    adjusted_props = base_fluid_props.copy()
    
    # Apply factors
    if 'density' in adjusted_props:
        adjusted_props['density'] *= factors['density_factor']
    
    if 'dynamic_viscosity' in adjusted_props:
        adjusted_props['dynamic_viscosity'] *= factors['viscosity_factor']
        
    if 'specific_heat_cp' in adjusted_props:
        adjusted_props['specific_heat_cp'] *= factors['cp_factor']
        
    if 'thermal_conductivity' in adjusted_props:
        adjusted_props['thermal_conductivity'] *= factors['k_factor']
    
    # Recalculate kinematic viscosity
    if 'density' in adjusted_props and 'dynamic_viscosity' in adjusted_props:
        adjusted_props['kinematic_viscosity'] = (
            adjusted_props['dynamic_viscosity'] / adjusted_props['density']
        )
    
    # Add sludge type info
    adjusted_props['fluid_name'] = f"Sludge ({sludge_type.replace('_', ' ')})"
    adjusted_props['base_fluid'] = 'water'
    
    return adjusted_props

def get_fouling_factor(application: str, conservative: bool = True) -> float:
    """Get recommended fouling factor for an application.
    
    Args:
        application: Type of application (see FOULING_FACTORS keys)
        conservative: If True, add 50% safety factor
        
    Returns:
        Fouling factor in m²K/W
    """
    if application not in FOULING_FACTORS:
        logger.warning(f"Unknown application '{application}', using raw_wastewater")
        application = 'raw_wastewater'
    
    ff = FOULING_FACTORS[application]
    
    if conservative:
        ff *= 1.5
        
    return ff

def get_heat_transfer_coefficient(application: str, use_average: bool = True) -> float:
    """Get typical heat transfer coefficient for an application.
    
    Args:
        application: Type of application (see HEAT_TRANSFER_COEFFICIENTS keys)
        use_average: If True, return average of min/max; else return tuple
        
    Returns:
        Heat transfer coefficient in W/m²K
    """
    if application not in HEAT_TRANSFER_COEFFICIENTS:
        logger.warning(f"Unknown application '{application}'")
        return 500 if use_average else (250, 750)
    
    min_val, max_val = HEAT_TRANSFER_COEFFICIENTS[application]
    
    if use_average:
        return (min_val + max_val) / 2
    else:
        return (min_val, max_val)

def get_soil_conductivity(soil_type: str, moisture_content: Optional[str] = None) -> float:
    """Get soil thermal conductivity.
    
    Args:
        soil_type: Base soil type ('sand', 'clay', etc.)
        moisture_content: Moisture level ('dry', 'moist', 'saturated')
        
    Returns:
        Thermal conductivity in W/mK
    """
    # Build key
    if moisture_content:
        key = f"{moisture_content}_{soil_type}"
    else:
        key = soil_type
        
    if key in SOIL_CONDUCTIVITY:
        return SOIL_CONDUCTIVITY[key]
    
    # Try alternate forms
    if soil_type in SOIL_CONDUCTIVITY:
        return SOIL_CONDUCTIVITY[soil_type]
        
    # Default
    logger.warning(f"Unknown soil type '{key}', using average_moist")
    return SOIL_CONDUCTIVITY['average_moist']

def get_pipe_diameter(nominal_size_inches: int) -> float:
    """Convert nominal pipe size to actual outer diameter in meters.
    
    Args:
        nominal_size_inches: Nominal pipe size in inches
        
    Returns:
        Outer diameter in meters
    """
    key = f"{nominal_size_inches}_inch"
    
    if key in PIPE_SIZES:
        return PIPE_SIZES[key]
    
    # Approximate if not in table
    logger.warning(f"Pipe size {nominal_size_inches}\" not in table, approximating")
    return nominal_size_inches * 0.0254  # Simple inch to meter conversion

# Utility function to get all typical values for a category
def get_typical_values(category: str) -> Dict:
    """Get all typical values for a given category.
    
    Args:
        category: One of 'temperatures', 'fouling_factors', 'heat_transfer',
                  'soil', 'sludge', 'pipes', 'insulation', 'heat_loss', 'heat_duty'
                  
    Returns:
        Dictionary of typical values
    """
    categories = {
        'temperatures': TEMPERATURES,
        'fouling_factors': FOULING_FACTORS,
        'heat_transfer': HEAT_TRANSFER_COEFFICIENTS,
        'soil': SOIL_CONDUCTIVITY,
        'sludge': SLUDGE_PROPERTIES,
        'pipes': PIPE_SIZES,
        'insulation': INSULATION_THICKNESS,
        'heat_loss': TYPICAL_HEAT_LOSSES,
        'heat_duty': TYPICAL_HEAT_DUTIES,
    }
    
    if category in categories:
        return categories[category]
    else:
        available = ', '.join(categories.keys())
        raise ValueError(f"Unknown category '{category}'. Available: {available}")

if __name__ == "__main__":
    # Test the module
    print("Wastewater Engineering Data Module")
    print("="*50)
    
    # Test sludge properties
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.fluid_properties import get_fluid_properties
    import json
    
    water = json.loads(get_fluid_properties("water", 308.15, 101325))
    sludge = get_sludge_properties(water, 'thickened_6_percent')
    
    print("\nSludge Properties (6% thickened vs water):")
    print(f"  Density: {sludge['density']:.1f} vs {water['density']:.1f} kg/m³")
    print(f"  Viscosity: {sludge['dynamic_viscosity']:.4f} vs {water['dynamic_viscosity']:.4f} Pa·s")
    
    # Test fouling factors
    print("\nFouling Factors:")
    for app in ['clean_water', 'raw_wastewater', 'primary_sludge']:
        ff = get_fouling_factor(app, conservative=True)
        print(f"  {app}: {ff:.4f} m²K/W (with 50% safety)")
    
    # Test heat transfer coefficients
    print("\nTypical Heat Transfer Coefficients:")
    for app in ['water_in_tubes', 'sludge_in_tubes', 'tank_to_air_forced']:
        h = get_heat_transfer_coefficient(app)
        h_range = get_heat_transfer_coefficient(app, use_average=False)
        print(f"  {app}: {h:.0f} W/m²K (range: {h_range[0]}-{h_range[1]})")