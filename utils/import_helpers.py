"""
Import helpers for optional dependencies.

This module provides functions to gracefully handle optional dependencies
and provide fallback values when packages are not available.
"""

import logging
import math
import json
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger("heat-transfer-mcp.imports")

"""Optional dependency detection and conservative fallbacks.

This module centralizes availability checks for optional scientific
dependencies and provides basic fallbacks where strictly necessary.

Notes:
- Avoid gating one library's usage behind another (e.g., don't gate
  thermo usage on ht being available). Each flag reflects the actual
  library presence so tools can choose the best available modeling.
"""

# Library availability flags (kept lightweight to avoid import side‑effects)
HT_AVAILABLE = False
THERMO_AVAILABLE = False
FLUIDS_AVAILABLE = False
CHEMICALS_AVAILABLE = False

# Heat transfer correlations library
try:
    import ht  # noqa: F401
    HT_AVAILABLE = True
    logger.info("Heat transfer (ht) library successfully imported")
except ImportError:
    logger.warning("Heat transfer (ht) library not available. Some correlations will fall back.")

# Thermophysical properties library
try:
    import thermo  # noqa: F401
    THERMO_AVAILABLE = True
    logger.info("Thermo library successfully imported")
except ImportError:
    logger.warning("Thermo library not available. Property calculations will be limited.")

# Fluid mechanics helpers library
try:
    import fluids  # noqa: F401
    FLUIDS_AVAILABLE = True
    logger.info("Fluids library successfully imported")
except ImportError:
    logger.warning("Fluids library not available. Some flow utilities will be limited.")

# Chemicals property backend used by thermo
try:
    import chemicals  # noqa: F401
    CHEMICALS_AVAILABLE = True
    logger.info("Chemicals library successfully imported")
except ImportError:
    logger.warning("Chemicals library not available. Some property methods may be missing.")

# Meteostat availability check
METEOSTAT_AVAILABLE = False
try:
    import meteostat
    from meteostat import Point, Daily, Hourly
    METEOSTAT_AVAILABLE = True
    logger.info("Meteostat package successfully imported")
except ImportError:
    logger.warning("Meteostat module not available. Weather data retrieval will be disabled.")

# Pandas for data processing
PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    logger.info("Pandas package successfully imported")
except ImportError:
    logger.warning("Pandas module not available. Data processing will be limited.")

def get_fluid_properties_fallback(
    fluid_name: str, 
    temperature_k: float,
    pressure_pa: float = 101325.0
) -> Optional[Dict[str, float]]:
    """
    Fallback function for basic fluid properties when HT library is not available.
    
    Args:
        fluid_name: Name of the fluid
        temperature_k: Temperature in Kelvin
        pressure_pa: Pressure in Pascals
        
    Returns:
        Dictionary of basic fluid properties or None if not recognized
    """
    # Very basic properties for common fluids at roughly atmospheric conditions
    # These are approximate values and shouldn't be used for precise calculations
    fluid_name_lower = fluid_name.lower()
    
    if "water" in fluid_name_lower:
        # Rough approximation for water near standard conditions
        return {
            "name": "Water",
            "density": 1000.0,  # kg/m³
            "specific_heat_cp": 4180.0,  # J/(kg·K)
            "thermal_conductivity": 0.6,  # W/(m·K)
            "dynamic_viscosity": 0.001,  # Pa·s
            "kinematic_viscosity": 1.0e-6,  # m²/s
            "prandtl_number": 7.0
        }
    elif "air" in fluid_name_lower:
        # Simple but reasonable air estimate if thermo is not available.
        # Use ideal-gas density and Sutherland-like temperature scalings.
        R_air = 287.0  # J/(kg·K)
        density = pressure_pa / (R_air * temperature_k)
        T_ref = 293.15
        mu_ref = 1.81e-5
        k_ref = 0.026
        mu = mu_ref * (temperature_k / T_ref) ** 1.5 * (T_ref + 110.4) / (temperature_k + 110.4)
        k = k_ref * (temperature_k / T_ref) ** 0.8
        return {
            "name": "Air",
            "density": density,
            "specific_heat_cp": 1007.0,
            "thermal_conductivity": k,
            "dynamic_viscosity": mu,
            "kinematic_viscosity": mu / density,
            "prandtl_number": 0.71,
        }
    else:
        logger.warning(f"Fluid '{fluid_name}' not recognized in fallback properties.")
        return None

def get_material_thermal_conductivity_fallback(material_name: str) -> Optional[float]:
    """
    Fallback function to provide thermal conductivity values for common materials
    when the HT library is not available.
    
    Args:
        material_name: Name of the material
        
    Returns:
        Thermal conductivity in W/(m·K) or None if not recognized
    """
    material_dict = {
        # Metals
        "aluminum": 237.0,
        "copper": 398.0,
        "iron": 80.0,
        "steel": 45.0,
        "stainless steel": 15.0,
        "brass": 109.0,
        "bronze": 110.0,
        "gold": 314.0,
        "silver": 429.0,
        "platinum": 70.0,
        "lead": 35.0,
        "zinc": 116.0,
        
        # Building materials
        "concrete": 1.7,
        "brick": 0.8,
        "glass": 1.05,
        "wood": 0.12,
        "plywood": 0.13,
        "drywall": 0.17,
        "asphalt": 0.7,
        "ceramic": 1.5,
        "marble": 2.8,
        "granite": 2.9,
        
        # Insulation
        "fiberglass": 0.04,
        "mineral wool": 0.044,
        "cellulose": 0.039,
        "foam": 0.035,
        "polystyrene": 0.033,
        "polyurethane": 0.025,
        "cork": 0.043,
        "air (still)": 0.026,
        "glass wool": 0.038,
        
        # Fluids (thermal conductivity values for reference)
        "water": 0.6,
        "air": 0.026,
        "oil": 0.15,
        "glycol": 0.258,
        
        # Other
        "soil": 0.8,
        "ice": 2.18,
        "rubber": 0.16,
        "ptfe": 0.25,
        "pvc": 0.19,
        "plastic": 0.2
    }
    
    material_name_lower = material_name.lower()
    
    if material_name_lower in material_dict:
        return material_dict[material_name_lower]
    else:
        # Attempt partial match
        for key in material_dict:
            if key in material_name_lower or material_name_lower in key:
                logger.warning(f"Using partial match: '{key}' for material '{material_name}'")
                return material_dict[key]
    
    logger.warning(f"Material '{material_name}' not recognized in fallback properties.")
    return None
