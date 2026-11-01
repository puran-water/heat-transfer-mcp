"""
Import helpers for optional dependencies.

This module provides functions to gracefully handle optional dependencies
and provide fallback values when packages are not available.
"""

import logging
from typing import Any, Dict, Optional

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

# CoolProp: the declared reference-EOS property branch (pyproject.toml). Absent, every lookup not naming thermo refuses
# (tools.fluid_properties.STRICT_PROPERTY_RULE); thermo answers only when named, from its own methods
COOLPROP_AVAILABLE = False
try:
    import CoolProp  # noqa: F401

    COOLPROP_AVAILABLE = True
    logger.info("CoolProp library successfully imported (high-accuracy fluid properties available)")
except ImportError:
    logger.warning("CoolProp not available: property lookups refuse unless property_source='thermo' is named.")

# Meteostat availability check.
#
# Meteostat 2 removed the `Daily`/`Hourly` classes in favour of `daily()`/`hourly()`
# functions, so probing for the 1.x names alone reports the package as missing whenever a
# 2.x release is installed. Every tool gated on this flag then refuses to run even though
# the data is perfectly reachable through the v2 API. Detect the package itself and record
# which generation is present; `heat_transfer_mcp.weather.provider` handles both.
METEOSTAT_AVAILABLE = False
METEOSTAT_API_GENERATION = None
try:
    import meteostat

    if hasattr(meteostat, "Daily") and hasattr(meteostat, "Hourly"):
        METEOSTAT_API_GENERATION = 1
    elif hasattr(meteostat, "daily") and hasattr(meteostat, "hourly"):
        METEOSTAT_API_GENERATION = 2
    METEOSTAT_AVAILABLE = METEOSTAT_API_GENERATION is not None
    if METEOSTAT_AVAILABLE:
        logger.info(
            "Meteostat package successfully imported (v%s API)", METEOSTAT_API_GENERATION
        )
    else:
        logger.warning(
            "Meteostat is installed but exposes neither the v1 nor the v2 fetch API; "
            "weather data retrieval disabled."
        )
except ImportError:
    logger.warning("Meteostat module not available. Weather data retrieval will be disabled.")

# Pandas for data processing
PANDAS_AVAILABLE = False
try:
    import pandas as pd  # noqa: F401  # imported to probe availability, matching the flags above

    PANDAS_AVAILABLE = True
    logger.info("Pandas package successfully imported")
except ImportError:
    logger.warning("Pandas module not available. Data processing will be limited.")


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
        "plastic": 0.2,
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


# ============== Tank Geometry Helpers (using fluids.geometry.TANK) ============== #


def get_tank_geometry(geometry: str, dimensions: Dict[str, float], head_type: str = "ellipsoidal") -> Optional[Dict[str, Any]]:
    """
    Get tank geometry calculations using fluids.geometry.TANK when available.

    Args:
        geometry: Tank geometry type ('vertical_cylinder_tank', 'horizontal_cylinder_tank', 'sphere')
        dimensions: Dict with 'diameter' and 'height' (vertical) or 'length' (horizontal)
        head_type: Head type for cylinder tanks ('ellipsoidal', 'torispherical', 'flat', 'conical', 'spherical')

    Returns:
        Dict with V_total, SA_tank, SA_lateral, SA_ends, or None if calculation fails
    """
    if not FLUIDS_AVAILABLE:
        logger.debug("fluids library not available, using fallback geometry calculations")
        return None

    try:
        from fluids.geometry import TANK

        geometry_lower = geometry.lower()
        diameter = dimensions.get("diameter", 0)

        if diameter <= 0:
            return None

        # Determine orientation and length
        if "horizontal" in geometry_lower:
            horizontal = True
            length = dimensions.get("length", dimensions.get("height", 0))
        else:
            horizontal = False
            length = dimensions.get("height", dimensions.get("length", 0))

        if length <= 0:
            return None

        # Map head types
        head_map = {
            "ellipsoidal": "ellipsoidal",
            "torispherical": "torispherical",
            "flat": None,  # None means flat end
            "conical": "conical",
            "spherical": "spherical",
            "hemispherical": "spherical",
        }
        side_head = head_map.get(head_type.lower(), "ellipsoidal")

        # Create TANK instance
        if "sphere" in geometry_lower:
            # For sphere, use equal D and L with spherical heads
            tank = TANK(D=diameter, L=0, horizontal=False, sideA="spherical", sideB="spherical")
        else:
            tank = TANK(D=diameter, L=length, horizontal=horizontal, sideA=side_head, sideB=side_head)

        return {
            "V_total_m3": tank.V_total,
            "SA_tank_m2": tank.SA_tank if hasattr(tank, "SA_tank") else tank.A,
            "SA_lateral_m2": tank.A_lateral if hasattr(tank, "A_lateral") else None,
            "SA_sideA_m2": tank.A_sideA if hasattr(tank, "A_sideA") else None,
            "SA_sideB_m2": tank.A_sideB if hasattr(tank, "A_sideB") else None,
            "h_max_m": (
                tank.h_max
                if hasattr(tank, "h_max")
                else (diameter if "sphere" in geometry_lower else (diameter if horizontal else length))
            ),
            "source": "fluids.geometry.TANK",
        }
    except Exception as e:
        logger.warning(f"fluids.geometry.TANK calculation failed: {e}")
        return None


def get_tank_partial_fill(
    geometry: str, dimensions: Dict[str, float], fill_height: float, head_type: str = "ellipsoidal"
) -> Optional[Dict[str, Any]]:
    """
    Get partial fill volume and wetted area using fluids.geometry.TANK.

    Args:
        geometry: Tank geometry type
        dimensions: Tank dimensions
        fill_height: Height of liquid from bottom (m)
        head_type: Type of tank heads

    Returns:
        Dict with V_liquid, SA_wetted, or None if calculation fails
    """
    if not FLUIDS_AVAILABLE:
        return None

    try:
        from fluids.geometry import TANK

        geometry_lower = geometry.lower()
        diameter = dimensions.get("diameter", 0)

        if "horizontal" in geometry_lower:
            horizontal = True
            length = dimensions.get("length", dimensions.get("height", 0))
        else:
            horizontal = False
            length = dimensions.get("height", dimensions.get("length", 0))

        head_map = {
            "ellipsoidal": "ellipsoidal",
            "torispherical": "torispherical",
            "flat": None,
            "conical": "conical",
            "spherical": "spherical",
        }
        side_head = head_map.get(head_type.lower(), "ellipsoidal")

        tank = TANK(D=diameter, L=length, horizontal=horizontal, sideA=side_head, sideB=side_head)

        # Clamp fill height to valid range
        h_clamped = max(0, min(fill_height, tank.h_max))

        return {
            "V_liquid_m3": tank.V_from_h(h_clamped),
            "SA_wetted_m2": tank.SA_from_h(h_clamped) if hasattr(tank, "SA_from_h") else None,
            "fill_fraction": h_clamped / tank.h_max if tank.h_max > 0 else 0,
            "source": "fluids.geometry.TANK",
        }
    except Exception as e:
        logger.warning(f"fluids.geometry.TANK partial fill calculation failed: {e}")
        return None


# ============== Material Properties (using ht.insulation) ============== #


def get_material_k_from_ht(material_name: str, temperature_k: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Get material thermal conductivity using ht.insulation database (390+ materials).

    Args:
        material_name: Material name (fuzzy matching supported)
        temperature_k: Temperature in Kelvin (for temperature-dependent refractories)

    Returns:
        Dict with k, material_id, source, or None if not found
    """
    if not HT_AVAILABLE:
        return None

    try:
        from ht.insulation import k_material, nearest_material

        # Try direct lookup first
        try:
            if temperature_k and temperature_k > 673.15:  # > 400°C, refractory range
                k = k_material(material_name, T=temperature_k)
            else:
                k = k_material(material_name)

            return {
                "thermal_conductivity_w_mk": k,
                "material_id": material_name,
                "temperature_k": temperature_k,
                "source": "ht.insulation.k_material",
            }
        except (KeyError, ValueError):
            pass

        # Try fuzzy matching
        try:
            matched_name = nearest_material(material_name)
            if matched_name:
                if temperature_k and temperature_k > 673.15:
                    k = k_material(matched_name, T=temperature_k)
                else:
                    k = k_material(matched_name)

                return {
                    "thermal_conductivity_w_mk": k,
                    "material_id": matched_name,
                    "material_requested": material_name,
                    "temperature_k": temperature_k,
                    "fuzzy_match": True,
                    "source": "ht.insulation.k_material",
                }
        except Exception:
            pass

        return None
    except Exception as e:
        logger.warning(f"ht.insulation.k_material lookup failed: {e}")
        return None


def get_material_properties_from_ht(material_name: str, temperature_k: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Get full material properties (k, rho, Cp) using ht.insulation when available.

    Args:
        material_name: Material name
        temperature_k: Temperature in Kelvin

    Returns:
        Dict with thermal_conductivity, density, specific_heat, or None
    """
    if not HT_AVAILABLE:
        return None

    try:
        from ht.insulation import Cp_material, k_material, nearest_material, rho_material

        # Try fuzzy match for material name
        try:
            matched_name = nearest_material(material_name)
        except Exception:
            matched_name = material_name

        result = {
            "material_id": matched_name or material_name,
            "material_requested": material_name,
            "source": "ht.insulation",
        }

        # Get thermal conductivity
        try:
            if temperature_k and temperature_k > 673.15:
                result["thermal_conductivity_w_mk"] = k_material(matched_name, T=temperature_k)
            else:
                result["thermal_conductivity_w_mk"] = k_material(matched_name)
        except Exception:
            pass

        # Get density
        try:
            result["density_kg_m3"] = rho_material(matched_name)
        except Exception:
            pass

        # Get specific heat
        try:
            result["specific_heat_j_kgk"] = Cp_material(matched_name)
        except Exception:
            pass

        if "thermal_conductivity_w_mk" in result or "density_kg_m3" in result:
            return result
        return None

    except Exception as e:
        logger.warning(f"ht.insulation material lookup failed: {e}")
        return None


# ============== CoolProp Integration ============== #


def coolprop_fluid_properties(fluid_name: str, temperature_k: float, pressure_pa: float = 101325.0) -> Dict[str, Any]:
    """CoolProp properties at (T, P); raises, naming why, when CoolProp gives none (the named-branch path of
    tools.fluid_properties.get_fluid_properties)."""
    from CoolProp.CoolProp import PropsSI

    # Map common names to CoolProp names
    coolprop_name_map = {
        "water": "Water",
        "air": "Air",
        "nitrogen": "Nitrogen",
        "oxygen": "Oxygen",
        "carbon dioxide": "CarbonDioxide",
        "co2": "CarbonDioxide",
        "methane": "Methane",
        "ethanol": "Ethanol",
        "ammonia": "Ammonia",
        "hydrogen": "Hydrogen",
        "helium": "Helium",
        "argon": "Argon",
        "propane": "Propane",
        "butane": "Butane",
        "r134a": "R134a",
        "r410a": "R410A",
    }

    cp_name = coolprop_name_map.get(fluid_name.lower(), fluid_name)

    if temperature_k < PropsSI("Ttriple", cp_name):
        raise ValueError("Below triple temperature: solid/metastable states are outside this fluid tool")
    if temperature_k > PropsSI("Tmax", cp_name) or pressure_pa > PropsSI("pmax", cp_name):
        raise ValueError("Above the equation of state's Tmax/pmax: CoolProp would extrapolate outside its range")
    density = PropsSI("D", "T", temperature_k, "P", pressure_pa, cp_name)
    cp = PropsSI("C", "T", temperature_k, "P", pressure_pa, cp_name)
    k = PropsSI("L", "T", temperature_k, "P", pressure_pa, cp_name)
    mu = PropsSI("V", "T", temperature_k, "P", pressure_pa, cp_name)
    phase = PropsSI("Phase", "T", temperature_k, "P", pressure_pa, cp_name)

    # Map phase number to string
    phase_map = {
        0: "liquid",
        1: "supercritical",
        2: "supercritical_gas",
        3: "supercritical_liquid",
        5: "gas",
        6: "two_phase",
    }
    phase_str = phase_map.get(int(phase), "unknown")
    if cp_name == "Air" and phase_str not in ("gas", "supercritical_gas", "supercritical"):
        # the thermo branch's DryAirLemmon guard: CoolProp's Air is pseudo-pure, blind to the mixture's VLE glide
        raise ValueError("Cryogenic liquid or two-phase air requires a phase-qualified mixture model")

    return {
        "fluid_name": fluid_name,
        "coolprop_name": cp_name,
        "temperature_k": temperature_k,
        "pressure_pa": pressure_pa,
        "density": density,
        "specific_heat_cp": cp,
        "thermal_conductivity": k,
        "dynamic_viscosity": mu,
        "kinematic_viscosity": mu / density if density > 0 else None,
        "prandtl_number": (mu * cp) / k if k > 0 else None,
        "phase": phase_str,
        "isobaric_expansion_per_K": PropsSI("ISOBARIC_EXPANSION_COEFFICIENT", "T", temperature_k, "P", pressure_pa, cp_name),
        "source": "CoolProp",
        "accuracy": "high (reference EOS)",
    }


def get_fluid_properties_coolprop(
    fluid_name: str, temperature_k: float, pressure_pa: float = 101325.0
) -> Optional[Dict[str, Any]]:
    """
    Get fluid properties using CoolProp for high-accuracy calculations.

    Args:
        fluid_name: CoolProp fluid name (e.g., 'Water', 'Air', 'Nitrogen')
        temperature_k: Temperature in Kelvin
        pressure_pa: Pressure in Pascals

    Returns:
        Dict with fluid properties or None if CoolProp unavailable/fails
    """
    if not COOLPROP_AVAILABLE:
        return None
    try:
        return coolprop_fluid_properties(fluid_name, temperature_k, pressure_pa)
    except Exception as prop_error:
        logger.debug(f"CoolProp property calculation failed for {fluid_name}: {prop_error}")
        return None
