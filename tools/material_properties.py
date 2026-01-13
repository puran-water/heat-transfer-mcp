"""
Material properties tool to retrieve thermal properties of common materials.

This module provides functionality to obtain thermal conductivity values using the HT library's
insulation module which contains 390+ materials from VDI and ASHRAE databases.
"""

import json
import logging
import math
from typing import Optional

from utils.import_helpers import HT_AVAILABLE, get_material_thermal_conductivity_fallback

logger = logging.getLogger("heat-transfer-mcp.material_properties")


def get_material_properties(material_name: str, temperature: Optional[float] = None) -> str:
    """Retrieves thermal conductivity of common materials.

    Args:
        material_name: Name of the material (e.g., 'steel', 'concrete', 'glass wool')
        temperature: Optional temperature in Kelvin for temperature-dependent properties

    Returns:
        JSON string with material properties including thermal conductivity
    """
    try:
        # Validate material name
        if not isinstance(material_name, str) or not material_name.strip():
            return json.dumps({"error": "Material name must be a non-empty string."})
        name_clean = material_name.strip()
        if not any(ch.isalpha() for ch in name_clean):
            return json.dumps({"error": "Material name must contain alphabetic characters (e.g., 'steel', 'concrete')."})

        # Optional: validate temperature if provided
        if temperature is not None:
            try:
                T = float(temperature)
            except (TypeError, ValueError):
                return json.dumps({"error": "Temperature must be a numeric value in Kelvin when provided."})
            if T < 0.0:
                return json.dumps({"error": "Temperature cannot be below 0 K (absolute zero)."})
            if not math.isfinite(T):
                return json.dumps({"error": "Temperature must be a finite real number."})
        result = {}

        if HT_AVAILABLE:
            try:
                # Import the insulation module which has material properties
                from ht.insulation import nearest_material, k_material, rho_material, Cp_material

                logger.info(f"Attempting to get properties for material '{material_name}' using HT insulation module")

                # Use fuzzy matching to find the best material match
                matched_material = nearest_material(name_clean)
                logger.info(f"Matched '{material_name}' to '{matched_material}'")

                # Get thermal conductivity
                k_value = k_material(matched_material, T if temperature is not None else None)

                result = {
                    "material_name": material_name,
                    "matched_name": matched_material,
                    "thermal_conductivity_k": float(k_value),
                    "source": "ht.insulation module",
                }

                # Try to get density if available
                try:
                    rho_value = rho_material(matched_material)
                    if rho_value is not None:
                        result["density_rho"] = float(rho_value)
                        result["density_unit"] = "kg/m³"
                except:
                    pass  # Not all materials have density

                # Try to get heat capacity if available
                try:
                    cp_value = Cp_material(matched_material, T if temperature is not None else None)
                    if cp_value is not None:
                        result["specific_heat_cp"] = float(cp_value)
                        result["specific_heat_unit"] = "J/(kg·K)"
                except:
                    pass  # Not all materials have heat capacity

                # Add temperature if specified
                if temperature is not None:
                    result["temperature_k"] = T

                logger.info(f"Successfully retrieved properties from HT insulation module")

            except ImportError as e:
                logger.warning(f"HT insulation module not available: {e}")
                # Fall back to hardcoded values
                k_value = get_material_thermal_conductivity_fallback(name_clean)
                if k_value is not None:
                    result = {"material_name": material_name, "thermal_conductivity_k": k_value, "source": "fallback database"}
                else:
                    return json.dumps({"error": f"Could not find thermal conductivity for material '{material_name}'."})

            except Exception as e:
                logger.error(f"Error using HT insulation module: {e}")
                # Fall back to hardcoded values
                k_value = get_material_thermal_conductivity_fallback(name_clean)
                if k_value is not None:
                    result = {"material_name": material_name, "thermal_conductivity_k": k_value, "source": "fallback database"}
                else:
                    return json.dumps(
                        {"error": f"Could not find thermal conductivity for material '{material_name}'. Error: {str(e)}"}
                    )
        else:
            # HT not available, use fallback
            logger.info(f"HT library not available, using fallback for '{material_name}'")
            k_value = get_material_thermal_conductivity_fallback(name_clean)
            if k_value is not None:
                result = {"material_name": material_name, "thermal_conductivity_k": k_value, "source": "fallback database"}
            else:
                return json.dumps({"error": f"Could not find thermal conductivity for material '{material_name}'."})

        # Add units for clarity
        if "thermal_conductivity_k" in result:
            result["unit"] = "W/(m·K)"

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in get_material_properties: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
