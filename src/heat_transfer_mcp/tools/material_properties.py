"""
Material properties tool to retrieve thermal properties of common materials.

This module provides functionality to obtain thermal conductivity values using the HT library's
insulation module which contains 390+ materials from VDI and ASHRAE databases.
"""

import json
import logging
import math
from difflib import SequenceMatcher
from typing import Optional

from heat_transfer_mcp.materials.resolver import UnknownMaterialError, resolve_material
from heat_transfer_mcp.utils.import_helpers import HT_AVAILABLE, get_material_thermal_conductivity_fallback

logger = logging.getLogger("heat-transfer-mcp.material_properties")

_HT_FUZZY_MIN_SCORE = 0.72


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
        resolver_error = None

        try:
            entry = resolve_material(name_clean, temperature_C=(T - 273.15) if temperature is not None else None)
            result = {
                "material_name": material_name,
                "matched_name": entry.name,
                "canonical_id": entry.canonical_id,
                "aliases": entry.aliases,
                "thermal_conductivity_k": entry.k_W_mK,
                "density_rho": entry.rho_kg_m3,
                "density_unit": "kg/m³",
                "specific_heat_cp": entry.cp_J_kgK,
                "specific_heat_unit": "J/(kg·K)",
                "emissivity": entry.emissivity,
                "source": entry.source,
                "applicable_temp_range_C": entry.applicable_temp_range_C,
            }
            if temperature is not None:
                result["temperature_k"] = T
                result["temperature_c"] = T - 273.15
            if entry.warnings:
                result["warnings"] = entry.warnings
            result["unit"] = "W/(m·K)"
            return json.dumps(result)
        except UnknownMaterialError as e:
            resolver_error = e
            if _should_skip_ht_fuzzy(name_clean, resolver_error):
                return json.dumps(_unknown_material_payload(material_name, resolver_error))

        if HT_AVAILABLE:
            try:
                # Import the insulation module which has material properties
                from ht.insulation import Cp_material, k_material, nearest_material, rho_material

                logger.info(f"Attempting to get properties for material '{material_name}' using HT insulation module")

                # Use fuzzy matching to find the best material match
                matched_material = nearest_material(name_clean)
                match_score = _material_match_score(name_clean, matched_material)
                if not _is_acceptable_ht_fuzzy_match(name_clean, matched_material, match_score):
                    payload = _unknown_material_payload(material_name, resolver_error)
                    payload["details"] = {
                        "match_score": match_score,
                        "minimum_match_score": _HT_FUZZY_MIN_SCORE,
                    }
                    return json.dumps(payload)
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

                logger.info("Successfully retrieved properties from HT insulation module")

            except ImportError as e:
                logger.warning(f"HT insulation module not available: {e}")
                # Fall back to hardcoded values
                k_value = get_material_thermal_conductivity_fallback(name_clean)
                if k_value is not None:
                    result = {"material_name": material_name, "thermal_conductivity_k": k_value, "source": "fallback database"}
                else:
                    return json.dumps(_unknown_material_payload(material_name, resolver_error))

            except Exception as e:
                logger.error(f"Error using HT insulation module: {e}")
                # Fall back to hardcoded values
                k_value = get_material_thermal_conductivity_fallback(name_clean)
                if k_value is not None:
                    result = {"material_name": material_name, "thermal_conductivity_k": k_value, "source": "fallback database"}
                else:
                    payload = _unknown_material_payload(material_name, resolver_error)
                    payload["details"] = str(e)
                    return json.dumps(payload)
        else:
            # HT not available, use fallback
            logger.info(f"HT library not available, using fallback for '{material_name}'")
            k_value = get_material_thermal_conductivity_fallback(name_clean)
            if k_value is not None:
                result = {"material_name": material_name, "thermal_conductivity_k": k_value, "source": "fallback database"}
            else:
                return json.dumps(_unknown_material_payload(material_name, resolver_error))

        # Add units for clarity
        if "thermal_conductivity_k" in result:
            result["unit"] = "W/(m·K)"

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in get_material_properties: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})


def _unknown_material_payload(material_name: str, resolver_error: Optional[UnknownMaterialError]) -> dict:
    message = f"Could not find thermal conductivity for material '{material_name}'."
    payload = {"code": "unknown_material", "message": message, "error": message}
    if resolver_error is not None:
        payload["alternatives"] = resolver_error.alternatives
    return payload


def _should_skip_ht_fuzzy(material_name: str, resolver_error: UnknownMaterialError) -> bool:
    """Avoid fuzzy insulation matches for alloy/catalog-style material names."""
    if not resolver_error.alternatives:
        return False
    normalized = _normalize_material_token(material_name)
    catalog_markers = (
        "astm",
        "asme",
        "sa106",
        "sa516",
        "a106",
        "a516",
        "uns",
        "ss",
        "stainless",
        "304",
        "316",
        "317",
        "2205",
        "2507",
    )
    return any(marker in normalized for marker in catalog_markers)


def _is_acceptable_ht_fuzzy_match(material_name: str, matched_material: str, score: float) -> bool:
    query = _normalize_material_token(material_name)
    matched = _normalize_material_token(matched_material)
    if not query or not matched:
        return False
    if query == matched:
        return True
    if len(query) >= 6 and query in matched:
        return True
    if len(matched) >= 6 and matched in query:
        return True
    return score >= _HT_FUZZY_MIN_SCORE


def _material_match_score(material_name: str, matched_material: str) -> float:
    return SequenceMatcher(None, _normalize_material_token(material_name), _normalize_material_token(matched_material)).ratio()


def _normalize_material_token(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())
