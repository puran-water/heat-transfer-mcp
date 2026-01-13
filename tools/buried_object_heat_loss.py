"""
Buried object heat loss calculation tool.

This module provides functionality to calculate steady-state heat loss
from buried objects (pipes/cylinders, spheres) using conduction shape factors.
"""

import json
import logging
import math
from typing import Dict, Optional, Any

from utils.import_helpers import HT_AVAILABLE

logger = logging.getLogger("heat-transfer-mcp.buried_object_heat_loss")


def calculate_buried_object_heat_loss(
    object_type: str,
    diameter: float,
    length: Optional[float] = None,
    burial_depth: float = None,
    soil_conductivity: float = None,
    object_temperature: float = None,
    ground_surface_temperature: float = None,
) -> str:
    """Calculates buried object heat loss using shape factors.

    Args:
        object_type: Type of object ('pipe', 'sphere')
        diameter: Outer diameter of the pipe or sphere in meters
        length: Length of the buried pipe in meters (REQUIRED for 'pipe')
        burial_depth: Depth from ground surface to the centerline of the object in meters
        soil_conductivity: Thermal conductivity of the surrounding soil (W/mK)
        object_temperature: Temperature of the object's outer surface in Kelvin (K)
        ground_surface_temperature: Temperature of the ground surface in Kelvin (K)

    Returns:
        JSON string with heat loss results
    """
    try:
        # Validate required inputs
        if object_type is None or not isinstance(object_type, str) or not object_type.strip():
            return json.dumps({"error": "object_type must be a non-empty string ('pipe' or 'sphere')."})
        obj_type_lower = object_type.lower()
        if obj_type_lower not in {"pipe", "sphere"}:
            return json.dumps({"error": f"Unsupported object_type: {object_type}. Use 'pipe' or 'sphere'."})
        if obj_type_lower == "pipe" and length is None:
            return json.dumps({"error": "Length is required for object_type 'pipe'."})
        try:
            diameter = float(diameter)
            burial_depth = float(burial_depth) if burial_depth is not None else None
            soil_conductivity = float(soil_conductivity)
            if obj_type_lower == "pipe" and length is not None:
                length = float(length)
        except (TypeError, ValueError):
            return json.dumps(
                {"error": "diameter, burial_depth, soil_conductivity (and length for pipe) must be numeric values."}
            )
        if burial_depth is None:
            return json.dumps({"error": "burial_depth is required."})
        if diameter <= 0.0:
            return json.dumps({"error": "diameter must be positive."})
        if soil_conductivity <= 0.0:
            return json.dumps({"error": "soil_conductivity must be positive."})
        if burial_depth < 0.0:
            return json.dumps({"error": "burial_depth cannot be negative."})
        # Validate temperatures
        try:
            object_temperature = float(object_temperature)
            ground_surface_temperature = float(ground_surface_temperature)
        except (TypeError, ValueError):
            return json.dumps({"error": "object_temperature and ground_surface_temperature must be numeric values in Kelvin."})
        for T_val, label in [
            (object_temperature, "object_temperature"),
            (ground_surface_temperature, "ground_surface_temperature"),
        ]:
            if not math.isfinite(T_val):
                return json.dumps({"error": f"{label} must be a finite real number."})
            if T_val < 0.0:
                return json.dumps({"error": f"{label} cannot be below 0 K (absolute zero)."})

        if burial_depth < diameter / 2.0:
            logger.warning("Burial depth is less than object radius. Object is not fully buried. Results may be inaccurate.")

        Q_loss = 0.0
        shape_factor_S = 0.0
        deltaT = object_temperature - ground_surface_temperature
        method_used = f"Shape Factor ({object_type})"

        if obj_type_lower == "pipe":
            # Shape factor per unit length for pipe buried in semi-infinite medium
            # Formula: S_per_L = 2 * pi / arccosh(2*z/D)
            # where z = burial_depth, D = diameter
            arg_cosh = 2.0 * burial_depth / diameter

            if arg_cosh > 1.0:  # Required for arccosh
                shape_factor_per_L = (2.0 * math.pi) / math.acosh(arg_cosh)
                shape_factor_S = shape_factor_per_L
                Q_loss = shape_factor_per_L * soil_conductivity * length * deltaT
            else:
                return json.dumps(
                    {"error": "Cannot calculate shape factor: burial depth must be greater than radius (2*z/D > 1)."}
                )

        elif obj_type_lower == "sphere":
            # Shape factor for sphere buried in semi-infinite medium
            # S = 4 * pi * r / (1 - r / (2*z))
            # where r = D/2, z = burial_depth
            radius = diameter / 2.0

            if burial_depth > radius:
                shape_factor_S = (4.0 * math.pi * radius) / (1.0 - radius / (2.0 * burial_depth))
                Q_loss = shape_factor_S * soil_conductivity * deltaT
            else:
                return json.dumps({"error": "Sphere must be fully buried (burial_depth > radius)."})

        else:
            return json.dumps({"error": f"Unsupported object_type: {object_type}. Use 'pipe' or 'sphere'."})

        # Create result
        result = {
            "total_heat_loss_watts": Q_loss,
            "shape_factor_S_or_S_per_L_m": shape_factor_S,  # S for sphere, S/L for pipe
            "driving_temperature_difference_k": deltaT,
            "method_used": method_used,
            "object_type": object_type,
            "diameter_m": diameter,
            "burial_depth_m": burial_depth,
            "soil_thermal_conductivity_w_mk": soil_conductivity,
            "object_temperature_k": object_temperature,
            "object_temperature_c": object_temperature - 273.15,
            "ground_surface_temperature_k": ground_surface_temperature,
            "ground_surface_temperature_c": ground_surface_temperature - 273.15,
        }

        # Add length for pipe
        if object_type.lower() == "pipe" and length is not None:
            result["length_m"] = length
            result["heat_loss_per_length_w_m"] = Q_loss / length if length > 0 else 0

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in calculate_buried_object_heat_loss: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
