"""
Ground heat loss calculation tool.

This module provides functionality to estimate steady-state heat loss from structures
in contact with the ground (e.g., slab-on-grade, basement walls/floor) using shape
factors or simplified methods.
"""

import json
import logging
import math
from typing import Dict, Optional, Any

from utils.import_helpers import HT_AVAILABLE, get_material_thermal_conductivity_fallback
from utils.helpers import calculate_radiation_heat_transfer

logger = logging.getLogger("heat-transfer-mcp.ground_heat_loss")


def calculate_ground_heat_loss(
    structure_type: str,
    dimensions: Dict[str, float],
    depth: float = 0.0,
    insulation_R_value_si: float = 0.0,
    wall_thickness: float = 0.2,
    wall_conductivity: float = 1.7,
    soil_conductivity: float = None,
    internal_temperature: float = None,
    average_external_air_temperature: float = None,
    internal_convection_coefficient_h: float = 8.0,
) -> str:
    """Estimates ground heat loss using simplified methods/shape factors.

    Args:
        structure_type: Type of structure ('slab_on_grade', 'basement_wall', 'basement_floor')
        dimensions: Dictionary of dimensions in meters (e.g., slab: {'length': L, 'width': W},
                   wall: {'height': H, 'length': L}, floor: {'length': L, 'width': W})
        depth: Depth below grade in meters (e.g., for basement wall/floor average depth).
              Default 0 for slab.
        insulation_R_value_si: R-value of any insulation applied (m²K/W)
        wall_thickness: Thickness of the concrete wall/slab in meters
        wall_conductivity: Thermal conductivity of the wall/slab material (W/mK, default for concrete)
        soil_conductivity: Thermal conductivity of the surrounding soil (W/mK)
        internal_temperature: Internal air/fluid temperature in Kelvin (K)
        average_external_air_temperature: Average annual external air temperature in Kelvin (K),
                                      used to estimate deep ground temperature
        internal_convection_coefficient_h: Internal surface convection coefficient (W/m²K)

    Returns:
        JSON string with heat loss results
    """
    try:
        # Validate required inputs
        if soil_conductivity is None or internal_temperature is None or average_external_air_temperature is None:
            return json.dumps(
                {"error": "soil_conductivity, internal_temperature, and average_external_air_temperature are required."}
            )

        structure_type_lower = structure_type.lower()
        length = dimensions.get("length")
        width = dimensions.get("width")
        height = dimensions.get("height")  # For walls

        # Estimate deep ground temperature = average annual air temperature
        ground_temperature = average_external_air_temperature

        # Calculate wall/slab resistance
        R_wall = wall_thickness / wall_conductivity if wall_conductivity > 0 else float("inf")
        R_internal_conv = 1.0 / internal_convection_coefficient_h if internal_convection_coefficient_h > 0 else float("inf")

        # Total resistance excluding soil path and insulation
        R_structure = R_internal_conv + R_wall

        # Add insulation resistance
        R_total_structure = R_structure + insulation_R_value_si

        Q_ground = 0.0
        method_used = ""

        if "slab_on_grade" in structure_type_lower:
            if length is None or width is None:
                return json.dumps({"error": "Slab requires 'length' and 'width' dimensions."})

            perimeter = 2 * (length + width)
            area = length * width

            # Calculate characteristic dimension B'
            B_prime = area / (0.5 * perimeter) if perimeter > 0 else 0

            # Calculate effective soil path and resistance
            effective_path = B_prime / 2.0  # Simplified approximation
            R_soil_path = effective_path / soil_conductivity if soil_conductivity > 0 else float("inf")
            R_total_slab = R_total_structure + R_soil_path
            U_slab_equiv = 1.0 / R_total_slab if R_total_slab > 0 else 0

            # Calculate heat loss
            Q_ground = U_slab_equiv * area * (internal_temperature - ground_temperature)
            method_used = f"Slab-on-Grade (Equivalent U-value based on B'={B_prime:.2f}m, R_total={R_total_slab:.3f})"

        elif "basement_wall" in structure_type_lower:
            if height is None or length is None:
                return json.dumps({"error": "Basement wall requires 'height' and 'length' dimensions."})

            area = height * length

            # Effective soil path increases with depth
            effective_path = depth / 2.0 if depth > 0 else 0.5  # Assume 0.5m path if depth=0
            R_soil_path = effective_path / soil_conductivity if soil_conductivity > 0 else float("inf")
            R_total_wall = R_total_structure + R_soil_path
            U_wall_equiv = 1.0 / R_total_wall if R_total_wall > 0 else 0

            # Calculate heat loss
            Q_ground = U_wall_equiv * area * (internal_temperature - ground_temperature)
            method_used = f"Basement Wall (Equivalent U-value based on avg depth={depth:.2f}m, R_total={R_total_wall:.3f})"

        elif "basement_floor" in structure_type_lower:
            if length is None or width is None:
                return json.dumps({"error": "Basement floor requires 'length' and 'width' dimensions."})

            area = length * width

            # Path length approximately equals depth
            effective_path = depth
            R_soil_path = effective_path / soil_conductivity if soil_conductivity > 0 else float("inf")
            R_total_floor = R_total_structure + R_soil_path
            U_floor_equiv = 1.0 / R_total_floor if R_total_floor > 0 else 0

            # Calculate heat loss
            Q_ground = U_floor_equiv * area * (internal_temperature - ground_temperature)
            method_used = f"Basement Floor (Equivalent U-value based on depth={depth:.2f}m, R_total={R_total_floor:.3f})"

        else:
            return json.dumps({"error": f"Unsupported structure_type: {structure_type}"})

        # Create result
        result = {
            "ground_heat_loss_watts": Q_ground,
            "method_used": method_used,
            "assumed_ground_temp_k": ground_temperature,
            "structure_type": structure_type,
            "soil_thermal_conductivity_w_mk": soil_conductivity,
            "internal_temperature_k": internal_temperature,
            "internal_temperature_c": internal_temperature - 273.15,
            "ground_temperature_k": ground_temperature,
            "ground_temperature_c": ground_temperature - 273.15,
            "structure_dimensions_m": dimensions,
            "calculated_resistances_m2k_w": {
                "wall_material": R_wall,
                "internal_convection": R_internal_conv,
                "insulation": insulation_R_value_si,
                "soil_path": R_soil_path if "R_soil_path" in locals() else None,
                "total": R_total_structure + (R_soil_path if "R_soil_path" in locals() else 0),
            },
        }

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in calculate_ground_heat_loss: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
