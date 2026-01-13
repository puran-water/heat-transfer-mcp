"""
Heat exchanger physical dimensions estimation tool.

This module provides functionality to estimate the number of tubes and bundle
diameter for a shell-and-tube heat exchanger based on required area and tube geometry.
"""

import json
import logging
import math
from typing import Dict, Optional, Any

from utils.import_helpers import HT_AVAILABLE

logger = logging.getLogger("heat-transfer-mcp.estimate_hx_dims")


def estimate_hx_physical_dims(
    required_area: float,
    tube_outer_diameter: float,
    tube_length_effective: float,
    tube_pitch: float,
    tube_layout_angle: int,
    number_of_tube_passes: int = 1,
    method: str = "HEDH",
) -> str:
    """Estimates tube count and bundle diameter for a heat exchanger.

    Args:
        required_area: Total heat transfer area required (A) in m²
        tube_outer_diameter: Outer diameter of one tube (do) in meters
        tube_length_effective: Effective length of one tube available for heat transfer (L) in meters
        tube_pitch: Center-to-center distance between tubes (Pt) in meters
        tube_layout_angle: Tube layout angle (30, 45, 60, 90 degrees)
        number_of_tube_passes: Number of tube passes (Ntp). Default 1
        method: Method for estimation ('HEDH', 'VDI')

    Returns:
        JSON string with heat exchanger physical dimensions
    """
    try:
        # Parameter validation
        if required_area <= 0 or tube_outer_diameter <= 0 or tube_length_effective <= 0:
            return json.dumps({"error": "Required area, tube outer diameter, and tube length must be positive."})

        N_tubes = 0
        D_bundle = 0.0
        method_used = ""

        # 1. Calculate number of tubes based on required area
        area_per_tube = math.pi * tube_outer_diameter * tube_length_effective

        if area_per_tube <= 0:
            return json.dumps({"error": "Calculated area per tube is zero or negative."})

        N_tubes_calc = math.ceil(required_area / area_per_tube)
        N_tubes = N_tubes_calc

        # 2. Calculate bundle diameter based on method
        if method.upper() == "HEDH":
            method_used = "HEDH Manual Calculation"

            # D_bundle = (Do + (pitch)*sqrt(1/0.78)*sqrt(C1*N))
            if tube_layout_angle in (30, 60):
                C1 = 13.0 / 15.0
            elif tube_layout_angle in (45, 90):
                C1 = 1.0
            else:
                return json.dumps({"error": "Invalid tube_layout_angle. Use 30, 45, 60, or 90."})

            D_bundle = tube_outer_diameter + (1.0 / 0.78) ** 0.5 * tube_pitch * (C1 * N_tubes) ** 0.5

        elif method.upper() == "VDI":
            method_used = "VDI Manual Calculation"

            # Determine f1 and f2 based on passes and layout
            if number_of_tube_passes == 1:
                f2 = 0.0
            elif number_of_tube_passes == 2:
                f2 = 22.0
            elif number_of_tube_passes == 4:
                f2 = 70.0
            elif number_of_tube_passes == 6:
                f2 = 90.0
            elif number_of_tube_passes == 8:
                f2 = 105.0
            else:
                return json.dumps({"error": "Unsupported number_of_tube_passes for VDI method. Use 1, 2, 4, 6, or 8."})

            if tube_layout_angle in (30, 60):
                f1 = 1.1
            elif tube_layout_angle in (45, 90):
                f1 = 1.3
            else:
                return json.dumps({"error": "Invalid tube_layout_angle. Use 30, 45, 60, or 90."})

            # VDI formula usually works in mm; we'll convert and then convert back
            Do_mm = tube_outer_diameter * 1000.0
            pitch_mm = tube_pitch * 1000.0

            D_bundle_mm = (f1 * N_tubes * pitch_mm**2 + f2 * N_tubes**0.5 * pitch_mm + Do_mm) ** 0.5
            D_bundle = D_bundle_mm / 1000.0

        else:
            return json.dumps({"error": "Invalid method. Choose 'HEDH' or 'VDI'."})

        # Create result
        total_tube_surface_area = N_tubes * area_per_tube
        shell_side_flow_area = (math.pi / 4.0) * (D_bundle**2 - N_tubes * tube_outer_diameter**2)

        result = {
            "estimated_number_of_tubes": N_tubes,
            "estimated_bundle_diameter_m": D_bundle,
            "calculation_method": method_used,
            "tube_layout_angle_deg": tube_layout_angle,
            "tube_pitch_to_diameter_ratio": tube_pitch / tube_outer_diameter,
            "tube_geometry": {
                "outer_diameter_m": tube_outer_diameter,
                "effective_length_m": tube_length_effective,
                "pitch_m": tube_pitch,
            },
            "total_calculated_area_m2": total_tube_surface_area,
            "number_of_tube_passes": number_of_tube_passes,
            "estimated_shell_side_flow_area_m2": shell_side_flow_area,
        }

        # Warnings for unconventional designs
        bundle_to_tube_ratio = D_bundle / tube_outer_diameter
        if bundle_to_tube_ratio > 100:
            result["warnings"] = ["Very large bundle-to-tube diameter ratio. Consider multiple shells or larger tubes."]

        if N_tubes < 7:
            if "warnings" not in result:
                result["warnings"] = []
            result["warnings"].append("Very small number of tubes. Consider longer tubes or different exchanger type.")

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in estimate_hx_physical_dims: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
