"""
Overall heat transfer coefficient tool to calculate U-values for composite structures.

This module provides functionality to calculate the overall heat transfer coefficient (U-Value)
for various geometries including flat walls and cylinders with multiple material layers.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

from utils.import_helpers import HT_AVAILABLE
from tools.material_properties import get_material_properties

logger = logging.getLogger("heat-transfer-mcp.overall_heat_transfer")


def calculate_overall_heat_transfer_coefficient(
    geometry: str,
    layers: List[Dict[str, Any]],
    inner_convection_coefficient_h: float,
    outer_convection_coefficient_h: float,
    inner_diameter: Optional[float] = None,
    outer_diameter: Optional[float] = None,
) -> str:
    """Calculates the overall heat transfer coefficient (U-Value) for composite structures.

    Args:
        geometry: Geometry type ('flat_wall' or 'cylinder')
        layers: List of material layers, each with 'thickness' and either 'thermal_conductivity_k' or 'material_name'
        inner_convection_coefficient_h: Convective coefficient on inner surface (W/m²K)
        outer_convection_coefficient_h: Convective coefficient on outer surface (W/m²K)
        inner_diameter: Inner diameter in meters (required for 'cylinder' geometry)
        outer_diameter: Outer diameter in meters (optional consistency check for 'cylinder')

    Returns:
        JSON string with calculated U-value and related parameters
    """
    # Validate inputs
    if not isinstance(layers, list) or len(layers) == 0:
        return json.dumps({"error": "At least one material layer must be provided in 'layers'."})
    if geometry.lower() == "cylinder" and inner_diameter is None:
        return json.dumps({"error": "Inner diameter is required for cylinder geometry."})

    try:
        # Process layers
        processed_layers = []
        calculated_outer_diameter = None
        current_radius = inner_diameter / 2.0 if inner_diameter else None
        layer_resistances = []

        for i, layer in enumerate(layers):
            thickness = layer.get("thickness")
            k = layer.get("thermal_conductivity_k")
            material_name = layer.get("material_name")

            # Validate layer data
            if thickness is None or thickness <= 0:
                return json.dumps({"error": f"Invalid thickness provided for layer {i+1}."})
            if k is None and material_name is None:
                return json.dumps({"error": f"Layer {i+1} must have either 'thermal_conductivity_k' or 'material_name'."})

            # If material name provided but no k, look up thermal conductivity
            if k is None:
                material_props_json = get_material_properties(material_name)
                material_props = json.loads(material_props_json)

                if "error" in material_props:
                    return json.dumps(
                        {"error": f"Failed to get material properties for layer {i+1}: {material_props['error']}"}
                    )

                k = material_props.get("thermal_conductivity_k")
                if k is None:
                    return json.dumps(
                        {"error": f"Could not determine thermal conductivity for material '{material_name}' in layer {i+1}."}
                    )

            # Validate thermal conductivity
            if k <= 0:
                return json.dumps({"error": f"Invalid thermal conductivity k={k} for layer {i+1}."})

            # Calculate resistance based on geometry
            if geometry.lower() == "flat_wall":
                # Flat wall: R = L/k
                R_layer = thickness / k
                layer_resistances.append(R_layer)

                # Add processed layer info
                processed_layers.append(
                    {
                        "layer": i + 1,
                        "material": material_name if material_name else "Unknown",
                        "thickness_m": thickness,
                        "thermal_conductivity_w_mk": k,
                        "thermal_resistance_m2k_w": R_layer,
                    }
                )

            elif geometry.lower() == "cylinder":
                # Cylindrical geometry
                if current_radius is None:
                    return json.dumps({"error": "Internal error: current_radius not set for cylinder."})

                outer_radius_layer = current_radius + thickness

                # For cylinder: R = ln(r2/r1) / (2πkL)
                # We calculate per unit length
                R_layer_per_L = math.log(outer_radius_layer / current_radius) / (2.0 * math.pi * k)

                # Store with radii info
                layer_data = {
                    "layer": i + 1,
                    "material": material_name if material_name else "Unknown",
                    "inner_radius_m": current_radius,
                    "outer_radius_m": outer_radius_layer,
                    "thickness_m": thickness,
                    "thermal_conductivity_w_mk": k,
                    "thermal_resistance_per_length_mk_w": R_layer_per_L,
                }

                processed_layers.append(layer_data)
                layer_resistances.append(R_layer_per_L)
                current_radius = outer_radius_layer  # Update for next layer

        # For cylinder, check calculated outer diameter vs provided
        if geometry.lower() == "cylinder":
            calculated_outer_diameter = current_radius * 2.0

            if outer_diameter and not math.isclose(calculated_outer_diameter, outer_diameter, rel_tol=1e-6):
                # Add warning but use calculated value
                logger.warning(
                    f"Provided outer diameter {outer_diameter} does not match calculated {calculated_outer_diameter}"
                )

        # Calculate total resistance and U-value
        if geometry.lower() == "flat_wall":
            # For flat wall: U = 1 / (Rconv_in + Rcond + Rconv_out)
            R_conv_inner = 1.0 / inner_convection_coefficient_h
            R_conv_outer = 1.0 / outer_convection_coefficient_h
            R_cond_total = sum(layer_resistances)
            R_total = R_conv_inner + R_cond_total + R_conv_outer
            U_value = 1.0 / R_total

            # Create result
            result = {
                "overall_heat_transfer_coefficient_U": U_value,
                "reference_area_basis": "flat (same for all surfaces)",
                "total_resistance_m2k_w": R_total,
                "convection_resistance_inner_m2k_w": R_conv_inner,
                "conduction_resistance_total_m2k_w": R_cond_total,
                "convection_resistance_outer_m2k_w": R_conv_outer,
                "layers": processed_layers,
                "geometry": geometry,
                "unit": "W/(m²·K)",
            }

        elif geometry.lower() == "cylinder":
            # For cylinder, need to specify reference area
            r_inner = inner_diameter / 2.0
            r_outer = calculated_outer_diameter / 2.0

            # Calculate resistances per unit length
            R_conv_inner_per_L = 1.0 / (inner_convection_coefficient_h * 2.0 * math.pi * r_inner)
            R_conv_outer_per_L = 1.0 / (outer_convection_coefficient_h * 2.0 * math.pi * r_outer)
            R_cond_total_per_L = sum(layer_resistances)
            R_total_per_L = R_conv_inner_per_L + R_cond_total_per_L + R_conv_outer_per_L

            # Calculate U-values based on different reference areas
            # U_outer = 1 / (A_outer * R_total) = 1 / (2πro * R_per_L)
            U_outer = 1.0 / (2.0 * math.pi * r_outer * R_total_per_L)

            # U_inner = 1 / (A_inner * R_total) = 1 / (2πri * R_per_L)
            U_inner = 1.0 / (2.0 * math.pi * r_inner * R_total_per_L)

            # By convention, often use outer area
            result = {
                "overall_heat_transfer_coefficient_U_outer": U_outer,
                "overall_heat_transfer_coefficient_U_inner": U_inner,
                "reference_area_basis": f"outer (Do={calculated_outer_diameter:.6f}m)",
                "total_resistance_per_length_mk_w": R_total_per_L,
                "convection_resistance_inner_per_length_mk_w": R_conv_inner_per_L,
                "conduction_resistance_total_per_length_mk_w": R_cond_total_per_L,
                "convection_resistance_outer_per_length_mk_w": R_conv_outer_per_L,
                "inner_diameter_m": inner_diameter,
                "calculated_outer_diameter_m": calculated_outer_diameter,
                "layers": processed_layers,
                "geometry": geometry,
                "unit": "W/(m²·K)",
            }
        else:
            return json.dumps({"error": f"Unsupported geometry: {geometry}. Use 'flat_wall' or 'cylinder'."})

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in calculate_overall_heat_transfer_coefficient: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
