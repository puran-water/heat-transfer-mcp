"""
Surface heat transfer tool to calculate net heat transfer from external surfaces.

This module provides functionality to calculate heat loss/gain from surfaces
considering convection, radiation, and optional solar gain.
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

from heat_transfer_mcp.convection.mixed import blend_convection_coefficients
from heat_transfer_mcp.utils.constants import STEFAN_BOLTZMANN
from heat_transfer_mcp.utils.helpers import estimate_sky_temperature
from heat_transfer_mcp.utils.validation import ValidationError, require_non_negative, validate_geometry_dimensions

from heat_transfer_mcp.tools.convection_coefficient import calculate_convection_coefficient
from heat_transfer_mcp.tools.fluid_properties import PropertyLookupRefused, calculation_property_source, property_refusal
from heat_transfer_mcp.tools.material_properties import get_material_properties
from heat_transfer_mcp.tools.overall_heat_transfer import calculate_overall_heat_transfer_coefficient

logger = logging.getLogger("heat-transfer-mcp.surface_heat_transfer")


def calculate_surface_heat_transfer(
    geometry: str,
    dimensions: Dict[str, float],
    internal_temperature: float,
    ambient_air_temperature: float,
    wind_speed: float,
    surface_emissivity: float,
    internal_to_surface_conductance: Optional[float] = None,
    wall_layers: Optional[List[Dict[str, Any]]] = None,
    fluid_name_internal: str = "water",
    fluid_name_external: str = "air",
    include_solar_gain: bool = False,
    incident_solar_radiation: Optional[float] = None,
    surface_absorptivity: float = 0.8,
    sky_temperature: Optional[float] = None,
    internal_convection_coefficient_h_override: Optional[float] = None,
    view_factor_sky_vertical: float = 0.5,  # Vertical walls see 50% sky
    view_factor_sky_horizontal: float = 1.0,  # Roof sees 100% sky
    ground_temperature: Optional[float] = None,  # Default to ambient
    property_source: str = "CoolProp",
) -> str:
    """Calculates net heat loss/gain from an external surface considering convection and radiation.

    Args:
        geometry: Surface geometry type (e.g., 'vertical_cylinder_tank', 'flat_surface')
        dimensions: Dictionary of dimensions in meters (e.g., {diameter: d, height: h})
        internal_temperature: Temperature of the fluid/contents inside in Kelvin
        ambient_air_temperature: Ambient air temperature in Kelvin
        wind_speed: Wind speed in m/s
        surface_emissivity: Emissivity of the outer surface (0 to 1)
        internal_to_surface_conductance: Conductance from contents to outer surface, on outer area (W/m²K); EXCLUDES external convection/radiation
        wall_layers: Optional wall layers if internal_to_surface_conductance is not provided
        fluid_name_internal: Internal fluid name (default 'water')
        fluid_name_external: External fluid name (default 'air')
        include_solar_gain: Whether to include solar gain in the calculation
        incident_solar_radiation: Area-averaged incident flux on the modeled surface (W/m²), after projection/shading
        surface_absorptivity: Absorptivity of outer surface for solar radiation (0 to 1)
        sky_temperature: Effective sky temperature for radiation calculation in Kelvin
        property_source: The branch every external-fluid property lookup reads (CALCULATION_PROPERTY_RULE)

    Returns:
        JSON string with calculated heat transfer results
    """
    # Basic input validation (allow no layers/U => internal convection only)
    try:
        for name, value in {"internal_temperature": internal_temperature, "ambient_air_temperature": ambient_air_temperature,
                            "sky_temperature": sky_temperature, "ground_temperature": ground_temperature,
                            "internal_to_surface_conductance": internal_to_surface_conductance,
                            "internal_convection_coefficient_h_override": internal_convection_coefficient_h_override}.items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValidationError(f"{name} must be positive and finite")
        for name, value in {"surface_emissivity": surface_emissivity, "surface_absorptivity": surface_absorptivity,
                            "view_factor_sky_vertical": view_factor_sky_vertical, "view_factor_sky_horizontal": view_factor_sky_horizontal}.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValidationError(f"{name} must lie in [0, 1]")
        require_non_negative(float(wind_speed), "wind_speed")
        validate_geometry_dimensions(geometry, dimensions)
    except ValidationError as ve:
        return json.dumps({"error": str(ve)})

    if include_solar_gain and incident_solar_radiation is None:
        return json.dumps({"error": "Incident solar radiation is required when include_solar_gain is True."})
    if incident_solar_radiation is not None:
        try:
            require_non_negative(float(incident_solar_radiation), "incident_solar_radiation")
        except ValidationError as ve:
            return json.dumps({"error": str(ve)})

    try:
        property_source = calculation_property_source(property_source)
        # 1. Calculate Surface Area
        outer_surface_area = 0.0
        lateral_area = 0.0
        top_area = 0.0
        geometry_lower = geometry.lower()
        diameter = dimensions.get("diameter")
        height = dimensions.get("height")
        length = dimensions.get("length")
        width = dimensions.get("width")

        if ("cylinder_tank" in geometry_lower or "tank" in geometry_lower) and diameter and (height or length):
            if "vertical" in geometry_lower:
                # Vertical cylinder tanks: walls + top cap only (bottom is ground-contact)
                axial_dim = height if height else length  # height is preferred for vertical
                lateral_area = math.pi * diameter * axial_dim
                top_area = math.pi * (diameter / 2) ** 2
                outer_surface_area = lateral_area + top_area
            elif "horizontal" in geometry_lower:
                # Horizontal cylinder - use length (or height as fallback)
                axial_dim = length if length else height
                outer_surface_area = math.pi * diameter * axial_dim + 2 * math.pi * (diameter / 2) ** 2
            else:
                # Default to vertical if not specified - includes both endcaps
                axial_dim = height if height else length
                outer_surface_area = math.pi * diameter * axial_dim + 2 * math.pi * (diameter / 2) ** 2
        elif "flat_surface" in geometry_lower or "wall" in geometry_lower:
            if length and width:
                outer_surface_area = length * width
            else:
                return json.dumps({"error": f"Missing dimensions for {geometry}. Need 'length' and 'width'."})
        elif "pipe" in geometry_lower and diameter and length:
            outer_surface_area = math.pi * diameter * length
        else:
            return json.dumps({"error": f"Unsupported or incompletely defined geometry: {geometry}"})

        # 2. Estimate surface temperature and heat transfer iteratively
        # Start with an initial guess
        T_internal_K = internal_temperature
        T_ambient_K = ambient_air_temperature

        # Determine sky temperature if not provided
        T_sky_K = sky_temperature if sky_temperature else estimate_sky_temperature(T_ambient_K)

        # Iteration setup
        max_iterations = 50
        tolerance_W = 1.0  # Watts - dimensionally consistent tolerance
        # Determine internal convection estimate regardless of wall specification
        if internal_convection_coefficient_h_override is not None:
            h_inner_assumed = float(internal_convection_coefficient_h_override)
        else:
            if fluid_name_internal.lower() == "water":
                h_inner_assumed = 1000.0  # W/m²K (typical for water)
            else:
                h_inner_assumed = 100.0  # W/m²K (moderate value for other fluids)

        # Calculate internal thermal resistance path
        if internal_to_surface_conductance is not None:
            # Use the provided U-value directly
            R_internal_plus_wall = 1.0 / float(internal_to_surface_conductance)
        else:
            # wall_layers may be None or empty -> treat as no wall conduction
            if wall_layers and len(wall_layers) > 0:
                if "cylinder" in geometry_lower or "pipe" in geometry_lower:
                    # Use overall_heat_transfer tool for cylindrical calculation excluding external convection
                    inner_diameter = dimensions.get("inner_diameter")
                    if inner_diameter is None:
                        total_wall_thickness = sum(layer.get("thickness", 0) for layer in wall_layers)
                        inner_diameter = diameter - 2 * total_wall_thickness

                    u_value_json = calculate_overall_heat_transfer_coefficient(
                        geometry="cylinder",
                        layers=wall_layers,
                        inner_convection_coefficient_h=h_inner_assumed,
                        outer_convection_coefficient_h=1e10,  # exclude external resistance
                        inner_diameter=inner_diameter,
                        outer_diameter=diameter,
                    )
                    u_value_data = json.loads(u_value_json)
                    if "error" in u_value_data:
                        return json.dumps(
                            {"error": f"Failed to calculate overall heat transfer coefficient: {u_value_data['error']}"}
                        )
                    R_conv_inner_per_L = u_value_data.get("convection_resistance_inner_per_length_mk_w", 0)
                    R_cond_total_per_L = u_value_data.get("conduction_resistance_total_per_length_mk_w", 0)
                    r_outer = diameter / 2
                    A_out_per_L = 2 * math.pi * r_outer
                    # Convert per-length resistance (K·m/W) to area-based (K·m²/W) by multiplying by outer area per unit length
                    R_internal_plus_wall = (R_conv_inner_per_L + R_cond_total_per_L) * A_out_per_L
                else:
                    # Flat wall path
                    R_wall_total = 0.0
                    for layer in wall_layers:
                        thickness = layer.get("thickness")
                        k = layer.get("thermal_conductivity_k")
                        material_name = layer.get("material_name")
                        if k is None and material_name:
                            material_props_json = get_material_properties(material_name)
                            material_props = json.loads(material_props_json)
                            if "error" in material_props:
                                return json.dumps({"error": f"Failed to get material properties: {material_props['error']}"})
                            k = material_props.get("thermal_conductivity_k")
                        if thickness is None or k is None:
                            return json.dumps({"error": "Missing thickness or thermal conductivity for wall layer."})
                        R_wall_total += thickness / k
                    R_internal_plus_wall = 1.0 / h_inner_assumed + R_wall_total
            else:
                # No layers provided -> internal convection only
                R_internal_plus_wall = 1.0 / h_inner_assumed

        # Solve the actual wall/surface energy residual. A cold sky can cool
        # cladding below ambient; solar can raise it above the contents.
        from scipy.optimize import brentq

        T_ground_K = ground_temperature if ground_temperature is not None else T_ambient_K
        if lateral_area and top_area:
            f_sky = (lateral_area * view_factor_sky_vertical + top_area * view_factor_sky_horizontal) / outer_surface_area
        elif "horizontal" in geometry_lower:
            f_sky = 0.5 if "cylinder" in geometry_lower else view_factor_sky_horizontal
        else:
            f_sky = view_factor_sky_vertical
        T_eff4 = f_sky * T_sky_K**4 + (1 - f_sky) * T_ground_K**4
        Q_solar = incident_solar_radiation * surface_absorptivity * outer_surface_area if include_solar_gain else 0.0
        calls = 0
        data_sources = set()  # the branch each convection lookup answered from

        def coefficient(Ts, geometry, dimension, flow_type):
            values = json.loads(calculate_convection_coefficient(
                geometry=geometry, characteristic_dimension=dimension,
                fluid_name=fluid_name_external, bulk_fluid_temperature=T_ambient_K,
                surface_temperature=Ts, pressure=101325.0, flow_type=flow_type,
                fluid_velocity=wind_speed if flow_type == "forced" else None, property_source=property_source,
            ))
            if values.get("code") == "property_source_refused":
                raise PropertyLookupRefused(values)
            if "error" in values:
                raise ValueError(f"External {flow_type} convection refused: {values['error']}")
            data_sources.add(values.get("data_source"))
            return values["convection_coefficient_h"]

        def rates(Ts):
            nonlocal calls
            calls += 1
            if lateral_area and top_area:
                # Vertical side uses height; upward-facing cap uses A/perimeter.
                h_side = coefficient(Ts, "vertical_wall", height or length, "natural")
                h_cap = coefficient(Ts, "horizontal_plate_up", diameter / 4, "natural")
                hn = (h_side * lateral_area + h_cap * top_area) / outer_surface_area
            else:
                if "flat" in geometry_lower or "wall" in geometry_lower:
                    if "vertical" in geometry_lower or "wall" in geometry_lower:
                        g, dim = "vertical_wall", length
                    elif "horizontal" in geometry_lower:
                        g, dim = "horizontal_plate_up", length * width / (2 * (length + width))
                    else:
                        raise ValueError("Flat-surface natural convection requires an explicit vertical or horizontal orientation")
                else:
                    g, dim = "pipe_external", diameter
                hn = coefficient(Ts, g, dim, "natural")
            h = hn
            if wind_speed > 0:
                forced_g = "pipe_external" if diameter else "flat_plate_external"
                hf = coefficient(Ts, forced_g, diameter or length, "forced")
                h = blend_convection_coefficients(hf, hn, n=3.0)
            q_cond = (T_internal_K - Ts) * outer_surface_area / R_internal_plus_wall
            q_conv = h * outer_surface_area * (Ts - T_ambient_K)
            q_rad = surface_emissivity * STEFAN_BOLTZMANN * outer_surface_area * (Ts**4 - T_eff4)
            return q_cond, q_conv, q_rad, h

        def residual(Ts):
            qc, qv, qr, _ = rates(Ts)
            return qc - qv - qr + Q_solar

        lower = min(T_internal_K, T_ambient_K, T_sky_K, T_ground_K)
        upper = max(T_internal_K, T_ambient_K, T_sky_K, T_ground_K)
        if lower == upper and Q_solar == 0:
            Ts = lower
        else:
            # Solar can put the root outside the no-source temperature bracket.
            while residual(upper) > 0:
                upper *= 1.2
            Ts = brentq(residual, lower, upper, xtol=1e-9, maxiter=max_iterations)
        qc, qv, qr, h = rates(Ts)
        qnet = qv + qr - Q_solar
        closure = qc - qnet
        if abs(closure) > tolerance_W:
            raise ValueError(f"Surface energy balance did not close: residual {closure} W")
        assumptions = ["External pressure 101325 Pa", "Mixed convection: cubic blend",
                       "Uniform outer surface temperature; closed tank cap; no evaporation",
                       "Tank lateral natural convection uses a vertical-plate approximation; curvature is not resolved"]
        if internal_convection_coefficient_h_override is None and internal_to_surface_conductance is None:
            assumptions.append(f"Internal convection assumed {h_inner_assumed} W/m2/K")
        if sky_temperature is None:
            assumptions.append("Sky temperature estimated from ambient; no measured longwave radiation")
        return json.dumps({
            "total_heat_rate_loss_w": qnet, "convective_heat_rate_w": qv,
            "radiative_heat_rate_w": qr, "solar_gain_rate_w": Q_solar,
            "conduction_heat_rate_w": qc, "energy_balance_residual_w": closure,
            "estimated_outer_surface_temp_k": Ts, "estimated_outer_surface_temp_c": Ts - 273.15,
            "outer_surface_area_m2": outer_surface_area, "external_convection_coefficient_w_m2k": h,
            "internal_plus_wall_resistance_k_w": R_internal_plus_wall / outer_surface_area,
            "geometry": geometry, "converged": True, "iterations_required": calls,
            "data_source": ", ".join(sorted(filter(None, data_sources))) or None,
            "sky_temperature_k": T_sky_K, "ground_temperature_k": T_ground_K,
            "radiation_model": {"effective_env_temp_k": T_eff4**0.25,
                                "view_factors": {"sky_vertical": view_factor_sky_vertical,
                                                 "sky_horizontal": view_factor_sky_horizontal}},
            "assumptions": assumptions,
        }, allow_nan=False)

    except PropertyLookupRefused as refused:
        return property_refusal(refused, "External convection refused")
    except Exception as e:
        logger.error(f"Unexpected error in calculate_surface_heat_transfer: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
