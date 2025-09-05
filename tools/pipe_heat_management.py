"""
Omnibus tool: pipe_heat_management

Pipe heat loss and thermal management for above/below ground piping. Computes
heat loss per length, required insulation R-value, heat-trace power for
temperature maintenance and freeze protection, and simple time-to-freeze.

Consolidates: buried_object_heat_loss + convection_coefficient + overall_heat_transfer
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional

from tools.surface_heat_transfer import calculate_surface_heat_transfer
from tools.buried_object_heat_loss import calculate_buried_object_heat_loss
from tools.overall_heat_transfer import calculate_overall_heat_transfer_coefficient
from utils.validation import (
    ValidationError,
    require_positive,
    require_non_negative,
)

logger = logging.getLogger("heat-transfer-mcp.pipe_heat_management")


def _surface_loss_per_meter(
    outer_diameter_m: float,
    internal_temperature_K: float,
    ambient_temperature_K: float,
    wind_speed_m_s: float,
    surface_emissivity: float,
    wall_layers: Optional[List[Dict[str, Any]]],
    fluid_name_internal: str,
    fluid_name_external: str,
) -> Dict[str, Any]:
    """Use the surface_heat_transfer tool on a 1 m length to get W/m."""
    dims = {"diameter": float(outer_diameter_m), "length": 1.0}
    data = json.loads(
        calculate_surface_heat_transfer(
            geometry="pipe",
            dimensions=dims,
            internal_temperature=float(internal_temperature_K),
            ambient_air_temperature=float(ambient_temperature_K),
            wind_speed=float(wind_speed_m_s),
            surface_emissivity=float(surface_emissivity),
            overall_heat_transfer_coefficient_U=None,
            wall_layers=wall_layers or [],
            fluid_name_internal=fluid_name_internal,
            fluid_name_external=fluid_name_external,
            include_solar_gain=False,
            incident_solar_radiation=None,
        )
    )
    if "error" in data:
        return data
    q_w = data.get("total_heat_rate_loss_w", 0.0)
    L = 1.0
    return {
        "heat_loss_per_length_w_m": q_w / L,
        "details": data,
    }


def pipe_heat_management(
    # Geometry
    outer_diameter_m: float,
    length_m: float,
    # Thermal state
    internal_temperature_K: float,
    ambient_air_temperature_K: float,
    wind_speed_m_s: float = 2.0,
    fluid_name_internal: str = "water",
    fluid_name_external: str = "air",
    # Layers/insulation
    wall_layers: Optional[List[Dict[str, Any]]] = None,
    surface_emissivity: float = 0.85,
    # Installation
    installation: str = "above_ground",  # 'above_ground' or 'buried'
    burial_depth_m: Optional[float] = None,
    soil_conductivity_w_mk: Optional[float] = None,
    ground_surface_temperature_K: Optional[float] = None,
    # Objectives
    solve_for: Optional[str] = None,  # 'heat_trace_w_per_m', 'R_value', 'freeze_time_h'
    target_temperature_K: Optional[float] = None,  # For maintenance
    target_heat_loss_w_per_m: Optional[float] = None,
    # Freeze analysis
    pipe_inner_diameter_m: Optional[float] = None,
    fluid_density_kg_m3: Optional[float] = 1000.0,
    fluid_cp_j_kgk: Optional[float] = 4180.0,
    freeze_temperature_K: float = 273.15,
    include_latent_heat: bool = True,
    latent_heat_j_kg: float = 334000.0,
    heat_trace_w_per_m: float = 0.0,
) -> str:
    """Comprehensive pipe heat management tool.

    Args:
        outer_diameter_m: Outer diameter of pipe+insulation (m) for external losses.
        length_m: Pipe length (m).
        internal_temperature_K: Fluid temperature inside pipe (K).
        ambient_air_temperature_K: Ambient air temperature (K) (or ground surface temp for buried case if provided).
        wind_speed_m_s: Ambient wind (m/s) for above-ground.
        fluid_name_internal, fluid_name_external: Fluid names for property/convection usage in subtools.
        wall_layers: Layers from internal wall outwards including insulation; used for conduction to surface.
        surface_emissivity: Emissivity for radiation.
        installation: 'above_ground' or 'buried'.
        burial_depth_m, soil_conductivity_w_mk, ground_surface_temperature_K: Buried heat loss parameters.
        solve_for: One of {'heat_trace_w_per_m', 'R_value', 'freeze_time_h'}.
        target_temperature_K: Desired maintenance temperature for solve_for heat_trace.
        target_heat_loss_w_per_m: Desired cap on heat loss per meter (alternative to target_temperature).
        pipe_inner_diameter_m: For freeze time mass inventory; if None, assumes 80% of OD as ID.
        fluid_density_kg_m3, fluid_cp_j_kgk: Fluid properties for freeze-time.
        freeze_temperature_K: Freeze threshold.
        include_latent_heat: Include latent heat for freeze-time estimate.
        latent_heat_j_kg: Latent heat of fusion (J/kg).
        heat_trace_w_per_m: Installed heat trace for freeze-time or net loss calculations.

    Returns:
        JSON with heat loss per meter and requested design outputs.
    """
    try:
        # Basic input validation
        try:
            require_positive(float(outer_diameter_m), "outer_diameter_m")
            require_positive(float(length_m), "length_m")
            require_non_negative(float(wind_speed_m_s), "wind_speed_m_s")
        except ValidationError as ve:
            return json.dumps({"error": str(ve)})

        installation_lower = (installation or "above_ground").lower()
        if installation_lower not in {"above_ground", "buried"}:
            return json.dumps({"error": f"Unsupported installation: {installation}"})

        result: Dict[str, Any] = {
            "installation": installation_lower,
            "inputs_used": {
                "outer_diameter_m": outer_diameter_m,
                "length_m": length_m,
            },
            "calculation_methods": {
                "above_ground": "Iterative surface solver on 1 m section via tools.surface_heat_transfer",
                "buried": "Shape factor conduction via tools.buried_object_heat_loss (outer surface ~ contents temperature assumption)",
            },
        }

        if installation_lower == "above_ground":
            surf = _surface_loss_per_meter(
                outer_diameter_m=outer_diameter_m,
                internal_temperature_K=internal_temperature_K,
                ambient_temperature_K=ambient_air_temperature_K,
                wind_speed_m_s=wind_speed_m_s,
                surface_emissivity=surface_emissivity,
                wall_layers=wall_layers,
                fluid_name_internal=fluid_name_internal,
                fluid_name_external=fluid_name_external,
            )
            if "error" in surf:
                return json.dumps(surf)
            q_loss_w_m = surf.get("heat_loss_per_length_w_m", 0.0)
            result["heat_loss_per_length_w_m"] = q_loss_w_m
            result["surface_details"] = surf.get("details")
        else:
            if burial_depth_m is None or soil_conductivity_w_mk is None:
                return json.dumps({"error": "burial_depth_m and soil_conductivity_w_mk required for buried"})
            # Use contents temperature as outer surface proxy (conservative upper bound)
            obj_json = calculate_buried_object_heat_loss(
                object_type="pipe",
                diameter=float(outer_diameter_m),
                length=float(length_m),
                burial_depth=float(burial_depth_m),
                soil_conductivity=float(soil_conductivity_w_mk),
                object_temperature=float(internal_temperature_K),
                ground_surface_temperature=float(ground_surface_temperature_K or ambient_air_temperature_K),
            )
            obj = json.loads(obj_json)
            if "error" in obj:
                return json.dumps(obj)
            q_total = obj.get("total_heat_loss_watts", 0.0)
            q_loss_w_m = q_total / float(length_m)
            result["heat_loss_per_length_w_m"] = q_loss_w_m
            result["buried_details"] = obj

        # Solve-for handling
        if solve_for:
            s = str(solve_for).lower()
            if s == "heat_trace_w_per_m":
                if target_heat_loss_w_per_m is not None:
                    # Heat trace power equals net required to offset loss down to target
                    q_trace = max(0.0, result["heat_loss_per_length_w_m"] - float(target_heat_loss_w_per_m))
                    result.update({"mode": "solve_for", "solve_for": s, "required_heat_trace_w_per_m": q_trace})
                elif target_temperature_K is not None:
                    # Approximate: assume heat loss scales with (T_surface - T_ambient)
                    # Use linear proportion from current condition
                    delta_now = float(internal_temperature_K) - float(ambient_air_temperature_K)
                    delta_target = float(target_temperature_K) - float(ambient_air_temperature_K)
                    q_now = result["heat_loss_per_length_w_m"]
                    if delta_now <= 0:
                        q_trace = 0.0
                    else:
                        q_target = q_now * (delta_target / delta_now)
                        q_trace = max(0.0, q_now - q_target)
                    result.update({"mode": "solve_for", "solve_for": s, "required_heat_trace_w_per_m": q_trace})
                else:
                    return json.dumps({"error": "Provide target_heat_loss_w_per_m or target_temperature_K for heat_trace sizing"})
            elif s == "freeze_time_h":
                # Simple energy inventory to reach freeze_temperature and freeze
                Di = float(pipe_inner_diameter_m or (0.8 * float(outer_diameter_m)))
                area_flow = math.pi * (Di ** 2) / 4.0
                vol_per_m = area_flow * 1.0  # 1 m
                rho = float(fluid_density_kg_m3 or 1000.0)
                cp = float(fluid_cp_j_kgk or 4180.0)
                mass_per_m = rho * vol_per_m
                # sensible energy to cool to freeze temp
                sensible = mass_per_m * cp * max(0.0, float(internal_temperature_K) - float(freeze_temperature_K))
                latent = mass_per_m * float(latent_heat_j_kg) if include_latent_heat else 0.0
                energy_j = sensible + latent
                q_net = max(1e-6, result["heat_loss_per_length_w_m"] - float(heat_trace_w_per_m))  # W/m
                t_s = energy_j / q_net
                result.update({
                    "mode": "solve_for",
                    "solve_for": s,
                    "freeze_time_hours": t_s / 3600.0,
                    "assumptions": {
                        "includes_latent": include_latent_heat,
                        "net_loss_w_per_m": q_net,
                    },
                })
            elif s == "r_value":
                return json.dumps({"error": "R_value solve not yet implemented for pipe_heat_management"})
            else:
                return json.dumps({"error": f"Unsupported solve_for: {solve_for}"})

        return json.dumps(result)
    except Exception as e:
        logger.error(f"pipe_heat_management failed: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
