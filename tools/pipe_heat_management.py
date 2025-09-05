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
    solve_for: Optional[str] = None,  # 'heat_trace_w_per_m', 'heat_trace_steady_w_per_m', 'heat_trace_delta_w_per_m', 'freeze_protection_w_per_m', 'R_value', 'freeze_time_h'
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
    # Heat trace sizing helpers
    heat_trace_safety_factor: float = 1.25,
    freeze_protection_margin_K: float = 5.0,
    round_recommendation_to_catalog: bool = True,
    available_heat_trace_ratings_w_per_m: Optional[List[float]] = None,
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
        solve_for: One of {
            'heat_trace_w_per_m',           # Steady-state heat trace at maintenance temp (if target_temperature_K provided)
            'heat_trace_steady_w_per_m',    # Explicit steady-state mode (same as above)
            'heat_trace_delta_w_per_m',     # Delta power to change from internal_temperature to target_temperature (info-only)
            'freeze_protection_w_per_m',    # Heat trace to protect from freezing (target or freeze + margin)
            'R_value',
            'freeze_time_h'
        }.
        target_temperature_K: Desired maintenance temperature for steady-state heat trace sizing.
        target_heat_loss_w_per_m: Desired cap on heat loss per meter (alternative to target_temperature).
        pipe_inner_diameter_m: For freeze time mass inventory; if None, assumes 80% of OD as ID.
        fluid_density_kg_m3, fluid_cp_j_kgk: Fluid properties for freeze-time.
        freeze_temperature_K: Freeze threshold.
        include_latent_heat: Include latent heat for freeze-time estimate.
        latent_heat_j_kg: Latent heat of fusion (J/kg).
        heat_trace_w_per_m: Installed heat trace for freeze-time or net loss calculations.
        heat_trace_safety_factor: Safety factor for sizing recommendations (default 1.25).
        freeze_protection_margin_K: If target_temperature_K not given in freeze_protection mode, use freeze_temperature_K + margin.
        round_recommendation_to_catalog: If True, round recommended power up to a standard catalog size.
        available_heat_trace_ratings_w_per_m: Catalog ratings to select from; defaults to common values if None.

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

        # Common helper: recompute steady-state loss at a target temperature
        def _steady_state_loss_at_temperature(temp_K: float) -> Dict[str, Any]:
            if installation_lower == "above_ground":
                surf_t = _surface_loss_per_meter(
                    outer_diameter_m=outer_diameter_m,
                    internal_temperature_K=float(temp_K),
                    ambient_temperature_K=ambient_air_temperature_K,
                    wind_speed_m_s=wind_speed_m_s,
                    surface_emissivity=surface_emissivity,
                    wall_layers=wall_layers,
                    fluid_name_internal=fluid_name_internal,
                    fluid_name_external=fluid_name_external,
                )
                return {
                    "q_w_m": float(surf_t.get("heat_loss_per_length_w_m", 0.0)),
                    "details": surf_t.get("details"),
                }
            else:
                obj_json_t = calculate_buried_object_heat_loss(
                    object_type="pipe",
                    diameter=float(outer_diameter_m),
                    length=float(length_m),
                    burial_depth=float(burial_depth_m),
                    soil_conductivity=float(soil_conductivity_w_mk),
                    object_temperature=float(temp_K),
                    ground_surface_temperature=float(ground_surface_temperature_K or ambient_air_temperature_K),
                )
                obj_t = json.loads(obj_json_t)
                if "error" in obj_t:
                    return {"error": obj_t.get("error", "buried calc error")}
                q_total_t = obj_t.get("total_heat_loss_watts", 0.0)
                q_loss_t = q_total_t / float(length_m)
                return {"q_w_m": float(q_loss_t), "details": obj_t}

        # Solve-for handling
        if solve_for:
            s = str(solve_for).lower()
            if s in {"heat_trace_w_per_m", "heat_trace_steady_w_per_m", "freeze_protection_w_per_m", "heat_trace_delta_w_per_m"}:
                # Prepare common values
                q_now = float(result.get("heat_loss_per_length_w_m", 0.0))
                warnings: List[str] = []
                details: Dict[str, Any] = {}

                # Determine target temp for steady-state modes
                if s == "freeze_protection_w_per_m":
                    # Use provided target if any; otherwise freeze temp + margin
                    maintenance_target_K = (
                        float(target_temperature_K)
                        if target_temperature_K is not None
                        else float(freeze_temperature_K) + float(freeze_protection_margin_K)
                    )
                    if maintenance_target_K <= float(freeze_temperature_K):
                        warnings.append(
                            "Target temperature is at/below freeze temperature; increased to freeze + margin for sizing."
                        )
                        maintenance_target_K = float(freeze_temperature_K) + float(max(0.0, freeze_protection_margin_K))
                    # Compute steady-state loss at maintenance target
                    q_target_obj = _steady_state_loss_at_temperature(maintenance_target_K)
                    if "error" in q_target_obj:
                        return json.dumps({"error": q_target_obj["error"]})
                    q_target = float(q_target_obj["q_w_m"])
                    details_key = "buried_details_at_target" if installation_lower == "buried" else "surface_details_at_target"
                    details[details_key] = q_target_obj.get("details")

                    # Required heat trace equals steady-state loss at target (>= 0)
                    required = max(0.0, q_target)

                    # Recommendation with safety factor and optional rounding
                    rec = required * float(heat_trace_safety_factor)
                    catalog = available_heat_trace_ratings_w_per_m or [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
                    selected = None
                    if round_recommendation_to_catalog and rec > 0.0:
                        selected = next((r for r in catalog if r >= rec), catalog[-1])

                    # Ambient sanity checks
                    if float(ambient_air_temperature_K) >= maintenance_target_K:
                        warnings.append(
                            "Ambient >= maintenance target; heat trace may not be required under these conditions."
                        )

                    result.update({
                        "mode": "solve_for",
                        "solve_for": s,
                        "calculation_method": "steady_state_at_target_temperature",
                        "steady_state_loss_at_internal_w_per_m": q_now,
                        "steady_state_loss_at_target_w_per_m": q_target,
                        "required_heat_trace_w_per_m": required,
                        "heat_trace_safety_factor_used": float(heat_trace_safety_factor),
                        "heat_trace_sizing_recommendation_w_per_m": rec,
                        **({"catalog_rating_selected_w_per_m": selected} if selected is not None else {}),
                        "target_temperature_K_used": maintenance_target_K,
                        "warnings": warnings,
                    })
                    result.update(details)

                elif s in {"heat_trace_w_per_m", "heat_trace_steady_w_per_m"}:
                    if target_temperature_K is None and target_heat_loss_w_per_m is None:
                        return json.dumps({
                            "error": "Provide target_temperature_K for steady-state sizing or target_heat_loss_w_per_m to cap losses."
                        })

                    # If a heat loss cap is provided, compute delta from current condition
                    if target_heat_loss_w_per_m is not None and target_temperature_K is None:
                        q_trace_delta = max(0.0, q_now - float(target_heat_loss_w_per_m))
                        result.update({
                            "mode": "solve_for",
                            "solve_for": s,
                            "calculation_method": "delta_to_cap_current_loss",
                            "steady_state_loss_at_internal_w_per_m": q_now,
                            "target_heat_loss_w_per_m": float(target_heat_loss_w_per_m),
                            "additional_heat_trace_required_w_per_m": q_trace_delta,
                            "explanation": "Additional power needed to reduce current steady-state loss down to target_heat_loss_w_per_m.",
                        })
                    else:
                        # Steady-state at target temperature
                        q_target_obj = _steady_state_loss_at_temperature(float(target_temperature_K))
                        if "error" in q_target_obj:
                            return json.dumps({"error": q_target_obj["error"]})
                        q_target = float(q_target_obj["q_w_m"])
                        details_key = "buried_details_at_target" if installation_lower == "buried" else "surface_details_at_target"
                        details[details_key] = q_target_obj.get("details")
                        required = max(0.0, q_target)
                        delta_info = max(0.0, q_now - q_target)

                        if abs(float(internal_temperature_K) - float(target_temperature_K)) > 1e-6:
                            warnings.append(
                                "Computed steady-state heat loss at target_temperature_K (maintenance). The 'additional delta power' shown is informational."
                            )

                        result.update({
                            "mode": "solve_for",
                            "solve_for": s,
                            "calculation_method": "steady_state_at_target_temperature",
                            "steady_state_loss_at_internal_w_per_m": q_now,
                            "steady_state_loss_at_target_w_per_m": q_target,
                            "required_heat_trace_w_per_m": required,
                            "additional_heat_trace_delta_w_per_m": delta_info,
                            "target_temperature_K_used": float(target_temperature_K),
                            "warnings": warnings,
                        })
                        result.update(details)

                elif s == "heat_trace_delta_w_per_m":
                    # Informational delta power between internal and target temperatures
                    if target_temperature_K is None:
                        return json.dumps({"error": "Provide target_temperature_K for heat_trace_delta_w_per_m."})
                    q_target_obj = _steady_state_loss_at_temperature(float(target_temperature_K))
                    if "error" in q_target_obj:
                        return json.dumps({"error": q_target_obj["error"]})
                    q_target = float(q_target_obj["q_w_m"])
                    delta_power = max(0.0, q_now - q_target)
                    details_key = "buried_details_at_target" if installation_lower == "buried" else "surface_details_at_target"
                    details[details_key] = q_target_obj.get("details")
                    result.update({
                        "mode": "solve_for",
                        "solve_for": s,
                        "calculation_method": "delta_between_internal_and_target",
                        "steady_state_loss_at_internal_w_per_m": q_now,
                        "steady_state_loss_at_target_w_per_m": q_target,
                        "additional_heat_trace_delta_w_per_m": delta_power,
                        "target_temperature_K_used": float(target_temperature_K),
                    })
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
