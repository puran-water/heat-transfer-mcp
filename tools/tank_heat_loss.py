"""
Omnibus tool: tank_heat_loss

Unified tank and vessel heat loss analysis with iterative surface-temperature
balancing, multi-layer walls/insulation, ambient effects (wind, radiation),
and optional ground/soil coupling. Supports solve_for and parameter sweeps,
and can use historical weather extremes when available.

Consolidates: surface_heat_transfer + ambient_conditions + overall_heat_transfer
+ convection_coefficient + material_properties + ground_heat_loss

Primary use cases optimized:
- Equalization tanks, digesters, storage tanks (vertical/horizontal cylinder,
  sphere, rectangular/flat surfaces)
- Iterative solution for outer cladding temperature (conduction to cladding
  equals convection+radiation losses, with optional solar gain)
- Above-ground and buried/ground-contact configurations
- Multi-layer insulation via thickness/k or equivalent R-values
- Solve-for capability (e.g., R-value required to meet max heat loss)
- Parameter sweep for insulation and ambient conditions
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from utils.import_helpers import METEOSTAT_AVAILABLE, PANDAS_AVAILABLE
from utils.constants import STEFAN_BOLTZMANN
from utils.weather_service import get_weather_service
from utils.helpers import estimate_sky_temperature

# Reuse existing tool functions
from tools.surface_heat_transfer import calculate_surface_heat_transfer
from tools.ground_heat_loss import calculate_ground_heat_loss
from utils.validation import (
    ValidationError,
    require_positive,
    require_non_negative,
    validate_geometry_dimensions,
)

logger = logging.getLogger("heat-transfer-mcp.tank_heat_loss")


# ----------------------------- Data Structures ----------------------------- #


@dataclass
class AmbientSpec:
    T_air_K: float
    wind_m_s: float
    T_sky_K: Optional[float] = None
    solar_irradiance_W_m2: Optional[float] = None
    source: str = "direct"


# ------------------------------ Helper Logic ------------------------------ #


def _make_wall_layers_from_R(
    total_R_insulation_m2K_W: float,
    default_k_insulation_W_mK: float = 0.035,
) -> List[Dict[str, Any]]:
    """Create an equivalent single insulation layer from an R-value.

    The relationship for a flat layer is R = L/k.
    For cylindrical, this serves as an approximation; the overall tool uses
    calculate_overall_heat_transfer_coefficient which handles cylindrical layers
    more rigorously when geometric radii are available.
    """
    if total_R_insulation_m2K_W <= 0:
        raise ValueError("total_R_insulation_m2K_W must be positive")
    if default_k_insulation_W_mK <= 0:
        raise ValueError("default_k_insulation_W_mK must be positive")
    thickness = total_R_insulation_m2K_W * default_k_insulation_W_mK
    return [
        {
            "material_name": "equivalent_insulation",
            "thickness": thickness,
            "thermal_conductivity_k": default_k_insulation_W_mK,
        }
    ]


@lru_cache(maxsize=64)
def _percentile_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    percentile: float,
    time_resolution: str = "daily",
) -> Optional[Dict[str, Any]]:
    """Query meteostat to compute percentile ambient conditions.

    Returns dict with keys: T_air_K, wind_m_s, meta
    """
    if not METEOSTAT_AVAILABLE or not PANDAS_AVAILABLE:
        return None

    try:
        from meteostat import Point, Daily, Hourly
        import pandas as pd

        # Parse date strings to datetime objects
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date

        loc = Point(float(latitude), float(longitude))
        if time_resolution.lower() == "hourly":
            ds = Hourly(loc, start_dt, end_dt).fetch()
        else:
            ds = Daily(loc, start_dt, end_dt).fetch()
        if ds is None or ds.empty:
            return None

        # Meteostat default cols include tavg (°C), tmin, tmax, wspd (km/h) or wdir/gust depending on res
        # For robustness, compute percentile on tmin for conservative design if available; else tavg.
        col_temp = "tmin" if "tmin" in ds.columns else ("tavg" if "tavg" in ds.columns else None)
        if col_temp is None:
            return None
        # Wind speed: prefer 'wspd' (km/h); fall back to derived if missing
        col_wind = "wspd" if "wspd" in ds.columns else None

        # For cold design, convert percentile: p=0.99 -> use 0.01 quantile
        cold_quantile = 1.0 - percentile if percentile > 0.5 else percentile
        temp_percentile_C = ds[col_temp].quantile(cold_quantile)

        # For wind during cold periods, use high percentile
        # First find cold days, then compute wind percentile within those
        if col_wind and col_wind in ds.columns:
            cold_days = ds[ds[col_temp] <= temp_percentile_C]
            if not cold_days.empty:
                wind_percentile_kmh = cold_days[col_wind].quantile(0.95)  # High wind during cold
            else:
                wind_percentile_kmh = ds[col_wind].quantile(0.95)  # Fallback
            wind_m_s = float(wind_percentile_kmh) / 3.6 if pd.notnull(wind_percentile_kmh) else 2.0
        else:
            wind_m_s = 2.0

        T_air_K = float(temp_percentile_C) + 273.15 if pd.notnull(temp_percentile_C) else None
        if T_air_K is None:
            return None
        return {
            "T_air_K": T_air_K,
            "wind_m_s": wind_m_s,
            "meta": {
                "percentile": percentile,
                "time_resolution": time_resolution,
                "columns": list(ds.columns),
            },
        }
    except Exception as e:
        logger.warning(f"Percentile weather fetch failed: {e}")
        return None


def _ambient_from_inputs(
    ambient_air_temperature: Optional[float],
    wind_speed: Optional[float],
    sky_temperature: Optional[float],
    include_solar_gain: bool,
    incident_solar_radiation: Optional[float],
    latitude: Optional[float],
    longitude: Optional[float],
    start_date: Optional[str],
    end_date: Optional[str],
    design_percentile: Optional[float],
    time_resolution: str = "daily",
    weather_mode: str = "auto",  # New parameter: "auto", "meteostat", "direct"
) -> Tuple[AmbientSpec, Optional[Dict[str, Any]]]:
    """Determine ambient conditions from direct values or weather statistics.

    Returns AmbientSpec and optional info dict describing the source.
    """
    info: Dict[str, Any] = {}

    # Determine weather mode
    if weather_mode == "auto":
        # Use Meteostat if coordinates available
        if latitude is not None and longitude is not None:
            weather_mode = "meteostat"
        else:
            weather_mode = "direct"

    # Try to get weather data if coordinates are available (even if direct values provided)
    weather_data = None
    dew_point_k = None
    if weather_mode == "meteostat" and latitude is not None and longitude is not None:
        try:
            weather_service = get_weather_service()
            # Convert dates if provided as strings
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) and start_date else None
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) and end_date else None

            # Get design conditions from weather service
            weather_data = weather_service.get_design_conditions(
                lat=float(latitude),
                lon=float(longitude),
                start_date=start_dt,
                end_date=end_dt,
                percentiles=[design_percentile] if design_percentile else [0.95],
                time_resolution=time_resolution,
            )

            # Extract dew point for sky temperature calculation
            if "concurrent_conditions" in weather_data:
                dew_point_k = weather_data["concurrent_conditions"].get("dew_point_k")
        except Exception as e:
            logger.warning(f"Failed to fetch weather data: {e}")
            weather_data = None

    # Hybrid mode - use provided ambient if given, else use fetched
    if ambient_air_temperature is not None and wind_speed is not None:
        # Direct values provided
        T_air = float(ambient_air_temperature)

        # Calculate sky temperature using dew point if available
        if sky_temperature is None and dew_point_k:
            sky_temperature = estimate_sky_temperature(T_air, dew_point_k)

        ambient = AmbientSpec(
            T_air_K=T_air,
            wind_m_s=float(wind_speed),
            T_sky_K=sky_temperature,
            solar_irradiance_W_m2=incident_solar_radiation if include_solar_gain else None,
            source="hybrid" if weather_data else "direct",
        )
        info["ambient_source"] = "hybrid" if weather_data else "direct"
        if weather_data:
            info["weather_meta"] = weather_data.get("data_summary", {})
            info["dew_point_k"] = dew_point_k
            info["sky_temperature_k"] = sky_temperature
        return ambient, info

    # Use fetched weather data if available
    if weather_data and "design_conditions" in weather_data and design_percentile:
        # Find the appropriate percentile data
        percentile_key = f"cold_{int(design_percentile*100)}th"
        if percentile_key in weather_data["design_conditions"]:
            design_cond = weather_data["design_conditions"][percentile_key]
            T_air = design_cond["temp_k"]

            # Calculate sky temperature using dew point if available
            if sky_temperature is None and dew_point_k:
                sky_temperature = estimate_sky_temperature(T_air, dew_point_k)

            ambient = AmbientSpec(
                T_air_K=T_air,
                wind_m_s=design_cond.get("wind_m_s", 2.0),
                T_sky_K=sky_temperature,
                solar_irradiance_W_m2=incident_solar_radiation if include_solar_gain else None,
                source="meteostat",
            )
            info["ambient_source"] = "meteostat"
            info["weather_meta"] = weather_data.get("data_summary", {})
            info["dew_point_k"] = dew_point_k
            info["sky_temperature_k"] = sky_temperature
            return ambient, info

    # Fallback minimal ambient
    ambient = AmbientSpec(
        T_air_K=float(ambient_air_temperature) if ambient_air_temperature is not None else 273.15,
        wind_m_s=float(wind_speed) if wind_speed is not None else 2.0,
        T_sky_K=sky_temperature,
        solar_irradiance_W_m2=incident_solar_radiation if include_solar_gain else None,
        source="fallback",
    )
    info["ambient_source"] = "fallback"
    return ambient, info


def _run_surface_solver(
    geometry: str,
    dimensions: Dict[str, float],
    internal_temperature: float,
    surface_emissivity: float,
    ambient: AmbientSpec,
    fluid_name_internal: str,
    fluid_name_external: str,
    wall_layers: Optional[List[Dict[str, Any]]],
    overall_heat_transfer_coefficient_U: Optional[float],
    h_inner_override_w_m2k: Optional[float] = None,
) -> Dict[str, Any]:
    """Delegate to the existing surface solver and parse JSON output into dict."""
    try:
        res_json = calculate_surface_heat_transfer(
            geometry=geometry,
            dimensions=dimensions,
            internal_temperature=internal_temperature,
            ambient_air_temperature=ambient.T_air_K,
            wind_speed=ambient.wind_m_s,
            surface_emissivity=surface_emissivity,
            overall_heat_transfer_coefficient_U=overall_heat_transfer_coefficient_U,
            wall_layers=wall_layers,
            fluid_name_internal=fluid_name_internal,
            fluid_name_external=fluid_name_external,
            include_solar_gain=ambient.solar_irradiance_W_m2 is not None,
            incident_solar_radiation=ambient.solar_irradiance_W_m2,
            surface_absorptivity=0.8,
            sky_temperature=ambient.T_sky_K,
            internal_convection_coefficient_h_override=h_inner_override_w_m2k,
        )
        data = json.loads(res_json)
        return data
    except Exception as e:
        return {"error": f"surface solver failed: {e}"}


def _add_ground_loss_if_requested(
    include_ground: bool,
    ground_config: Optional[Dict[str, Any]],
    contents_temperature: float,
    average_external_air_temperature: Optional[float],
) -> Dict[str, Any]:
    """Optionally compute ground/foundation heat loss and return dict with results."""
    if not include_ground or not ground_config:
        return {"ground_heat_loss_watts": 0.0}

    try:
        # Prepare call
        structure_type = ground_config.get("structure_type", "slab_on_grade")
        dimensions = ground_config.get("dimensions", {})
        depth = ground_config.get("depth", 0.0)
        insulation_R_value_si = ground_config.get("insulation_R_value_si", 0.0)
        wall_thickness = ground_config.get("wall_thickness", 0.2)
        wall_conductivity = ground_config.get("wall_conductivity", 1.7)
        soil_conductivity = ground_config.get("soil_conductivity")

        if average_external_air_temperature is None:
            # Use ambient of 10C as neutral if not provided
            average_external_air_temperature = 283.15

        gh_json = calculate_ground_heat_loss(
            structure_type=structure_type,
            dimensions=dimensions,
            depth=depth,
            insulation_R_value_si=insulation_R_value_si,
            wall_thickness=wall_thickness,
            wall_conductivity=wall_conductivity,
            soil_conductivity=soil_conductivity,
            internal_temperature=contents_temperature,
            average_external_air_temperature=average_external_air_temperature,
        )
        gh = json.loads(gh_json)
        if "error" in gh:
            return {"ground_error": gh["error"], "ground_heat_loss_watts": 0.0}
        return {
            "ground_heat_loss_watts": gh.get("ground_heat_loss_watts", 0.0),
            "ground_details": gh,
        }
    except Exception as e:
        return {"ground_error": str(e), "ground_heat_loss_watts": 0.0}


def _solve_for_R_value(
    target_heat_loss_W: float,
    geometry: str,
    dimensions: Dict[str, float],
    contents_temperature: float,
    ambient: AmbientSpec,
    surface_emissivity: float,
    fluid_name_internal: str,
    fluid_name_external: str,
    base_wall_layers: Optional[List[Dict[str, Any]]],
    default_k_insulation: float,
    r_bounds: Tuple[float, float] = (0.0, 10.0),
    tol_W: float = 5.0,
    max_iter: int = 40,
) -> Dict[str, Any]:
    """Find additional insulation R-value needed to meet a target heat loss.

    Uses a bounded secant/bisection hybrid for robustness.
    """
    R_lo, R_hi = r_bounds
    R_lo = max(0.0, float(R_lo))
    R_hi = max(R_lo + 1e-6, float(R_hi))

    def heat_loss_for_R(R_add: float) -> Tuple[float, Dict[str, Any]]:
        layers = list(base_wall_layers) if base_wall_layers else []
        if R_add > 0.0:
            layers = layers + _make_wall_layers_from_R(R_add, default_k_insulation)
        data = _run_surface_solver(
            geometry=geometry,
            dimensions=dimensions,
            internal_temperature=contents_temperature,
            surface_emissivity=surface_emissivity,
            ambient=ambient,
            fluid_name_internal=fluid_name_internal,
            fluid_name_external=fluid_name_external,
            wall_layers=layers,
            overall_heat_transfer_coefficient_U=None,
        )
        Q = data.get("total_heat_rate_loss_w", float("inf")) if isinstance(data, dict) else float("inf")
        return Q, data

    # Evaluate at bounds
    Q_lo, data_lo = heat_loss_for_R(R_lo)
    Q_hi, data_hi = heat_loss_for_R(R_hi)

    # If already under target at R_lo, return quickly
    if Q_lo <= target_heat_loss_W:
        return {
            "required_additional_R_value_m2K_W": R_lo,
            "heat_loss_w": Q_lo,
            "iterations": 0,
            "solver": "bracket_check",
            "final_details": data_lo,
        }

    # Expand upper bound if needed
    expand_count = 0
    while Q_hi > target_heat_loss_W and expand_count < 10:
        R_hi *= 2.0
        Q_hi, data_hi = heat_loss_for_R(R_hi)
        expand_count += 1

    # Root find on f(R) = Q(R) - target
    f_lo = Q_lo - target_heat_loss_W
    f_hi = Q_hi - target_heat_loss_W

    if f_lo * f_hi > 0:
        # Failed to bracket; return best seen
        best_R = R_hi if Q_hi < Q_lo else R_lo
        best_data = data_hi if Q_hi < Q_lo else data_lo
        return {
            "error": "Could not bracket solution for R-value",
            "best_R_value_m2K_W": best_R,
            "heat_loss_w": Q_hi if Q_hi < Q_lo else Q_lo,
            "iterations": 0,
            "solver": "bracketing_failed",
            "final_details": best_data,
        }

    R_a, R_b = R_lo, R_hi
    f_a, f_b = f_lo, f_hi
    last_details = None
    for i in range(max_iter):
        # Secant step within bounds
        if f_b != f_a:
            R_c = R_b - f_b * (R_b - R_a) / (f_b - f_a)
        else:
            R_c = (R_a + R_b) / 2.0
        # Keep inside bounds
        if not (R_a <= R_c <= R_b):
            R_c = (R_a + R_b) / 2.0
        Q_c, details = heat_loss_for_R(R_c)
        last_details = details
        f_c = Q_c - target_heat_loss_W
        if abs(f_c) <= tol_W:
            return {
                "required_additional_R_value_m2K_W": R_c,
                "heat_loss_w": Q_c,
                "iterations": i + 1,
                "solver": "secant_bisection",
                "final_details": details,
            }
        # Bisection update
        if f_a * f_c < 0:
            R_b, f_b = R_c, f_c
        else:
            R_a, f_a = R_c, f_c

    return {
        "error": "R-value solver did not converge",
        "best_R_value_m2K_W": (R_a + R_b) / 2.0,
        "iterations": max_iter,
        "final_details": last_details,
    }


def _grid_sweep(base_params: Dict[str, Any], sweep: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product sweep over provided parameter lists."""
    # Build list of keys and list-of-lists of values
    keys = list(sweep.keys())
    lists = [sweep[k] for k in keys]

    def _product(accum: List[Tuple[str, Any]], depth: int) -> List[List[Tuple[str, Any]]]:
        if depth == len(keys):
            return [accum]
        out: List[List[Tuple[str, Any]]] = []
        k = keys[depth]
        for v in lists[depth]:
            out.extend(_product(accum + [(k, v)], depth + 1))
        return out

    combos = _product([], 0)
    results: List[Dict[str, Any]] = []
    for combo in combos:
        params = dict(base_params)
        for k, v in combo:
            params[k] = v
        results.append(params)
    return results


# --------------------------------- Tool API -------------------------------- #


def tank_heat_loss(
    # Geometry and configuration
    geometry: str,
    dimensions: Dict[str, float],
    contents_temperature: float,
    fluid_name_internal: str = "water",
    fluid_name_external: str = "air",
    # Headspace parameters (for tanks with gas space above liquid)
    headspace_height_m: float = 0,
    headspace_fluid: str = "air",
    headspace_h_inner_override_w_m2k: Optional[float] = None,
    # Wall/insulation specification
    wall_layers: Optional[List[Dict[str, Any]]] = None,
    insulation_R_value_si: Optional[float] = None,
    assumed_insulation_k_w_mk: float = 0.035,
    # Ambient specification (direct or percentile)
    ambient_air_temperature: Optional[float] = None,
    wind_speed: Optional[float] = None,
    sky_temperature: Optional[float] = None,
    include_solar_gain: bool = False,
    incident_solar_radiation: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    design_percentile: Optional[float] = None,
    time_resolution: str = "daily",
    percentiles: Optional[List[float]] = None,
    # Surface properties
    surface_emissivity: float = 0.85,
    # Ground coupling (optional)
    include_ground_contact: bool = False,
    ground_config: Optional[Dict[str, Any]] = None,
    average_external_air_temperature: Optional[float] = None,
    # Advanced capabilities
    solve_for: Optional[str] = None,  # e.g., 'R_value'
    target_heat_loss_w: Optional[float] = None,
    sweep: Optional[Dict[str, List[Any]]] = None,
) -> str:
    """Comprehensive tank/vessel heat loss calculator with iterate-and-balance solver.

    Args:
        geometry: Geometry keyword e.g. 'vertical_cylinder_tank', 'horizontal_cylinder_tank', 'sphere', 'flat_surface'.
        dimensions: Dict of dimensions in meters. For cylinders, provide 'diameter' and 'height' (vertical) or 'length' (horizontal).
        contents_temperature: Internal contents temperature (K).
        fluid_name_internal: Internal fluid name (default 'water').
        fluid_name_external: External fluid name (default 'air').

        headspace_height_m: Height of gas headspace above liquid (m). For digesters/tanks with gas space.
        headspace_fluid: Fluid in headspace ('air', 'biogas', etc.). Default 'air'.
        headspace_h_inner_override_w_m2k: Override inner h for headspace zones. If None, uses 5 W/m²K for stagnant gas.

        wall_layers: List of dicts for layers: {thickness: m, thermal_conductivity_k: W/m-K, material_name: str?}.
        insulation_R_value_si: If provided, adds an equivalent insulation layer where thickness = R*k using assumed_insulation_k_w_mk.
        assumed_insulation_k_w_mk: Thermal conductivity to convert R-value to an equivalent thickness.

        ambient_air_temperature: Ambient air temperature (K). If None and latitude/longitude with design_percentile provided, uses weather percentile.
        wind_speed: Ambient wind speed (m/s).
        sky_temperature: Effective sky temperature (K) for radiation.
        include_solar_gain: Whether to include solar gain; if True and incident_solar_radiation not given, pass None to reuse defaults.
        incident_solar_radiation: Incident solar irradiance on the surface (W/m^2).
        latitude, longitude, start_date, end_date, design_percentile: When provided and meteostat is available, compute ambient at percentile.
        time_resolution: 'daily' or 'hourly' for the percentile analysis.

        surface_emissivity: Outer surface emissivity for radiation.

        include_ground_contact: If True, compute ground/foundation heat loss via ground_config.
        ground_config: Dict mirroring tools.ground_heat_loss inputs.
        average_external_air_temperature: Annual average air temp (K) for ground temperature estimate if ground is included.

        solve_for: One of {'R_value'}. Additional targets will be added over time.
        target_heat_loss_w: Required when solve_for == 'R_value'. Target total heat loss (W) for the above-ground portion.
        sweep: Dict mapping parameter name to list of values for Cartesian product sweeps (e.g., {'insulation_R_value_si': [0, 2, 4]}).

    Returns:
        JSON with total and component heat losses, surface temperature, and optional ground contribution.
        Includes 'calculation_methods' and 'inputs_used' for traceability.
    """
    try:
        # Validate key numeric inputs early
        try:
            validate_geometry_dimensions(geometry, dimensions)
            require_non_negative(headspace_height_m, "headspace_height_m")
            if wind_speed is not None:
                require_non_negative(float(wind_speed), "wind_speed")
            if insulation_R_value_si is not None and float(insulation_R_value_si) < 0:
                raise ValidationError("insulation_R_value_si must be >= 0 (0 means no insulation)")
        except ValidationError as ve:
            return json.dumps({"error": str(ve)})

        # Prepare ambient
        ambient, ambient_info = _ambient_from_inputs(
            ambient_air_temperature=ambient_air_temperature,
            wind_speed=wind_speed,
            sky_temperature=sky_temperature,
            include_solar_gain=include_solar_gain,
            incident_solar_radiation=incident_solar_radiation,
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date,
            design_percentile=design_percentile,
            time_resolution=time_resolution,
        )

        # Assemble wall layers (merge explicit layers with equivalent from R, if any)
        base_layers: Optional[List[Dict[str, Any]]] = None
        if wall_layers and isinstance(wall_layers, list):
            base_layers = list(wall_layers)
        if insulation_R_value_si is not None and insulation_R_value_si > 0:
            extra = _make_wall_layers_from_R(float(insulation_R_value_si), float(assumed_insulation_k_w_mk))
            base_layers = (base_layers or []) + extra

        # If sweep requested, iterate and return sweep results
        if sweep and isinstance(sweep, dict) and len(sweep) > 0:
            base_params = {
                "geometry": geometry,
                "dimensions": dimensions,
                "contents_temperature": contents_temperature,
                "fluid_name_internal": fluid_name_internal,
                "fluid_name_external": fluid_name_external,
                "surface_emissivity": surface_emissivity,
            }
            combos = _grid_sweep(base_params, sweep)
            sweep_results: List[Dict[str, Any]] = []
            for p in combos:
                # Handle per-combo layers/R
                combo_layers = base_layers
                # If the sweep overrides insulation_R_value_si, rebuild layers
                if "insulation_R_value_si" in p:
                    combo_layers = (wall_layers or []) + _make_wall_layers_from_R(
                        float(p["insulation_R_value_si"]), float(assumed_insulation_k_w_mk)
                    )
                data = _run_surface_solver(
                    geometry=p["geometry"],
                    dimensions=p["dimensions"],
                    internal_temperature=p["contents_temperature"],
                    surface_emissivity=p["surface_emissivity"],
                    ambient=ambient,
                    fluid_name_internal=fluid_name_internal,
                    fluid_name_external=fluid_name_external,
                    wall_layers=combo_layers,
                    overall_heat_transfer_coefficient_U=None,
                )
                if "error" in data:
                    sweep_results.append({"params": p, "error": data["error"]})
                else:
                    sweep_results.append(
                        {
                            "params": p,
                            "heat_loss_w": data.get("total_heat_rate_loss_w"),
                            "surface_temp_K": data.get("estimated_outer_surface_temp_k"),
                            "details": data,
                        }
                    )
            return json.dumps(
                {
                    "mode": "sweep",
                    "results": sweep_results,
                    "ambient": ambient.__dict__,
                    "ambient_info": ambient_info,
                    "calculation_methods": {
                        "surface": "Iterative balance using convection + radiation (ht correlations via subtool)",
                    },
                }
            )

        # Solve-for mode
        if solve_for:
            sf = str(solve_for).lower()
            if sf == "r_value":
                if target_heat_loss_w is None:
                    return json.dumps({"error": "target_heat_loss_w is required when solve_for='R_value'"})
                solve_res = _solve_for_R_value(
                    target_heat_loss_W=float(target_heat_loss_w),
                    geometry=geometry,
                    dimensions=dimensions,
                    contents_temperature=contents_temperature,
                    ambient=ambient,
                    surface_emissivity=surface_emissivity,
                    fluid_name_internal=fluid_name_internal,
                    fluid_name_external=fluid_name_external,
                    base_wall_layers=base_layers or [],
                    default_k_insulation=float(assumed_insulation_k_w_mk),
                )
                return json.dumps(
                    {
                        "mode": "solve_for",
                        "solve_for": "R_value",
                        "result": solve_res,
                        "ambient": ambient.__dict__,
                        "ambient_info": ambient_info,
                        "calculation_methods": {
                            "solver": "Hybrid secant/bisection on R to match target Q",
                            "surface": "Iterative balance using convection + radiation",
                        },
                    }
                )
            else:
                return json.dumps({"error": f"solve_for='{solve_for}' not supported yet; currently supported: ['R_value']"})

        # Standard single evaluation (no sweep/no solve)
        # Check if headspace modeling is needed
        if headspace_height_m > 0 and "cylinder_tank" in geometry.lower():
            # Headspace modeling - split tank into zones
            diameter = dimensions.get("diameter", 0)
            height = dimensions.get("height", dimensions.get("length", 0))
            if headspace_height_m >= height:
                return json.dumps({"error": "headspace_height_m must be less than total height/length"})

            # Calculate zone areas
            liquid_height = max(0, height - headspace_height_m)

            # Wetted wall area (in contact with liquid)
            wetted_wall_area = math.pi * diameter * liquid_height

            # Dry wall area (in contact with gas)
            dry_wall_area = math.pi * diameter * headspace_height_m

            # Roof area (top endcap)
            roof_area = math.pi * (diameter / 2) ** 2

            # Bottom area (always wetted)
            bottom_area = math.pi * (diameter / 2) ** 2

            # Total areas for each zone
            wetted_total_area = wetted_wall_area + bottom_area
            dry_total_area = dry_wall_area + roof_area

            # Estimate gas temperature (simple weighted average for now)
            # More sophisticated: solve energy balance in gas space
            gas_temp_K = 0.7 * contents_temperature + 0.3 * ambient.T_air_K

            # Inner h for headspace (natural convection in enclosed space)
            h_inner_headspace = (
                headspace_h_inner_override_w_m2k if headspace_h_inner_override_w_m2k else 5.0
            )  # W/m²K for stagnant gas

            # Calculate wetted zone heat loss
            wetted_dims = {"diameter": diameter, "height": liquid_height}
            wetted_data = _run_surface_solver(
                geometry="vertical_cylinder_tank",
                dimensions=wetted_dims,
                internal_temperature=contents_temperature,
                surface_emissivity=surface_emissivity,
                ambient=ambient,
                fluid_name_internal=fluid_name_internal,
                fluid_name_external=fluid_name_external,
                wall_layers=base_layers,
                overall_heat_transfer_coefficient_U=None,
            )

            # Calculate dry zone heat loss (roof + dry wall)
            # Need to modify internal h for this zone
            dry_dims = {"diameter": diameter, "height": headspace_height_m}
            # Create modified wall layers with adjusted inner h
            # This is approximate - ideally we'd pass h_inner to surface solver
            dry_data = _run_surface_solver(
                geometry="vertical_cylinder_tank",
                dimensions=dry_dims,
                internal_temperature=gas_temp_K,  # Use gas temperature
                surface_emissivity=surface_emissivity,
                ambient=ambient,
                fluid_name_internal=headspace_fluid,  # Gas instead of liquid
                fluid_name_external=fluid_name_external,
                wall_layers=base_layers,
                overall_heat_transfer_coefficient_U=None,
                h_inner_override_w_m2k=h_inner_headspace,
            )

            # Combine results
            if "error" in wetted_data:
                return json.dumps({"error": f"Wetted zone: {wetted_data['error']}"})
            if "error" in dry_data:
                return json.dumps({"error": f"Dry zone: {dry_data['error']}"})

            # Extract zone totals
            wetted_q = wetted_data.get("total_heat_rate_loss_w", 0.0)
            dry_q = dry_data.get("total_heat_rate_loss_w", 0.0)
            wetted_conv = wetted_data.get("convective_heat_rate_w", 0.0)
            dry_conv = dry_data.get("convective_heat_rate_w", 0.0)
            wetted_rad = wetted_data.get("radiative_heat_rate_w", 0.0)
            dry_rad = dry_data.get("radiative_heat_rate_w", 0.0)

            # Remove the interior gas/liquid interface contribution (not an external surface)
            # With updated surface area logic, the interface disc is only present in the wetted zone model.
            interface_area = math.pi * (diameter / 2) ** 2
            wetted_area_model = max(wetted_data.get("outer_surface_area_m2", wetted_total_area), 1e-9)
            qpp_w_total = wetted_q / wetted_area_model
            qpp_w_conv = wetted_conv / wetted_area_model
            qpp_w_rad = wetted_rad / wetted_area_model

            total_surface_q = wetted_q + dry_q - interface_area * qpp_w_total
            total_conv_q = wetted_conv + dry_conv - interface_area * qpp_w_conv
            total_rad_q = wetted_rad + dry_rad - interface_area * qpp_w_rad

            # Air-exposed areas only (exclude bottom)
            air_exposed_area = wetted_wall_area + dry_wall_area + roof_area
            surface_data = {
                "total_heat_rate_loss_w": total_surface_q,
                "convective_heat_rate_w": total_conv_q,
                "radiative_heat_rate_w": total_rad_q,
                "solar_gain_rate_w": wetted_data.get("solar_gain_rate_w", 0) + dry_data.get("solar_gain_rate_w", 0),
                "estimated_outer_surface_temp_k": (
                    (wetted_data.get("estimated_outer_surface_temp_k", 0) * wetted_wall_area)
                    + (dry_data.get("estimated_outer_surface_temp_k", 0) * (dry_wall_area + roof_area))
                )
                / max(air_exposed_area, 1e-9),
                "estimated_outer_surface_temp_c": None,  # Will calculate below
                "outer_surface_area_m2": air_exposed_area,
                "external_convection_coefficient_w_m2k": wetted_data.get("external_convection_coefficient_w_m2k"),
                "internal_plus_wall_resistance_k_w": wetted_data.get("internal_plus_wall_resistance_k_w"),
                "headspace_info": {
                    "headspace_height_m": headspace_height_m,
                    "liquid_height_m": liquid_height,
                    "gas_temp_estimate_k": gas_temp_K,
                    "gas_temp_estimate_c": gas_temp_K - 273.15,
                    "wetted_area_m2": wetted_wall_area,  # air-exposed wetted wall only
                    "dry_area_m2": dry_wall_area + roof_area,
                    "wetted_heat_loss_w": wetted_q,
                    "dry_heat_loss_w": dry_q,
                    "h_inner_headspace_w_m2k": h_inner_headspace,
                },
            }
            surface_data["estimated_outer_surface_temp_c"] = surface_data["estimated_outer_surface_temp_k"] - 273.15
        else:
            # Standard calculation without headspace
            surface_data = _run_surface_solver(
                geometry=geometry,
                dimensions=dimensions,
                internal_temperature=contents_temperature,
                surface_emissivity=surface_emissivity,
                ambient=ambient,
                fluid_name_internal=fluid_name_internal,
                fluid_name_external=fluid_name_external,
                wall_layers=base_layers,
                overall_heat_transfer_coefficient_U=None,
            )
            if "error" in surface_data:
                return json.dumps({"error": surface_data["error"]})

        # Ground/foundation coupling
        geometry_l = (geometry or "").lower()
        is_vertical_tank = "vertical" in geometry_l and "tank" in geometry_l
        effective_include_ground = include_ground_contact or is_vertical_tank

        # Build auto ground config for bottom contact if needed
        effective_ground_config: Optional[Dict[str, Any]] = None
        if effective_include_ground:
            effective_ground_config = dict(ground_config) if isinstance(ground_config, dict) else {}
            # If no dimensions provided, derive from circular bottom area -> square with same area
            g_dims = effective_ground_config.get("dimensions") or {}
            if not g_dims:
                d = dimensions.get("diameter")
                if d is not None and float(d) > 0:
                    r = float(d) / 2.0
                    bottom_area = math.pi * r * r
                    side = math.sqrt(bottom_area)
                    g_dims = {"length": side, "width": side}
                    effective_ground_config["dimensions"] = g_dims
            if not effective_ground_config.get("structure_type"):
                effective_ground_config["structure_type"] = "slab_on_grade"
            if effective_ground_config.get("soil_conductivity") is None:
                effective_ground_config["soil_conductivity"] = 0.8  # W/m·K fallback for soil
            if effective_ground_config.get("insulation_R_value_si") is None and insulation_R_value_si is not None:
                effective_ground_config["insulation_R_value_si"] = float(insulation_R_value_si)

        ground_info = _add_ground_loss_if_requested(
            include_ground=effective_include_ground,
            ground_config=effective_ground_config,
            contents_temperature=contents_temperature,
            average_external_air_temperature=average_external_air_temperature,
        )

        total_Q = surface_data.get("total_heat_rate_loss_w", 0.0) + ground_info.get("ground_heat_loss_watts", 0.0)

        # Optional percentile analysis (90/95/99 etc.) when site/time given
        percentile_results = None
        if percentiles and latitude is not None and longitude is not None and start_date and end_date:
            pr_list = []
            try:
                # Fetch weather data once for all percentiles
                weather_service = get_weather_service()
                start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
                end_dt = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date

                weather_data = weather_service.get_design_conditions(
                    lat=float(latitude),
                    lon=float(longitude),
                    start_date=start_dt,
                    end_date=end_dt,
                    percentiles=percentiles,
                    time_resolution=time_resolution,
                )

                if weather_data and "design_conditions" in weather_data:
                    for p in percentiles:
                        percentile_key = f"cold_{int(p*100)}th"
                        if percentile_key not in weather_data["design_conditions"]:
                            continue

                        design_cond = weather_data["design_conditions"][percentile_key]
                        amb_p = AmbientSpec(
                            T_air_K=design_cond["temp_k"], wind_m_s=design_cond.get("wind_m_s", 2.0), T_sky_K=sky_temperature
                        )

                        sd = _run_surface_solver(
                            geometry=geometry,
                            dimensions=dimensions,
                            internal_temperature=contents_temperature,
                            surface_emissivity=surface_emissivity,
                            ambient=amb_p,
                            fluid_name_internal=fluid_name_internal,
                            fluid_name_external=fluid_name_external,
                            wall_layers=base_layers,
                            overall_heat_transfer_coefficient_U=None,
                        )
                        if "error" in sd:
                            pr_list.append({"percentile": p, "error": sd["error"]})
                        else:
                            pr_list.append(
                                {
                                    "percentile": p,
                                    "ambient": {"T_air_K": amb_p.T_air_K, "wind_m_s": amb_p.wind_m_s},
                                    "heat_loss_w": sd.get("total_heat_rate_loss_w"),
                                    "surface_temp_K": sd.get("estimated_outer_surface_temp_k"),
                                }
                            )
            except Exception as e:
                logger.warning(f"Percentile analysis failed: {e}")

            percentile_results = pr_list if pr_list else None

        # Extract transparency fields from surface_data
        sky_temp_k = surface_data.get("sky_temperature_k", ambient.T_sky_K)
        radiation_model = surface_data.get("radiation_model", {})
        warnings = surface_data.get("warnings", [])

        result = {
            "total_heat_loss_w": total_Q,
            "sign_convention": "Positive = heat loss to ambient; Negative = heat gain from ambient",
            "above_ground_surface_loss_w": surface_data.get("total_heat_rate_loss_w", 0.0),
            "convective_heat_rate_w": surface_data.get("convective_heat_rate_w", None),
            "radiative_heat_rate_w": surface_data.get("radiative_heat_rate_w", None),
            "solar_gain_rate_w": surface_data.get("solar_gain_rate_w", 0.0),
            "estimated_outer_surface_temp_k": surface_data.get("estimated_outer_surface_temp_k"),
            "estimated_outer_surface_temp_c": surface_data.get("estimated_outer_surface_temp_c"),
            "outer_surface_area_m2": surface_data.get("outer_surface_area_m2"),
            "external_convection_coefficient_w_m2k": surface_data.get("external_convection_coefficient_w_m2k"),
            "internal_plus_wall_resistance_k_w": surface_data.get("internal_plus_wall_resistance_k_w"),
            "ground_heat_loss_w": ground_info.get("ground_heat_loss_watts", 0.0),
            "ground_details": ground_info.get("ground_details"),
            "ambient": ambient.__dict__,
            "ambient_info": ambient_info,
            "weather_data": {
                "source": ambient_info.get("ambient_source", "unknown"),
                "meta": ambient_info.get("weather_meta", {}),
                "dew_point_k": ambient_info.get("dew_point_k"),
                "sky_temperature_k": ambient_info.get("sky_temperature_k", sky_temp_k),
            },
            "radiation_model": radiation_model,
            "calculation_methods": {
                "surface": "Iterative balance using convection + radiation (Newtown-like with damping)",
                "external_convection": "ht correlations via tools.convection_coefficient",
                "overall_U": "tools.overall_heat_transfer when layers provided",
                "ground": "Equivalent U-value method via tools.ground_heat_loss",
            },
            "inputs_used": {
                "geometry": geometry,
                "dimensions": dimensions,
                "fluid_name_internal": fluid_name_internal,
                "fluid_name_external": fluid_name_external,
                "wall_layers_count": len(base_layers) if base_layers else 0,
                "insulation_R_value_si": insulation_R_value_si,
                "assumed_insulation_k_w_mk": assumed_insulation_k_w_mk,
                "include_solar_gain": include_solar_gain,
                "include_ground_contact": effective_include_ground,
                "headspace_height_m": headspace_height_m,
                "headspace_fluid": headspace_fluid,
            },
            "headspace_info": surface_data.get("headspace_info"),
            "percentile_analysis": percentile_results,
            "warnings": warnings,
        }

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in tank_heat_loss: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
