"""
Omnibus tool: extreme_conditions

Historical weather extremes and design-day selection using Meteostat when
available. Computes temperature and wind percentiles (e.g., 90th, 95th, 99th),
and concurrent extremes for conservative design (cold + wind). Optionally
returns basic solar stats if requested.

Consolidates ambient_conditions with statistical analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.import_helpers import METEOSTAT_AVAILABLE, PANDAS_AVAILABLE
from utils.weather_service import get_weather_service
from utils.validation import (
    ValidationError,
    validate_lat_lon,
    validate_date_range,
    ensure_list_nonempty,
)

logger = logging.getLogger("heat-transfer-mcp.extreme_conditions")


def extreme_conditions(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_resolution: str = "daily",
    percentiles: Optional[List[float]] = None,
    include_wind: bool = True,
    include_solar: bool = False,
) -> str:
    """Compute historical weather extremes for a site.

    Args:
        latitude, longitude: Coordinates of the site (decimal degrees). Required unless a known location_name is provided.
        location_name: Optional city name shortcut (limited defaults).
        start_date, end_date: Date range in YYYY-MM-DD.
        time_resolution: 'daily' or 'hourly'.
        percentiles: List like [0.90, 0.95, 0.99]. If None, defaults to [0.90, 0.95, 0.99].
        include_wind: If True, compute wind percentiles and concurrent extremes.
        include_solar: If True, provide simple solar statistics (if available).

    Returns:
        JSON with percentile statistics and a proposed design day.
    """
    if percentiles is None:
        percentiles = [0.90, 0.95, 0.99]

    # Minimal default geocoding
    if (latitude is None or longitude is None) and location_name:
        name = location_name.lower()
        if "london" in name:
            latitude, longitude = 51.5072, -0.1276
        elif "new york" in name:
            latitude, longitude = 40.7128, -74.0060
        elif "los angeles" in name:
            latitude, longitude = 34.0522, -118.2437
        elif "tokyo" in name:
            latitude, longitude = 35.6762, 139.6503
        elif "sydney" in name:
            latitude, longitude = -33.8688, 151.2093

    # Validate coordinates and dates
    try:
        validate_lat_lon(latitude, longitude)
        validate_date_range(start_date, end_date)
        ensure_list_nonempty("percentiles", percentiles)
        # Validate percentile values are within (0,1)
        for p in percentiles:
            if not (0.0 < float(p) < 1.0):
                raise ValidationError(f"percentile values must be between 0 and 1 (exclusive); got {p}")
        if time_resolution.lower() not in {"daily", "hourly"}:
            raise ValidationError("time_resolution must be 'daily' or 'hourly'")
    except ValidationError as ve:
        return json.dumps({"error": str(ve)})

    if not METEOSTAT_AVAILABLE or not PANDAS_AVAILABLE:
        return json.dumps({
            "error": "Meteostat and pandas are required for extreme conditions analysis",
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
        })

    try:
        from meteostat import Point, Daily, Hourly
        import pandas as pd

        # Parse date strings to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date

        loc = Point(float(latitude), float(longitude))
        if time_resolution.lower() == "hourly":
            ds = Hourly(loc, start_dt, end_dt).fetch()
        else:
            ds = Daily(loc, start_dt, end_dt).fetch()

        if ds is None or ds.empty:
            return json.dumps({"result": [], "message": "No weather data found"})

        out: Dict[str, Any] = {
            "location": {"latitude": latitude, "longitude": longitude, "name": location_name},
            "data_summary": {
                "source": "meteostat",
                "time_resolution": time_resolution,
                "date_range": {"start": start_date, "end": end_date},
                "data_points": len(ds),
                "available_metrics": [col for col in ['tmin', 'tavg', 'tmax', 'wspd', 'rhum', 'tdew', 'tsun'] if col in ds.columns],
            },
            "percentiles": {},
        }

        # Temperature percentiles (prefer tmin for conservative cold design; fallback tavg)
        if "tmin" in ds.columns:
            temp_series = ds["tmin"]
            temp_basis = "tmin"
        elif "tavg" in ds.columns:
            temp_series = ds["tavg"]
            temp_basis = "tavg"
        else:
            return json.dumps({"error": "No suitable temperature column in Meteostat dataset"})

        # For cold design, we want LOW temperature percentiles
        # p99 cold design means only 1% of days are colder
        temps = {}
        for p in percentiles:
            # Convert to cold percentile: p=0.99 -> use 0.01 quantile for cold design
            cold_quantile = 1.0 - p if p > 0.5 else p
            label = f"p{int(p*100)}_cold"
            temps[label] = float(temp_series.quantile(cold_quantile)) if pd.notnull(temp_series.quantile(cold_quantile)) else None
        
        out["percentiles"]["temperature_c"] = {"basis": temp_basis, "values": temps, "note": "p99_cold means 99% of days are warmer"}

        # Wind percentiles
        wind_stats = None
        if include_wind:
            if "wspd" in ds.columns:  # km/h
                wspd = ds["wspd"].copy()
                wspd_ms = wspd / 3.6
                wind_stats = {
                    f"p{int(p*100)}": float(wspd_ms.quantile(p)) if pd.notnull(wspd_ms.quantile(p)) else None
                    for p in percentiles
                }
            out["percentiles"]["wind_m_s"] = wind_stats

        # Concurrent extremes (cold + wind): find rows with low temp and high wind
        concurrent = None
        if include_wind and wind_stats is not None:
            # For cold design, identify the coldest percentile
            p = max(percentiles) if len(percentiles) else 0.99
            # Convert to cold percentile: p=0.99 -> use 0.01 quantile for cold
            cold_quantile = 1.0 - p if p > 0.5 else p
            temp_threshold = temp_series.quantile(cold_quantile)
            subset = ds[temp_series <= temp_threshold]  # Days colder than threshold
            if not subset.empty and "wspd" in subset.columns:
                w_ms = subset["wspd"] / 3.6
                concurrent = {
                    "temperature_threshold_c": float(temp_threshold) if pd.notnull(temp_threshold) else None,
                    "wind_p95_m_s": float(w_ms.quantile(0.95)) if pd.notnull(w_ms.quantile(0.95)) else None,
                    "count": int(subset.shape[0]),
                    "note": f"Wind speed at 95th percentile during coldest {100*(1-cold_quantile):.0f}% of days"
                }
        out["concurrent_extremes"] = concurrent

        # Solar summary if requested (coarse)
        if include_solar:
            sol = None
            for cand in ["tsun", "asum", "srad"]:
                if cand in ds.columns:
                    ser = ds[cand]
                    sol = {
                        "column": cand,
                        "p50": float(ser.quantile(0.5)) if pd.notnull(ser.quantile(0.5)) else None,
                        "p95": float(ser.quantile(0.95)) if pd.notnull(ser.quantile(0.95)) else None,
                    }
                    break
            out["solar_summary"] = sol

        # Propose a design day based on cold percentile and typical wind
        design = {
            "temperature_design_c": temps.get("p99_cold") if "p99_cold" in temps else temps.get("p95_cold"),
            "wind_design_m_s": (wind_stats or {}).get("p95") if wind_stats else None,
            "basis": "cold percentile with high wind",
        }
        out["design_day"] = design

        return json.dumps(out)
    except Exception as e:
        logger.error(f"extreme_conditions failed: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
