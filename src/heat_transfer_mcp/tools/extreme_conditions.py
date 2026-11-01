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

from heat_transfer_mcp.errors import CalculationError
from heat_transfer_mcp.weather.geocode import geocode_location
from heat_transfer_mcp.utils.import_helpers import METEOSTAT_AVAILABLE, PANDAS_AVAILABLE
from heat_transfer_mcp.utils.validation import ValidationError, ensure_list_nonempty, validate_date_range, validate_lat_lon

logger = logging.getLogger("heat-transfer-mcp.extreme_conditions")

#: Below this fraction of hours carrying an observation, a design dry-bulb derived from
#: the record is flagged rather than presented as usable.
MIN_COVERAGE_FOR_DESIGN = 0.60


def _record_years(frame: Any) -> float:
    """Length of the record in years, taken from its timestamps.

    Row count is not a proxy for elapsed time — a frame may hold several rows per hour, or
    have gaps — and an ASHRAE exceedance count is defined per year of record.
    """
    try:
        index = frame.index
        if getattr(index, "nlevels", 1) > 1:
            names = list(index.names or [])
            index = index.get_level_values(names.index("time") if "time" in names else -1)
        span = index.max() - index.min()
        years = span.total_seconds() / (365.25 * 24 * 3600)
        return max(years, 1.0 / 365.25)
    except Exception:
        return max(1.0, len(frame) / 8766.0)


def _provenance(frame: Any) -> Dict[str, Any]:
    """Station, sources and time span behind the statistics, for auditability."""
    attrs = getattr(frame, "attrs", {}) or {}
    prov: Dict[str, Any] = {
        "stations": attrs.get("stations"),
        "sources_available": attrs.get("sources_available"),
        "sources_used": attrs.get("sources_used"),
        "forecasts_excluded": attrs.get("forecasts_excluded"),
    }
    try:
        index = frame.index
        prov["first_observation"] = str(index.min())
        prov["last_observation"] = str(index.max())
        tz = getattr(index, "tz", None)
        prov["timezone"] = str(tz) if tz is not None else "naive (Meteostat returns UTC)"
    except Exception:
        pass
    return prov


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
    design_basis: str = "cold",
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
        design_basis: 'cold' (default, winterisation / freeze protection) or 'hot'
            (summer design dry-bulb for air-cooled equipment). The two bases answer
            opposite questions and are reported under separate keys.

    Note on the hot basis: ASHRAE 0.4/1/2% annual design dry-bulb values are defined by
    *hourly* exceedance — 35, 88 and 175 hours per year respectively — not by a percentile
    of daily maxima. Requesting the hot basis at daily resolution therefore returns
    percentiles of ``tmax``, which is a related but different statistic, and the response
    labels it as such. Pass ``time_resolution='hourly'`` for ASHRAE-comparable numbers.

    Returns:
        JSON with percentile statistics and a proposed design day.
    """
    if percentiles is None:
        percentiles = [0.90, 0.95, 0.99]

    if (latitude is None or longitude is None) and location_name:
        try:
            geocoded = geocode_location(location_name)
            latitude, longitude = geocoded.latitude, geocoded.longitude
        except CalculationError as exc:
            payload = exc.to_dict()
            payload["error"] = payload["message"]
            return json.dumps(payload)

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
        return json.dumps(
            {
                "error": "Meteostat and pandas are required for extreme conditions analysis",
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

    try:
        import pandas as pd
        from heat_transfer_mcp.weather.provider import _fetch_meteostat_frame

        # Parse date strings to datetime objects
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date

        # Routed through the shared provider so this tool works against both the
        # Meteostat 1.x and 2.x APIs rather than importing v1-only names directly.
        ds = _fetch_meteostat_frame(
            latitude=float(latitude),
            longitude=float(longitude),
            elevation=None,
            start=start_dt,
            end=end_dt,
            time_resolution=time_resolution,
        )

        if ds is None or ds.empty:
            return json.dumps({"result": [], "message": "No weather data found"})

        out: Dict[str, Any] = {
            "location": {"latitude": latitude, "longitude": longitude, "name": location_name},
            "data_summary": {
                "source": "meteostat",
                "time_resolution": time_resolution,
                "date_range": {"start": start_date, "end": end_date},
                "data_points": len(ds),
                "available_metrics": [
                    col for col in ["tmin", "tavg", "tmax", "wspd", "rhum", "tdew", "tsun"] if col in ds.columns
                ],
            },
            "percentiles": {},
        }

        hot_basis = str(design_basis).lower() == "hot"

        # The cold basis takes daily minima and the hot basis daily maxima; at hourly
        # resolution both work from the instantaneous `temp` series, which is what ASHRAE
        # exceedance hours are defined against.
        if hot_basis:
            preference = ["temp", "tmax", "tavg"]
        else:
            preference = ["temp", "tmin", "tavg"] if time_resolution.lower() == "hourly" else ["tmin", "tavg"]
        temp_basis = next((c for c in preference if c in ds.columns), None)
        if temp_basis is None:
            return json.dumps({"error": "No suitable temperature column in Meteostat dataset"})
        # Kept aligned with `ds` — the concurrent-extremes block masks the frame with this
        # series, and dropping rows here would make that mask unalignable. `quantile` and
        # `count` skip nulls on their own.
        temp_series = ds[temp_basis]
        if temp_series.count() == 0:
            return json.dumps({"error": f"Temperature column '{temp_basis}' contains no data"})

        temps = {}
        if hot_basis:
            # Hot design wants HIGH percentiles: p99 means only 1% of observations are warmer.
            for p in percentiles:
                q = float(p) if p > 0.5 else 1.0 - float(p)
                label = f"p{int(round(q * 100))}_hot"
                val = temp_series.quantile(q)
                temps[label] = float(val) if pd.notnull(val) else None
            # ASHRAE annual design dry-bulb, by hourly exceedance.
            ashrae = {}
            if time_resolution.lower() == "hourly":
                observations = int(temp_series.count())
                # The record length must come from the timestamps, not the row count. A
                # Meteostat frame can carry several rows per hour, so len(frame)/8760
                # invents years that were never observed and skews every exceedance value.
                years = _record_years(ds)
                for label, hours in (("0.4%", 35.0), ("1%", 88.0), ("2%", 175.0)):
                    # Exceedance is defined per average year, so scale to the record length
                    # and express it as a fraction of the observations actually present.
                    frac_exceeded = (hours * years) / observations if observations else 1.0
                    if 0.0 < frac_exceeded < 1.0:
                        val = temp_series.quantile(1.0 - frac_exceeded)
                        ashrae[label] = float(val) if pd.notnull(val) else None
                expected_hours = years * 8766.0
                coverage = observations / expected_hours if expected_hours else None
                block: Dict[str, Any] = {
                    "values": ashrae,
                    "definition": "annual exceedance of 35 / 88 / 175 hours",
                    "record_years": round(years, 2),
                    "observations": observations,
                    "coverage_fraction": round(coverage, 3) if coverage else None,
                    "provenance": _provenance(ds),
                }
                # A sparse record biases the upper tail: hours missing at random are fine,
                # but stations that report intermittently often miss the same part of the
                # day, and that is exactly where the design value lives.
                if coverage is not None and coverage < MIN_COVERAGE_FOR_DESIGN:
                    block["caution"] = (
                        f"Only {coverage:.1%} of the hours in this record carry a "
                        "temperature observation. That is too sparse to treat the values "
                        "above as a design dry-bulb on their own — corroborate against a "
                        "published climatic design table or a denser nearby station "
                        "before using them."
                    )
                out["ashrae_design_dry_bulb_c"] = block
            else:
                out["ashrae_design_dry_bulb_c"] = {
                    "values": None,
                    "note": (
                        "ASHRAE design dry-bulb is defined by hourly exceedance; at daily "
                        "resolution only percentiles of daily maxima are available. Re-run "
                        "with time_resolution='hourly' for ASHRAE-comparable values."
                    ),
                }
        else:
            # For cold design, we want LOW temperature percentiles
            # p99 cold design means only 1% of days are colder
            for p in percentiles:
                # Convert to cold percentile: p=0.99 -> use 0.01 quantile for cold design
                cold_quantile = 1.0 - p if p > 0.5 else p
                label = f"p{int(p*100)}_cold"
                temps[label] = (
                    float(temp_series.quantile(cold_quantile)) if pd.notnull(temp_series.quantile(cold_quantile)) else None
                )

        out["percentiles"]["temperature_c"] = {
            "basis": temp_basis,
            "design_basis": "hot" if hot_basis else "cold",
            "values": temps,
            "note": (
                "p99_hot means only 1% of observations are warmer"
                if hot_basis
                else "p99_cold means 99% of days are warmer"
            ),
        }

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
                if hot_basis:
                    # Hot-still air is the adverse case for air-cooled equipment, so the
                    # low-wind tail is what the design day needs; the requested percentile
                    # list is a high-side one and does not produce it.
                    low = wspd_ms.quantile(0.05)
                    wind_stats["p05"] = float(low) if pd.notnull(low) else None
            out["percentiles"]["wind_m_s"] = wind_stats

        # Concurrent extremes: the wind that accompanies the temperature extreme. Which
        # tail of the wind distribution is adverse depends on the basis — cold design is
        # worst in high wind (convective loss), hot design worst in still air (an
        # air-cooled bundle starved of natural draught and prone to recirculation).
        concurrent = None
        if include_wind and wind_stats is not None:
            p = max(percentiles) if len(percentiles) else 0.99
            if hot_basis:
                quantile = float(p) if p > 0.5 else 1.0 - float(p)
                temp_threshold = temp_series.quantile(quantile)
                subset = ds[temp_series >= temp_threshold]
                wind_q, wind_key = 0.05, "wind_p05_m_s"
                note_tail = f"hottest {100*(1-quantile):.0f}%"
                note_wind = "5th percentile (still-air case)"
            else:
                quantile = 1.0 - p if p > 0.5 else p
                temp_threshold = temp_series.quantile(quantile)
                subset = ds[temp_series <= temp_threshold]
                wind_q, wind_key = 0.95, "wind_p95_m_s"
                note_tail = f"coldest {100*quantile:.0f}%"
                note_wind = "95th percentile"
            if not subset.empty and "wspd" in subset.columns:
                w_ms = subset["wspd"] / 3.6
                concurrent = {
                    "temperature_threshold_c": float(temp_threshold) if pd.notnull(temp_threshold) else None,
                    wind_key: float(w_ms.quantile(wind_q)) if pd.notnull(w_ms.quantile(wind_q)) else None,
                    "count": int(subset.shape[0]),
                    "note": f"Wind speed at {note_wind} during the {note_tail} of observations",
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

        if hot_basis:
            ashrae_vals = (out.get("ashrae_design_dry_bulb_c") or {}).get("values") or {}
            design = {
                "temperature_design_c": (
                    ashrae_vals.get("0.4%")
                    or temps.get("p99_hot")
                    or temps.get("p95_hot")
                ),
                # Air-cooled equipment is worst-served by hot, still air, so the low wind
                # percentile is the adverse case here — the opposite of the cold basis.
                "wind_design_m_s": (wind_stats or {}).get("p05") if wind_stats else None,
                "basis": (
                    "ASHRAE 0.4% annual hourly exceedance"
                    if ashrae_vals.get("0.4%") is not None
                    else "high percentile of daily maxima (not ASHRAE-equivalent)"
                ),
            }
        else:
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
