"""Weather provider abstraction for ambient-condition tools."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from heat_transfer_mcp.utils.import_helpers import PANDAS_AVAILABLE
from heat_transfer_mcp.utils.validation import ValidationError, validate_lat_lon


def fetch_weather_data(
    *,
    latitude: float,
    longitude: float,
    elevation: Optional[float],
    location_name: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    time_resolution: str,
    parameters: List[str],
) -> Dict[str, Any]:
    """Fetch weather observations using the configured primary provider."""
    try:
        validate_lat_lon(latitude, longitude)
    except ValidationError as exc:
        return {"code": "invalid_location", "error": str(exc)}

    if not start_date or not end_date:
        return {"code": "invalid_date_range", "error": "Both 'start_date' and 'end_date' are required in YYYY-MM-DD format."}
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        return {"code": "invalid_date_range", "error": f"Invalid date format. Please use YYYY-MM-DD. Error: {exc}"}
    if start > end:
        return {"code": "invalid_date_range", "error": "'start_date' must be on or before 'end_date'."}

    if time_resolution.lower() not in {"daily", "hourly"}:
        return {"code": "invalid_time_resolution", "error": "time_resolution must be 'daily' or 'hourly'."}

    if not PANDAS_AVAILABLE:
        return _provider_unavailable(latitude, longitude, elevation, location_name, "pandas package is not installed")

    try:
        frame = _fetch_meteostat_frame(
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            start=start,
            end=end,
            time_resolution=time_resolution,
        )
    except Exception as exc:
        if _is_import_error(exc):
            return _provider_unavailable(latitude, longitude, elevation, location_name, str(exc))
        return {
            "code": "provider_error",
            "error": f"Error fetching weather data from meteostat: {exc}",
            "location": _location_payload(latitude, longitude, elevation, location_name),
            "provider": "meteostat",
        }

    return _format_weather_frame(
        frame=frame,
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        location_name=location_name,
        parameters=parameters,
        time_resolution=time_resolution,
    )


def _fetch_meteostat_frame(
    *,
    latitude: float,
    longitude: float,
    elevation: Optional[float],
    start: datetime,
    end: datetime,
    time_resolution: str,
) -> Any:
    try:
        return _fetch_meteostat_daily_hourly_api(latitude, longitude, elevation, start, end, time_resolution)
    except ImportError:
        return _fetch_meteostat_v2(latitude, longitude, elevation, start, end, time_resolution)


def _fetch_meteostat_daily_hourly_api(
    latitude: float,
    longitude: float,
    elevation: Optional[float],
    start: datetime,
    end: datetime,
    time_resolution: str,
) -> Any:
    from meteostat import Daily, Hourly, Point

    location = Point(float(latitude), float(longitude), elevation) if elevation is not None else Point(float(latitude), float(longitude))
    data = Hourly(location, start, end) if time_resolution.lower() == "hourly" else Daily(location, start, end)
    return data.fetch()


def _fetch_meteostat_v2(
    latitude: float,
    longitude: float,
    elevation: Optional[float],
    start: datetime,
    end: datetime,
    time_resolution: str,
) -> Any:
    _configure_meteostat_v2_cache()
    from meteostat import Point, stations
    from meteostat.api.daily import daily
    from meteostat.api.hourly import hourly

    point = Point(float(latitude), float(longitude), int(elevation) if elevation is not None else None)
    fetch = hourly if time_resolution.lower() == "hourly" else daily

    # Walk outward through the nearest stations rather than committing to the closest one.
    # Proximity does not imply data: a station can be listed as nearby yet hold nothing for
    # the requested period, and taking only `head(1)` reports "no weather data" for sites
    # that have a perfectly good record a few kilometres further out.
    nearby = stations.nearby(point, limit=_NEARBY_STATION_ATTEMPTS)
    if nearby is not None and not nearby.empty:
        for station_id in list(nearby.index[:_NEARBY_STATION_ATTEMPTS]):
            try:
                frame = _normalize_v2_columns(
                    getattr(fetch(nearby.loc[[station_id]], start, end), "_df", None)
                )
            except Exception:  # a bad station must not end the search
                continue
            if _frame_has_observations(frame):
                return collapse_v2_multiindex(frame)

    return collapse_v2_multiindex(
        _normalize_v2_columns(getattr(fetch(point, start, end), "_df", None))
    )


#: How many successive nearby stations to try before giving up.
_NEARBY_STATION_ATTEMPTS = 5


def _frame_has_observations(frame: Any) -> bool:
    """Whether a fetched frame carries any usable measurements.

    An empty frame and a frame of all-null columns are equally useless downstream, and the
    v2 API returns both.
    """
    if frame is None or not hasattr(frame, "empty") or frame.empty:
        return False
    for column in ("temp", "tavg", "tmin", "tmax"):
        if column in frame.columns and frame[column].count() > 0:
            return True
    return False


def _normalize_v2_columns(frame: Any) -> Any:
    """Coerce Meteostat 2 column labels to plain strings.

    The v2 frame labels some columns with ``Parameter`` enum members rather than strings,
    so downstream ``"tmax" in frame.columns`` tests miss them and the affected series are
    silently dropped from every result.
    """
    if frame is None or not hasattr(frame, "columns"):
        return frame
    renames = {
        col: col.value
        for col in frame.columns
        if not isinstance(col, str) and hasattr(col, "value")
    }
    return frame.rename(columns=renames) if renames else frame


#: Provider precedence when the same hour is reported by more than one source. Measured
#: observations outrank model output; a forecast has no place in a design statistic.
_SOURCE_PRECEDENCE = ("synop", "metar", "isd_lite", "model", "metno_forecast", "dwd_forecast")

#: Sources that are forecasts rather than observations.
_FORECAST_SOURCES = frozenset({"metno_forecast", "dwd_forecast", "model"})


def collapse_v2_multiindex(frame: Any, *, exclude_forecasts: bool = True) -> Any:
    """Flatten a Meteostat 2 ``[station, time, source]`` frame to one row per timestamp.

    The v2 API returns every source that reported a given hour stacked on top of each
    other, so a three-year request can come back with far more rows than there are hours.
    Taking percentiles over that frame double-counts any hour several providers covered —
    weighting the distribution by reporting density rather than by time — and, worse,
    silently mixes ``metno_forecast`` model output into what is meant to be a design
    statistic derived from observations.

    Rows are therefore deduplicated to one per timestamp, preferring measured sources in
    :data:`_SOURCE_PRECEDENCE` order, and forecast sources are dropped entirely unless the
    caller asks otherwise. The station and the sources actually used are recorded on
    ``frame.attrs`` so the response can carry its provenance.
    """
    if frame is None or not hasattr(frame, "index") or frame.empty:
        return frame
    names = list(frame.index.names or [])
    if frame.index.nlevels < 2 or "time" not in names:
        return frame

    import pandas as pd

    flat = frame.reset_index()
    stations = sorted(map(str, flat["station"].unique())) if "station" in flat else []

    if "source" in flat.columns:
        available = [str(s) for s in flat["source"].unique()]
        if exclude_forecasts:
            observed = flat[~flat["source"].astype(str).isin(_FORECAST_SOURCES)]
            # Only drop forecasts if measured data actually remains.
            if not observed.empty:
                flat = observed
        rank = {name: i for i, name in enumerate(_SOURCE_PRECEDENCE)}
        flat = flat.assign(
            _rank=flat["source"].astype(str).map(lambda s: rank.get(s, len(rank)))
        ).sort_values(["time", "_rank"])
        flat = flat.drop_duplicates(subset=["time"], keep="first").drop(columns=["_rank"])
        used = sorted(str(s) for s in flat["source"].unique())
    else:
        available, used = [], []
        flat = flat.drop_duplicates(subset=["time"], keep="first")

    out = flat.set_index(pd.DatetimeIndex(flat["time"], name="time")).drop(
        columns=[c for c in ("time", "station", "source") if c in flat.columns]
    )
    out = out.sort_index()
    out.attrs["stations"] = stations
    out.attrs["sources_available"] = sorted(available)
    out.attrs["sources_used"] = used
    out.attrs["forecasts_excluded"] = bool(exclude_forecasts)
    return out


def _configure_meteostat_v2_cache() -> None:
    from meteostat.api.config import config

    base = Path(os.environ.get("HEAT_TRANSFER_WEATHER_CACHE_DIR", Path(tempfile.gettempdir()) / "heat-transfer-mcp" / "meteostat"))
    config.cache_directory = str(base / "cache")
    config.stations_db_file = str(base / "stations.db")


def _format_weather_frame(
    *,
    frame: Any,
    latitude: float,
    longitude: float,
    elevation: Optional[float],
    location_name: Optional[str],
    parameters: List[str],
    time_resolution: str,
) -> Dict[str, Any]:
    import pandas as pd

    if frame is None or frame.empty:
        return {
            "result": [],
            "message": "No weather data found for the specified location/date range.",
            "location": _location_payload(latitude, longitude, elevation, location_name),
            "parameters": parameters,
            "time_resolution": time_resolution,
            "count": 0,
            "provider": "meteostat",
        }

    data_processed = frame.reset_index()
    data_processed = data_processed.rename(columns={"temp": "tavg", "dwpt": "tdew"})
    if "time" in data_processed.columns:
        data_processed["time"] = pd.to_datetime(data_processed["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    data_processed = data_processed.where(pd.notnull(data_processed), None)

    requested = [column for column in ["station", "time", "source"] if column in data_processed.columns]
    requested.extend(param for param in parameters if param in data_processed.columns and param not in requested)
    if requested:
        data_processed = data_processed[requested]
    result_list = data_processed.to_dict(orient="records")

    return {
        "result": result_list,
        "location": _location_payload(latitude, longitude, elevation, location_name),
        "parameters": parameters,
        "time_resolution": time_resolution,
        "count": len(result_list),
        "provider": "meteostat",
    }


def _is_import_error(exc: Exception) -> bool:
    return isinstance(exc, (ImportError, ModuleNotFoundError))


def _provider_unavailable(
    latitude: float,
    longitude: float,
    elevation: Optional[float],
    location_name: Optional[str],
    reason: str,
) -> Dict[str, Any]:
    return {
        "code": "provider_unavailable",
        "error": f"Weather provider unavailable: {reason}",
        "location": _location_payload(latitude, longitude, elevation, location_name),
        "provider": "meteostat",
    }


def _location_payload(
    latitude: float,
    longitude: float,
    elevation: Optional[float],
    location_name: Optional[str],
) -> Dict[str, Any]:
    return {"latitude": float(latitude), "longitude": float(longitude), "elevation": elevation, "name": location_name}
