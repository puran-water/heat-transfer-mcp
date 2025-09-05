"""
Shared validation helpers for Heat Transfer MCP tools.

Keep these light-weight and reusable to avoid duplicated checks
across omnibus tools.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


class ValidationError(ValueError):
    pass


def require_positive(value: float, name: str) -> None:
    if value is None or value <= 0:
        raise ValidationError(f"{name} must be > 0; got {value}")


def require_non_negative(value: float, name: str) -> None:
    if value is None or value < 0:
        raise ValidationError(f"{name} must be >= 0; got {value}")


def validate_geometry_dimensions(geometry: str, dimensions: Dict[str, Any]) -> None:
    g = (geometry or "").lower()
    if any(k not in dimensions for k in ("diameter", "height", "length", "width")):
        # Will only enforce keys that are present; geometry determines which are required
        pass
    # For cylinders/tanks: require diameter > 0 and height/length > 0 as applicable
    if "cylinder" in g or "tank" in g or "pipe" in g:
        if "diameter" in dimensions:
            require_positive(float(dimensions["diameter"]), "dimensions.diameter")
        if "height" in dimensions:
            require_positive(float(dimensions["height"]), "dimensions.height")
        if "length" in dimensions:
            require_positive(float(dimensions["length"]), "dimensions.length")
    # Flat surfaces/walls
    if "flat" in g or "wall" in g:
        if "length" in dimensions:
            require_positive(float(dimensions["length"]), "dimensions.length")
        if "width" in dimensions:
            require_positive(float(dimensions["width"]), "dimensions.width")


def validate_lat_lon(latitude: float, longitude: float) -> None:
    if latitude is None or longitude is None:
        raise ValidationError("latitude and longitude are required")
    if not (-90.0 <= float(latitude) <= 90.0):
        raise ValidationError(f"latitude must be between -90 and 90; got {latitude}")
    if not (-180.0 <= float(longitude) <= 180.0):
        raise ValidationError(f"longitude must be between -180 and 180; got {longitude}")


def validate_date_range(start_date: Any, end_date: Any) -> None:
    if not start_date or not end_date:
        raise ValidationError("start_date and end_date are required in YYYY-MM-DD format")
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        ed = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
    except Exception:
        raise ValidationError("start_date and end_date must be in YYYY-MM-DD format")
    if sd > ed:
        raise ValidationError("start_date must be <= end_date")


def ensure_list_nonempty(name: str, value: Any) -> None:
    if not isinstance(value, list) or len(value) == 0:
        raise ValidationError(f"{name} must be a non-empty list")


def set_in_path(obj: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split('.') if path else []
    if not parts:
        raise ValidationError("Empty parameter path")
    cur = obj
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

