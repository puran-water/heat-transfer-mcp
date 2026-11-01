"""Bounded sky-temperature defaults for radiation calculations."""

from __future__ import annotations

from typing import Any, Dict, Optional


def bounded_sky_temperature(
    *,
    ambient_air_temperature_K: float,
    sky_temperature_K: Optional[float] = None,
    cloud_condition: str = "clear",
) -> Dict[str, Any]:
    """Return a sky temperature and notes for the assumption used.

    When callers do not provide a sky temperature, use a bounded engineering
    default retained as an explicit assumption, not a measured longwave basis.
    It does not constrain the solved cladding temperature.
    """
    ambient = float(ambient_air_temperature_K)
    if sky_temperature_K is not None:
        return {
            "sky_temperature_k": float(sky_temperature_K),
            "source": "provided",
            "notes": ["Sky temperature supplied by caller."],
        }

    condition = (cloud_condition or "clear").strip().lower().replace("_", " ")
    if condition in {"overcast", "cloudy", "full cloud"}:
        depression_K = 0.0
        label = "overcast"
    elif condition in {"partial", "partly cloudy", "partial clouds", "partly cloudy sky"}:
        depression_K = 5.0
        label = "partial clouds"
    else:
        depression_K = 10.0
        label = "clear sky"

    sky = max(0.0, ambient - depression_K)
    return {
        "sky_temperature_k": sky,
        "source": "ambient_default",
        "cloud_condition": label,
        "depression_K": depression_K,
        "notes": [
            f"sky_temperature was not supplied; assumed {label} with T_sky = T_ambient - {depression_K:g} K.",
        ],
    }
