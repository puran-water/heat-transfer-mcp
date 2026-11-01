"""
Solar radiation tool to estimate incident radiation on surfaces.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from heat_transfer_mcp.errors import CalculationError
from heat_transfer_mcp.solar.model import calculate_tilted_surface_irradiance

logger = logging.getLogger("heat-transfer-mcp.solar_radiation")


def calculate_solar_radiation_on_surface(
    latitude: float,
    longitude: float,
    datetime_utc: str,
    surface_tilt: float,
    surface_azimuth: float,
    altitude: float = 0.0,
    atmospheric_pressure: Optional[float] = None,
    direct_normal_irradiance_dni: Optional[float] = None,
    diffuse_horizontal_irradiance_dhi: Optional[float] = None,
    allow_measured_night_irradiance: bool = False,
) -> str:
    """Estimate incident solar radiation on a tilted surface."""
    try:
        result = calculate_tilted_surface_irradiance(
            latitude=latitude,
            longitude=longitude,
            datetime_utc=datetime_utc,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            altitude=altitude,
            atmospheric_pressure=atmospheric_pressure,
            direct_normal_irradiance_dni=direct_normal_irradiance_dni,
            diffuse_horizontal_irradiance_dhi=diffuse_horizontal_irradiance_dhi,
            allow_measured_night_irradiance=allow_measured_night_irradiance,
        )
        return json.dumps(result)
    except CalculationError as exc:
        payload = exc.to_dict()
        payload["error"] = payload["message"]
        return json.dumps(payload)
    except ValueError as exc:
        return json.dumps({"error": f"Invalid solar input: {exc}"})
    except Exception as exc:
        logger.error(f"Unexpected error in calculate_solar_radiation_on_surface: {exc}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(exc)}"})
