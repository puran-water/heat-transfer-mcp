"""
Ambient conditions tool to retrieve location-specific weather data.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from heat_transfer_mcp.errors import CalculationError
from heat_transfer_mcp.weather.geocode import geocode_location
from heat_transfer_mcp.weather.provider import fetch_weather_data

logger = logging.getLogger("heat-transfer-mcp.ambient_conditions")


def get_ambient_conditions(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    elevation: Optional[float] = None,
    location_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_resolution: str = "daily",
    parameters: List[str] = ["tavg", "wspd", "pres", "rhum", "tsun"],
) -> str:
    """Fetch weather observations by location name or coordinates."""
    try:
        geocode_payload = None
        if location_name:
            geocoded = geocode_location(location_name)
            latitude = geocoded.latitude
            longitude = geocoded.longitude
            if elevation is None:
                elevation = geocoded.elevation
            geocode_payload = geocoded.to_location_dict()
        elif latitude is None or longitude is None:
            return json.dumps({"error": "Either location_name or both latitude and longitude must be provided."})

        response = fetch_weather_data(
            latitude=float(latitude),
            longitude=float(longitude),
            elevation=elevation,
            location_name=location_name,
            start_date=start_date,
            end_date=end_date,
            time_resolution=time_resolution,
            parameters=parameters,
        )
        if geocode_payload is not None:
            response["geocode"] = geocode_payload
            response["location"] = {**response.get("location", {}), **geocode_payload}
        return json.dumps(response)
    except CalculationError as exc:
        payload = exc.to_dict()
        payload["error"] = payload["message"]
        return json.dumps(payload)
    except Exception as exc:
        logger.error(f"General error in get_ambient_conditions: {exc}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(exc)}"})
