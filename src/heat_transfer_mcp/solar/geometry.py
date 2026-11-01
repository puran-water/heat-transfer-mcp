"""Solar position and tilted-surface geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class SolarGeometry:
    day_of_year: int
    hour_of_day_utc: float
    solar_altitude_deg: float
    solar_azimuth_deg: float
    angle_of_incidence_deg: float
    cos_angle_of_incidence: float


def parse_datetime_utc(datetime_utc: str) -> datetime:
    """Parse ISO UTC timestamps accepted by the MCP tool."""
    value = datetime.fromisoformat(datetime_utc.replace("Z", "+00:00"))
    # This field is UTC when no offset is supplied; explicit offsets must be converted.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def calculate_solar_geometry(
    *,
    latitude: float,
    longitude: float,
    datetime_utc: str,
    surface_tilt: float,
    surface_azimuth: float,
) -> SolarGeometry:
    """Calculate approximate solar altitude, azimuth, and surface incidence."""
    for name, value, low, high in (("latitude", latitude, -90, 90), ("longitude", longitude, -180, 180),
                                   ("surface_tilt", surface_tilt, 0, 180), ("surface_azimuth", surface_azimuth, 0, 360)):
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must lie in [{low}, {high}]")
    dt_obj = parse_datetime_utc(datetime_utc)
    day_of_year = dt_obj.timetuple().tm_yday
    hour_of_day_utc = dt_obj.hour + dt_obj.minute / 60.0 + dt_obj.second / 3600.0

    lat_rad = math.radians(latitude)
    declination = math.radians(23.45 * math.sin(math.radians((360.0 / 365.0) * (day_of_year - 81))))

    local_solar_hour = hour_of_day_utc + longitude / 15.0
    hour_angle = math.radians((local_solar_hour - 12.0) * 15.0)

    sin_altitude = (
        math.sin(lat_rad) * math.sin(declination)
        + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
    )
    sin_altitude = max(-1.0, min(1.0, sin_altitude))
    solar_altitude_rad = math.asin(sin_altitude)
    solar_altitude_deg = math.degrees(solar_altitude_rad)

    cos_altitude = max(1.0e-12, math.cos(solar_altitude_rad))
    cos_azimuth = (
        math.sin(declination) * math.cos(lat_rad)
        - math.cos(declination) * math.sin(lat_rad) * math.cos(hour_angle)
    ) / cos_altitude
    cos_azimuth = max(-1.0, min(1.0, cos_azimuth))
    sin_azimuth = -math.cos(declination) * math.sin(hour_angle) / cos_altitude
    solar_azimuth_deg = math.degrees(math.atan2(sin_azimuth, cos_azimuth)) % 360.0

    surface_tilt_rad = math.radians(surface_tilt)
    surface_azimuth_rad = math.radians(surface_azimuth)
    solar_azimuth_surface_rad = math.radians(solar_azimuth_deg)
    cos_aoi = (
        math.sin(solar_altitude_rad) * math.cos(surface_tilt_rad)
        + math.cos(solar_altitude_rad)
        * math.sin(surface_tilt_rad)
        * math.cos(solar_azimuth_surface_rad - surface_azimuth_rad)
    )
    cos_aoi = max(-1.0, min(1.0, cos_aoi))
    angle_of_incidence_deg = math.degrees(math.acos(cos_aoi))

    return SolarGeometry(
        day_of_year=day_of_year,
        hour_of_day_utc=hour_of_day_utc,
        solar_altitude_deg=solar_altitude_deg,
        solar_azimuth_deg=solar_azimuth_deg,
        angle_of_incidence_deg=angle_of_incidence_deg,
        cos_angle_of_incidence=cos_aoi,
    )
