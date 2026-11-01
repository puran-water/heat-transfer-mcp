"""Solar irradiance model with validation and nighttime handling."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from heat_transfer_mcp.utils.constants import DEFAULT_GROUND_ALBEDO

from heat_transfer_mcp.errors import CalculationError
from heat_transfer_mcp.solar.geometry import calculate_solar_geometry

NEGATIVE_IRRADIANCE = "NEGATIVE_IRRADIANCE"


def calculate_tilted_surface_irradiance(
    *,
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
) -> Dict[str, Any]:
    """Calculate irradiance on a tilted surface."""
    _validate_non_negative_irradiance(direct_normal_irradiance_dni, "direct_normal_irradiance_dni")
    _validate_non_negative_irradiance(diffuse_horizontal_irradiance_dhi, "diffuse_horizontal_irradiance_dhi")
    if atmospheric_pressure is not None and (direct_normal_irradiance_dni is None or diffuse_horizontal_irradiance_dhi is None):
        raise CalculationError(code="unsupported_pressure_correction", field="atmospheric_pressure",
            message="The clear-sky approximation has no pressure correction; supply qualified DNI and DHI for projection")

    geometry = calculate_solar_geometry(
        latitude=float(latitude),
        longitude=float(longitude),
        datetime_utc=datetime_utc,
        surface_tilt=float(surface_tilt),
        surface_azimuth=float(surface_azimuth),
    )
    is_nighttime = geometry.solar_altitude_deg <= 0.0
    supplied_irradiance = direct_normal_irradiance_dni is not None or diffuse_horizontal_irradiance_dhi is not None

    if is_nighttime and not allow_measured_night_irradiance:
        return _nighttime_zero_result(
            geometry=geometry,
            latitude=latitude,
            longitude=longitude,
            datetime_utc=datetime_utc,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            altitude=altitude,
            atmospheric_pressure=atmospheric_pressure,
            supplied_irradiance=supplied_irradiance,
        )

    dni, dhi, irradiance_source = _resolve_irradiance(
        geometry=geometry,
        altitude=float(altitude),
        direct_normal_irradiance_dni=direct_normal_irradiance_dni,
        diffuse_horizontal_irradiance_dhi=diffuse_horizontal_irradiance_dhi,
    )

    surface_tilt_rad = math.radians(surface_tilt)
    direct_tilt = 0.0
    if allow_measured_night_irradiance or geometry.solar_altitude_deg > 0.0:
        direct_tilt = max(0.0, dni * max(0.0, geometry.cos_angle_of_incidence))
    diffuse_tilt = max(0.0, dhi) * (1.0 + math.cos(surface_tilt_rad)) / 2.0

    raw_ghi = dni * math.sin(math.radians(geometry.solar_altitude_deg)) + dhi
    ghi = max(0.0, raw_ghi)
    reflected_tilt = ghi * DEFAULT_GROUND_ALBEDO * (1.0 - math.cos(surface_tilt_rad)) / 2.0
    total_tilt = max(0.0, direct_tilt + diffuse_tilt + reflected_tilt)

    result = _base_result(
        geometry=geometry,
        latitude=latitude,
        longitude=longitude,
        datetime_utc=datetime_utc,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        altitude=altitude,
        atmospheric_pressure=atmospheric_pressure,
    )
    result.update(
        {
            "total_solar_radiation_on_tilt_w_m2": total_tilt,
            "direct_solar_radiation_on_tilt_w_m2": direct_tilt,
            "diffuse_solar_radiation_on_tilt_w_m2": diffuse_tilt,
            "reflected_solar_radiation_on_tilt_w_m2": reflected_tilt,
            "global_horizontal_irradiance_w_m2": ghi,
            "raw_global_horizontal_irradiance_w_m2": raw_ghi,
            "direct_normal_irradiance_w_m2": dni,
            "diffuse_horizontal_irradiance_w_m2": dhi,
            "is_nighttime": is_nighttime,
            "irradiance_source": irradiance_source,
            "allow_measured_night_irradiance": allow_measured_night_irradiance,
        }
    )
    if raw_ghi < 0.0:
        result.setdefault("warnings", []).append("GHI was clamped to zero before reflected-radiation calculation.")
    return result


def _validate_non_negative_irradiance(value: Optional[float], field: str) -> None:
    if value is None:
        return
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise CalculationError(
            code=NEGATIVE_IRRADIANCE,
            message=f"{field} must be >= 0 W/m^2",
            field=field,
            details={"value": numeric},
        )


def _resolve_irradiance(
    *,
    geometry,
    altitude: float,
    direct_normal_irradiance_dni: Optional[float],
    diffuse_horizontal_irradiance_dhi: Optional[float],
) -> tuple[float, float, str]:
    dni_est, dhi_est = _estimate_clear_sky_irradiance(
        day_of_year=geometry.day_of_year,
        solar_altitude_deg=geometry.solar_altitude_deg,
        altitude=altitude,
    )
    dni = dni_est if direct_normal_irradiance_dni is None else float(direct_normal_irradiance_dni)
    dhi = dhi_est if diffuse_horizontal_irradiance_dhi is None else float(diffuse_horizontal_irradiance_dhi)
    if direct_normal_irradiance_dni is None and diffuse_horizontal_irradiance_dhi is None:
        source = "estimated_clear_sky"
    elif direct_normal_irradiance_dni is None or diffuse_horizontal_irradiance_dhi is None:
        source = "mixed_supplied_and_estimated"
    else:
        source = "supplied"
    return max(0.0, dni), max(0.0, dhi), source


def _estimate_clear_sky_irradiance(*, day_of_year: int, solar_altitude_deg: float, altitude: float) -> tuple[float, float]:
    if solar_altitude_deg <= 0.0:
        return 0.0, 0.0

    extraterrestrial_normal_radiation = 1367.0
    day_angle = 2.0 * math.pi * day_of_year / 365.0
    extraterrestrial_normal_radiation *= (
        1.00011
        + 0.034221 * math.cos(day_angle)
        + 0.00128 * math.sin(day_angle)
        + 0.000719 * math.cos(2.0 * day_angle)
        + 0.000077 * math.sin(2.0 * day_angle)
    )

    air_mass = 1.0 / (
        math.sin(math.radians(solar_altitude_deg)) + 0.50572 * (6.07995 + solar_altitude_deg) ** (-1.6364)
    )
    altitude_factor = math.exp(-altitude / 8000.0)
    beam_transmittance = 0.56 * (
        math.exp(-0.65 * air_mass * altitude_factor) + math.exp(-0.095 * air_mass * altitude_factor)
    )
    dni = extraterrestrial_normal_radiation * beam_transmittance
    diffuse_factor = max(0.0, 0.271 - 0.294 * beam_transmittance)
    dhi = extraterrestrial_normal_radiation * diffuse_factor * math.sin(math.radians(solar_altitude_deg))
    return max(0.0, dni), max(0.0, dhi)


def _nighttime_zero_result(
    *,
    geometry,
    latitude: float,
    longitude: float,
    datetime_utc: str,
    surface_tilt: float,
    surface_azimuth: float,
    altitude: float,
    atmospheric_pressure: Optional[float],
    supplied_irradiance: bool,
) -> Dict[str, Any]:
    result = _base_result(
        geometry=geometry,
        latitude=latitude,
        longitude=longitude,
        datetime_utc=datetime_utc,
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        altitude=altitude,
        atmospheric_pressure=atmospheric_pressure,
    )
    result.update(
        {
            "total_solar_radiation_on_tilt_w_m2": 0.0,
            "direct_solar_radiation_on_tilt_w_m2": 0.0,
            "diffuse_solar_radiation_on_tilt_w_m2": 0.0,
            "reflected_solar_radiation_on_tilt_w_m2": 0.0,
            "global_horizontal_irradiance_w_m2": 0.0,
            "raw_global_horizontal_irradiance_w_m2": 0.0,
            "direct_normal_irradiance_w_m2": 0.0,
            "diffuse_horizontal_irradiance_w_m2": 0.0,
            "is_nighttime": True,
            "irradiance_source": "nighttime_zeroed_supplied" if supplied_irradiance else "nighttime_zeroed_estimated",
            "allow_measured_night_irradiance": False,
            "note": "No solar radiation available because the sun is below the horizon.",
        }
    )
    return result


def _base_result(
    *,
    geometry,
    latitude: float,
    longitude: float,
    datetime_utc: str,
    surface_tilt: float,
    surface_azimuth: float,
    altitude: float,
    atmospheric_pressure: Optional[float],
) -> Dict[str, Any]:
    return {
        "solar_altitude_deg": geometry.solar_altitude_deg,
        "solar_azimuth_deg": geometry.solar_azimuth_deg,
        "angle_of_incidence_deg": geometry.angle_of_incidence_deg,
        "surface_orientation": {"tilt_deg": surface_tilt, "azimuth_deg": surface_azimuth},
        "location": {"latitude_deg": latitude, "longitude_deg": longitude, "altitude_m": altitude},
        "datetime_utc": datetime_utc,
        "day_of_year": geometry.day_of_year,
        "atmospheric_pressure_pa": atmospheric_pressure,
    }
