"""
Solar radiation tool to estimate incident radiation on surfaces.

This module provides functionality to calculate solar radiation components
on surfaces with different orientations and locations.
"""

import json
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from utils.constants import DEG_C_to_K, DEFAULT_GROUND_ALBEDO
from utils.import_helpers import HT_AVAILABLE

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
) -> str:
    """Estimates incident solar radiation on a tilted surface.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        datetime_utc: Date and time in UTC (ISO format)
        surface_tilt: Surface tilt angle from horizontal (0=horizontal, 90=vertical) in degrees
        surface_azimuth: Surface azimuth angle (0=North, 90=East, 180=South, 270=West) in degrees
        altitude: Altitude above sea level in meters
        atmospheric_pressure: Atmospheric pressure in Pascals
        direct_normal_irradiance_dni: Measured or estimated DNI (W/m²)
        diffuse_horizontal_irradiance_dhi: Measured or estimated DHI (W/m²)
        
    Returns:
        JSON string with calculated solar radiation components
    """
    try:
        # Parse datetime
        try:
            dt_obj = datetime.fromisoformat(datetime_utc.replace('Z', '+00:00'))
        except ValueError as e:
            return json.dumps({
                "error": f"Invalid datetime format. Please use ISO format (YYYY-MM-DDTHH:MM:SSZ). Error: {str(e)}"
            })
        
        # Extract day of year and hour
        day_of_year = dt_obj.timetuple().tm_yday
        hour_of_day_utc = dt_obj.hour + dt_obj.minute / 60.0 + dt_obj.second / 3600.0
        
        # Calculate solar position (altitude and azimuth)
        if HT_AVAILABLE:
            # Try using HT library functions if available
            import ht
            if hasattr(ht, 'solar'):
                try:
                    # Use ht.solar module if available
                    logger.info("Using HT library for solar position calculation")
                    # Implementation would depend on exact ht.solar functions
                    # This is a placeholder for HT-specific code
                    solar_altitude, solar_azimuth = None, None
                except Exception as ht_error:
                    logger.warning(f"Error using HT library for solar calculation: {ht_error}")
                    solar_altitude, solar_azimuth = None, None
            else:
                solar_altitude, solar_azimuth = None, None
        else:
            solar_altitude, solar_azimuth = None, None
        
        # If HT library didn't provide solar position, use our own calculation
        if solar_altitude is None or solar_azimuth is None:
            # Convert latitude and longitude to radians
            lat_rad = math.radians(latitude)
            lon_rad = math.radians(longitude)
            
            # Solar declination (approximate formula, radians)
            declination = math.radians(23.45 * math.sin(math.radians((360/365) * (day_of_year - 81))))
            
            # Solar hour angle (15° per hour)
            # Adjust hour_of_day to local solar time (approximate)
            local_solar_hour = hour_of_day_utc + longitude / 15.0
            hour_angle = math.radians((local_solar_hour - 12) * 15)
            
            # Solar altitude angle
            sin_altitude = (math.sin(lat_rad) * math.sin(declination) + 
                           math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle))
            solar_altitude_rad = math.asin(sin_altitude)
            solar_altitude_deg = math.degrees(solar_altitude_rad)
            
            # Solar azimuth angle
            cos_azimuth = ((math.sin(declination) * math.cos(lat_rad) - 
                           math.cos(declination) * math.sin(lat_rad) * math.cos(hour_angle)) / 
                           math.cos(solar_altitude_rad))
            solar_azimuth_rad = math.acos(max(-1.0, min(1.0, cos_azimuth)))  # Ensure within -1 to 1
            
            # Adjust for morning/afternoon
            if hour_angle < 0:
                solar_azimuth_deg = 180 - math.degrees(solar_azimuth_rad)
            else:
                solar_azimuth_deg = 180 + math.degrees(solar_azimuth_rad)
            
            # Normalize to 0-360 range
            solar_azimuth_deg = solar_azimuth_deg % 360
        
        # Get or estimate direct normal irradiance (DNI) and diffuse horizontal irradiance (DHI)
        if direct_normal_irradiance_dni is None or diffuse_horizontal_irradiance_dhi is None:
            # Estimate using clear sky model if not provided
            # Using simple ASHRAE clear sky model
            
            # First, check if it's nighttime (no radiation)
            if solar_altitude_deg <= 0:
                return json.dumps({
                    "total_solar_radiation_on_tilt_w_m2": 0.0,
                    "direct_solar_radiation_on_tilt_w_m2": 0.0,
                    "diffuse_solar_radiation_on_tilt_w_m2": 0.0,
                    "reflected_solar_radiation_on_tilt_w_m2": 0.0,
                    "solar_altitude_deg": solar_altitude_deg,
                    "solar_azimuth_deg": solar_azimuth_deg,
                    "is_nighttime": True,
                    "angle_of_incidence_deg": 90.0,
                    "note": "No solar radiation available (sun below horizon)"
                })
            
            # Simplified clear sky model
            extraterrestrial_normal_radiation = 1367.0  # Solar constant (W/m²)
            # Adjust for day of year (Earth's orbit eccentricity)
            day_angle = 2 * math.pi * day_of_year / 365.0
            extraterrestrial_normal_radiation *= (1.00011 + 0.034221 * math.cos(day_angle) + 
                                                 0.00128 * math.sin(day_angle) +
                                                 0.000719 * math.cos(2 * day_angle) + 
                                                 0.000077 * math.sin(2 * day_angle))
            
            # Air mass calculation
            if solar_altitude_deg > 0:
                air_mass = 1.0 / (math.sin(math.radians(solar_altitude_deg)) + 
                                 0.50572 * (6.07995 + solar_altitude_deg)**(-1.6364))
            else:
                air_mass = 0
                
            # Simplified atmospheric transmittance
            # Adjust for altitude if provided
            altitude_factor = math.exp(-altitude / 8000.0)  # Approximate scale height of atmosphere
            beam_transmittance = 0.56 * (math.exp(-0.65 * air_mass * altitude_factor) + 
                                        math.exp(-0.095 * air_mass * altitude_factor))
            
            # Estimate DNI and DHI
            dni_est = extraterrestrial_normal_radiation * beam_transmittance
            
            # Simplified diffuse radiation model
            diffuse_factor = 0.271 - 0.294 * beam_transmittance
            dhi_est = extraterrestrial_normal_radiation * diffuse_factor * math.sin(math.radians(solar_altitude_deg))
            
            # Use estimated values
            if direct_normal_irradiance_dni is None:
                direct_normal_irradiance_dni = dni_est
            if diffuse_horizontal_irradiance_dhi is None:
                diffuse_horizontal_irradiance_dhi = dhi_est
        
        # Calculate angle of incidence (AOI) on the tilted surface
        surface_tilt_rad = math.radians(surface_tilt)
        surface_azimuth_rad = math.radians(surface_azimuth)
        solar_altitude_rad = math.radians(solar_altitude_deg)
        solar_azimuth_rad = math.radians(solar_azimuth_deg)
        
        cos_aoi = (math.sin(solar_altitude_rad) * math.cos(surface_tilt_rad) + 
                  math.cos(solar_altitude_rad) * math.sin(surface_tilt_rad) * 
                  math.cos(solar_azimuth_rad - surface_azimuth_rad))
        
        # Ensure value is within -1 to 1 for acos
        cos_aoi = max(-1.0, min(1.0, cos_aoi))
        aoi_deg = math.degrees(math.acos(cos_aoi))
        
        # Calculate components of solar radiation on tilted surface
        # 1. Direct component on tilted surface
        if aoi_deg < 90 and solar_altitude_deg > 0:
            # Sun is above horizon and incident on the surface
            direct_tilt = direct_normal_irradiance_dni * cos_aoi
        else:
            direct_tilt = 0.0
        
        # 2. Diffuse component (isotropic sky model)
        diffuse_tilt = diffuse_horizontal_irradiance_dhi * (1.0 + math.cos(surface_tilt_rad)) / 2.0
        
        # 3. Ground-reflected component
        # Calculate global horizontal irradiance (GHI)
        ghi = direct_normal_irradiance_dni * math.sin(math.radians(solar_altitude_deg)) + diffuse_horizontal_irradiance_dhi
        
        # Assume ground reflectivity (albedo) = 0.2
        albedo = DEFAULT_GROUND_ALBEDO
        reflected_tilt = ghi * albedo * (1.0 - math.cos(surface_tilt_rad)) / 2.0
        
        # 4. Total radiation on tilted surface
        total_tilt = direct_tilt + diffuse_tilt + reflected_tilt
        
        # Create result
        result = {
            "total_solar_radiation_on_tilt_w_m2": total_tilt,
            "direct_solar_radiation_on_tilt_w_m2": direct_tilt,
            "diffuse_solar_radiation_on_tilt_w_m2": diffuse_tilt,
            "reflected_solar_radiation_on_tilt_w_m2": reflected_tilt,
            "solar_altitude_deg": solar_altitude_deg,
            "solar_azimuth_deg": solar_azimuth_deg,
            "angle_of_incidence_deg": aoi_deg,
            "global_horizontal_irradiance_w_m2": ghi,
            "direct_normal_irradiance_w_m2": direct_normal_irradiance_dni,
            "diffuse_horizontal_irradiance_w_m2": diffuse_horizontal_irradiance_dhi,
            "surface_orientation": {
                "tilt_deg": surface_tilt,
                "azimuth_deg": surface_azimuth
            },
            "location": {
                "latitude_deg": latitude,
                "longitude_deg": longitude,
                "altitude_m": altitude
            },
            "datetime_utc": datetime_utc,
            "day_of_year": day_of_year
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in calculate_solar_radiation_on_surface: {e}", exc_info=True)
        return json.dumps({
            "error": f"An unexpected error occurred: {str(e)}"
        })
