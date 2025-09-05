"""
WeatherDataService for efficient context-aware weather data fetching.

This service fetches weather data once per location, computes all statistics internally,
and returns only essential design values to preserve MCP context.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from utils.import_helpers import METEOSTAT_AVAILABLE, PANDAS_AVAILABLE
from utils.validation import validate_lat_lon

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("heat-transfer-mcp.weather_service")


class WeatherDataService:
    """Fetches weather data once, computes all statistics internally,
    returns only essential design values to preserve MCP context."""
    
    def __init__(self):
        """Initialize the weather data service with an in-memory cache."""
        self._cache = {}  # In-memory cache for session
        self._era5_cache = {}  # Separate cache for ERA5 data
        
    def get_design_conditions(
        self, 
        lat: float, 
        lon: float,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        percentiles: Optional[List[float]] = None,
        time_resolution: str = "daily"
    ) -> Dict[str, Any]:
        """
        Get design conditions for a location, returning only essential values.
        
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees  
            start_date: Start date for analysis (default: 10 years ago)
            end_date: End date for analysis (default: last complete year)
            percentiles: List of percentiles to compute (default: [0.90, 0.95, 0.99])
            time_resolution: "daily" or "hourly"
            
        Returns:
            Dict with only design values, not raw data:
            {
                "location": {"lat": 40.39, "lon": -84.38, "name": "Minster, OH"},
                "data_summary": {
                    "source": "meteostat",
                    "date_range": "2014-01-01 to 2023-12-31", 
                    "data_points": 3650,
                    "resolution": "daily"
                },
                "design_conditions": {
                    "cold_90th": {"temp_c": -6, "temp_k": 267.15, "wind_m_s": 5.8},
                    "cold_95th": {"temp_c": -9, "temp_k": 264.15, "wind_m_s": 6.63},
                    "cold_99th": {"temp_c": -15, "temp_k": 258.15, "wind_m_s": 8.14}
                },
                "concurrent_conditions": {
                    "dew_point_median_c": -12,
                    "dew_point_k": 261.15
                },
                "annual_average_c": 10
            }
        """
        # Validate inputs
        try:
            validate_lat_lon(lat, lon)
        except Exception as e:
            return {"error": str(e)}
            
        if percentiles is None:
            percentiles = [0.90, 0.95, 0.99]
            
        # Default to last 10 complete years if dates not provided
        if not end_date:
            # End at last complete year
            end_date = datetime.now().replace(year=datetime.now().year-1, month=12, day=31)
        if not start_date:
            # Start 10 years before end
            start_date = end_date.replace(year=end_date.year-9, month=1, day=1)
            
        # Convert to datetime if strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        # Check cache
        cache_key = f"{lat}_{lon}_{start_date.date()}_{end_date.date()}_{time_resolution}"
        if cache_key in self._cache:
            return self._process_percentiles(self._cache[cache_key], percentiles, cache_key)
            
        if not METEOSTAT_AVAILABLE or not PANDAS_AVAILABLE:
            return {
                "error": "Meteostat and pandas are required for weather data fetching",
                "location": {"lat": lat, "lon": lon}
            }
            
        try:
            from meteostat import Point, Daily, Hourly
            import pandas as pd
            
            # Fetch ONCE
            loc = Point(lat, lon)
            if time_resolution.lower() == "hourly":
                df = Hourly(loc, start_date, end_date).fetch()
            else:
                df = Daily(loc, start_date, end_date).fetch()
                
            if df is None or df.empty:
                return {
                    "error": "No weather data available for this location",
                    "location": {"lat": lat, "lon": lon}
                }
                
            # Cache the DataFrame
            self._cache[cache_key] = df
            
            # Process ALL percentiles internally
            return self._process_percentiles(df, percentiles, cache_key)
            
        except Exception as e:
            logger.error(f"Weather data fetch failed: {e}", exc_info=True)
            return {"error": str(e)}
            
    def _process_percentiles(
        self, 
        df: Any,  # pandas DataFrame
        percentiles: List[float],
        cache_key: str
    ) -> Dict[str, Any]:
        """Process data internally, return only design values."""
        import pandas as pd
        
        # Extract metadata
        parts = cache_key.split('_')
        lat, lon = float(parts[0]), float(parts[1])
        start_str = parts[2]
        end_str = parts[3]
        resolution = parts[4] if len(parts) > 4 else "daily"
        
        result = {
            "location": {"lat": lat, "lon": lon},
            "data_summary": {
                "source": "meteostat",
                "date_range": f"{start_str} to {end_str}",
                "data_points": len(df),
                "resolution": resolution
            },
            "design_conditions": {},
            "concurrent_conditions": {},
            "annual_average_c": None
        }
        
        # Temperature percentiles for cold design
        temp_col = None
        if "tmin" in df.columns:
            temp_col = "tmin"
        elif "tavg" in df.columns:
            temp_col = "tavg"
        elif "temp" in df.columns:
            temp_col = "temp"
            
        if temp_col:
            temp_series = df[temp_col].dropna()
            
            # For cold design, we want LOW temperature percentiles
            # p99 cold design means only 1% of days are colder
            for p in percentiles:
                cold_quantile = 1.0 - p if p > 0.5 else p
                temp_c = float(temp_series.quantile(cold_quantile))
                
                # Get concurrent wind speed
                wind_m_s = None
                if "wspd" in df.columns:
                    # Find wind speed at temperature percentile
                    temp_threshold = temp_series.quantile(cold_quantile)
                    cold_days = df[df[temp_col] <= temp_threshold]
                    if not cold_days.empty and "wspd" in cold_days.columns:
                        wind_kmh = cold_days["wspd"].quantile(0.95)  # 95th percentile wind during cold
                        wind_m_s = float(wind_kmh / 3.6) if pd.notnull(wind_kmh) else None
                        
                label = f"cold_{int(p*100)}th"
                result["design_conditions"][label] = {
                    "temp_c": round(temp_c, 1),
                    "temp_k": round(temp_c + 273.15, 2),
                    "wind_m_s": round(wind_m_s, 2) if wind_m_s else None
                }
                
            # Annual average
            if "tavg" in df.columns:
                result["annual_average_c"] = round(float(df["tavg"].mean()), 1)
            else:
                result["annual_average_c"] = round(float(temp_series.mean()), 1)
                
        # Dew point for sky temperature calculations
        if "tdew" in df.columns:
            dew_series = df["tdew"].dropna()
            if not dew_series.empty:
                dew_median_c = float(dew_series.median())
                result["concurrent_conditions"]["dew_point_median_c"] = round(dew_median_c, 1)
                result["concurrent_conditions"]["dew_point_k"] = round(dew_median_c + 273.15, 2)
        elif "rhum" in df.columns and temp_col:
            # Estimate dew point from temperature and humidity
            temp_median = df[temp_col].median()
            rhum_median = df["rhum"].median()
            if pd.notnull(temp_median) and pd.notnull(rhum_median):
                # Magnus formula approximation
                a = 17.27
                b = 237.7
                alpha = (a * temp_median) / (b + temp_median) + pd.np.log(rhum_median/100)
                dew_c = (b * alpha) / (a - alpha)
                result["concurrent_conditions"]["dew_point_median_c"] = round(float(dew_c), 1)
                result["concurrent_conditions"]["dew_point_k"] = round(float(dew_c) + 273.15, 2)
        else:
            # Fallback: Estimate dew point as temperature minus typical dew point depression
            # For cold climates in winter, typical depression is 3-5°C
            if temp_col:
                temp_median = df[temp_col].median()
                if pd.notnull(temp_median):
                    # Try to fetch from ERA5 if requests is available
                    era5_dew = self._fetch_era5_dew_point(lat, lon, start_str, end_str)
                    if era5_dew is not None:
                        result["concurrent_conditions"]["dew_point_median_c"] = round(era5_dew, 1)
                        result["concurrent_conditions"]["dew_point_k"] = round(era5_dew + 273.15, 2)
                        result["concurrent_conditions"]["dew_point_source"] = "era5"
                    else:
                        # Use larger depression (5°C) for conservative sky temperature estimate
                        estimated_dew_c = temp_median - 5.0
                        result["concurrent_conditions"]["dew_point_median_c"] = round(float(estimated_dew_c), 1)
                        result["concurrent_conditions"]["dew_point_k"] = round(float(estimated_dew_c) + 273.15, 2)
                        result["concurrent_conditions"]["dew_point_source"] = "estimated"
                
        return result
        
    def get_location_name(self, lat: float, lon: float) -> str:
        """Get a descriptive name for a location (simple geocoding)."""
        # Simple lookup for common locations
        known_locations = {
            (40.39, -84.38): "Minster, OH",
            (51.51, -0.13): "London, UK", 
            (40.71, -74.01): "New York, NY",
            (34.05, -118.24): "Los Angeles, CA",
            (35.68, 139.65): "Tokyo, Japan",
            (-33.87, 151.21): "Sydney, Australia"
        }
        
        # Check for close matches
        for (known_lat, known_lon), name in known_locations.items():
            if abs(lat - known_lat) < 0.1 and abs(lon - known_lon) < 0.1:
                return name
                
        return f"Location ({lat:.2f}, {lon:.2f})"


    def _fetch_era5_dew_point(self, lat: float, lon: float, start_date: str, end_date: str) -> Optional[float]:
        """Fetch dew point data from Open-Meteo ERA5 reanalysis.
        
        Returns median dew point in Celsius or None if fetch fails.
        """
        if requests is None:
            return None
            
        cache_key = f"era5_{lat}_{lon}_{start_date}_{end_date}"
        if cache_key in self._era5_cache:
            return self._era5_cache[cache_key]
            
        try:
            # For long date ranges, fetch by year to avoid timeout
            import pandas as pd
            from datetime import datetime
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # If range > 2 years, sample the middle year for efficiency
            if (end_dt - start_dt).days > 730:
                # Use middle year as representative
                mid_year = start_dt.year + (end_dt.year - start_dt.year) // 2
                sample_start = f"{mid_year}-01-01"
                sample_end = f"{mid_year}-12-31"
            else:
                sample_start = start_date
                sample_end = end_date
            
            url = "https://archive-api.open-meteo.com/v1/era5"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": sample_start,
                "end_date": sample_end,
                "daily": "dewpoint_2m_mean",  # Daily mean for efficiency
                "timezone": "UTC"
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if "daily" in data and "dewpoint_2m_mean" in data["daily"]:
                    dew_points = [d for d in data["daily"]["dewpoint_2m_mean"] if d is not None]
                    if dew_points:
                        median_dew = pd.Series(dew_points).median()
                        self._era5_cache[cache_key] = float(median_dew)
                        logger.info(f"Fetched ERA5 dew point for ({lat}, {lon}): {median_dew:.1f}°C")
                        return float(median_dew)
        except Exception as e:
            logger.warning(f"ERA5 dew point fetch failed: {e}")
            
        return None

# Module-level singleton
_weather_service = None

def get_weather_service() -> WeatherDataService:
    """Get the singleton weather service instance."""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherDataService()
    return _weather_service