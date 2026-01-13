"""
Ambient conditions tool to retrieve location-specific weather data.

This module provides tools to fetch weather data (temperature, wind speed,
solar radiation, humidity) for heat transfer calculations.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from utils.import_helpers import METEOSTAT_AVAILABLE, PANDAS_AVAILABLE

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
    """Provides location-specific weather data for heat transfer calculations.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        elevation: Elevation in meters (optional, improves accuracy)
        location_name: City name or well-known location (e.g., 'London, UK')
        start_date: Start date for weather data in YYYY-MM-DD format
        end_date: End date for weather data in YYYY-MM-DD format
        time_resolution: Time resolution for data ('daily' or 'hourly')
        parameters: List of weather parameters to fetch

    Returns:
        JSON string with weather data for the specified location and time period
    """
    if not METEOSTAT_AVAILABLE:
        return json.dumps({"error": "Weather data retrieval is not available. The meteostat package is not installed."})

    if not PANDAS_AVAILABLE:
        return json.dumps({"error": "Weather data processing is not available. The pandas package is not installed."})

    try:
        from meteostat import Point, Daily, Hourly
        import pandas as pd

        # Handle parameters
        if latitude is None or longitude is None:
            if location_name:
                # Basic fallback for well-known locations
                # In a real implementation, use geocoding API or meteostat's location search
                location_name_lower = location_name.lower()
                if "london" in location_name_lower:
                    latitude, longitude, elevation = 51.5072, -0.1276, 35
                elif "new york" in location_name_lower:
                    latitude, longitude, elevation = 40.7128, -74.0060, 10
                elif "los angeles" in location_name_lower:
                    latitude, longitude, elevation = 34.0522, -118.2437, 93
                elif "tokyo" in location_name_lower:
                    latitude, longitude, elevation = 35.6762, 139.6503, 40
                elif "sydney" in location_name_lower:
                    latitude, longitude, elevation = -33.8688, 151.2093, 58
                else:
                    return json.dumps(
                        {"error": "Latitude and longitude are required if location_name is not a recognized default."}
                    )
            else:
                return json.dumps({"error": "Either latitude/longitude or a recognized location_name must be provided."})

        # Validate latitude and longitude ranges
        try:
            lat_val = float(latitude)
            lon_val = float(longitude)
        except (TypeError, ValueError):
            return json.dumps({"error": "Latitude and longitude must be numeric values."})
        if not (-90.0 <= lat_val <= 90.0) or not (-180.0 <= lon_val <= 180.0):
            return json.dumps(
                {"error": "Invalid coordinates. Latitude must be between -90 and 90; longitude between -180 and 180."}
            )

        # Parse dates
        if not start_date or not end_date:
            return json.dumps({"error": "Both 'start_date' and 'end_date' are required in YYYY-MM-DD format."})
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            return json.dumps({"error": f"Invalid date format. Please use YYYY-MM-DD. Error: {str(e)}"})
        if start > end:
            return json.dumps({"error": "'start_date' must be on or before 'end_date'."})

        # Create location point
        if elevation is not None:
            location = Point(lat_val, lon_val, elevation)
        else:
            location = Point(lat_val, lon_val)

        # Fetch data based on time resolution
        try:
            if time_resolution and time_resolution.lower() == "hourly":
                data = Hourly(location, start, end)
            else:  # Default to daily
                data = Daily(location, start, end)

            data = data.fetch()

            # Check if we got any data
            if data.empty:
                return json.dumps({"result": [], "message": "No weather data found for the specified location/date range."})

            # Process the data
            # Reset index to make time a column
            data_processed = data.reset_index()

            # Convert time to string
            data_processed["time"] = data_processed["time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

            # Handle NaN values
            data_processed = data_processed.where(pd.notnull(data_processed), None)

            # Convert to dict
            result_list = data_processed.to_dict(orient="records")

            # Prepare response
            response = {
                "result": result_list,
                "location": {"latitude": latitude, "longitude": longitude, "elevation": elevation, "name": location_name},
                "parameters": parameters,
                "time_resolution": time_resolution,
                "count": len(result_list),
            }

            return json.dumps(response)

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}", exc_info=True)
            return json.dumps({"error": f"Error fetching weather data: {str(e)}"})

    except Exception as e:
        logger.error(f"General error in get_ambient_conditions: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
