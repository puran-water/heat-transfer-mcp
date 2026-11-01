"""Solar geometry and irradiance models."""

from heat_transfer_mcp.solar.geometry import calculate_solar_geometry
from heat_transfer_mcp.solar.model import calculate_tilted_surface_irradiance

__all__ = ["calculate_solar_geometry", "calculate_tilted_surface_irradiance"]
