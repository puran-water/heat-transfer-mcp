"""
Enhanced MCP Server for heat transfer calculations with automatic unit conversion.

This server automatically detects and enables unit conversion on startup,
requiring no manual activation.

Author: Claude AI
"""

import logging
import os
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("heat-transfer-mcp")

# Initialize the MCP server
mcp = FastMCP("heat-transfer-calculator")

# Auto-detect unit conversion capability
UNIT_CONVERSION_ENABLED = False

try:
    from utils.unit_aware_decorator import make_tool_unit_aware, TOOL_MAPPINGS

    UNIT_CONVERSION_ENABLED = True
    logger.info("Unit conversion system detected and enabled")
except ImportError:
    logger.warning("Unit conversion system not available - using SI units only")

# Import all tools
from tools.ambient_conditions import get_ambient_conditions
from tools.fluid_properties import get_fluid_properties
from tools.material_properties import get_material_properties
from tools.convection_coefficient import calculate_convection_coefficient, calculate_two_phase_h
from tools.overall_heat_transfer import calculate_overall_heat_transfer_coefficient
from tools.surface_heat_transfer import calculate_surface_heat_transfer
from tools.heat_exchanger import calculate_heat_exchanger_performance
from tools.heat_duty import calculate_heat_duty
from tools.solar_radiation import calculate_solar_radiation_on_surface
from tools.ground_heat_loss import calculate_ground_heat_loss
from tools.buried_object_heat_loss import calculate_buried_object_heat_loss
from tools.hx_shell_side_h_kern import calculate_hx_shell_side_h_kern
from tools.estimate_hx_physical_dims import estimate_hx_physical_dims
from tools.tank_heat_loss import tank_heat_loss
from tools.extreme_conditions import extreme_conditions
from tools.pipe_heat_management import pipe_heat_management
from tools.parameter_optimization import parameter_optimization
from tools.calculate_pressure_drop import calculate_pressure_drop
from tools.plate_heat_exchanger_sizing import size_plate_heat_exchanger
from tools.size_shell_tube_heat_exchanger import size_shell_tube_heat_exchanger
from tools.size_double_pipe_heat_exchanger import size_double_pipe_heat_exchanger

# List of all tools
# Consolidated omnibus tools (primary)
OMNIBUS_TOOLS = [
    tank_heat_loss,
    extreme_conditions,
    pipe_heat_management,
    parameter_optimization,
]

# Supporting tools (helpers and type-specific sizing)
TOOLS = [
    get_ambient_conditions,
    get_fluid_properties,
    get_material_properties,
    calculate_convection_coefficient,
    calculate_two_phase_h,
    calculate_overall_heat_transfer_coefficient,
    calculate_surface_heat_transfer,
    calculate_heat_exchanger_performance,
    calculate_heat_duty,
    calculate_solar_radiation_on_surface,
    calculate_ground_heat_loss,
    calculate_buried_object_heat_loss,
    calculate_hx_shell_side_h_kern,
    estimate_hx_physical_dims,
    calculate_pressure_drop,
    # Type-specific HX sizing tools (thermal-hydraulic coupled)
    size_plate_heat_exchanger,
    size_shell_tube_heat_exchanger,
    size_double_pipe_heat_exchanger,
]

# Register all tools with automatic unit awareness if available
for tool in OMNIBUS_TOOLS + TOOLS:
    if UNIT_CONVERSION_ENABLED:
        # Apply unit-aware decorator
        tool_name = tool.__name__
        if tool_name in TOOL_MAPPINGS:
            unit_aware_tool = make_tool_unit_aware(tool)
            logger.info(f"Registered {tool_name} with unit conversion support")
        else:
            unit_aware_tool = tool
            logger.info(f"Registered {tool_name} (no unit mappings)")

        # Register with MCP
        mcp.tool()(unit_aware_tool)
    else:
        # Register original tool
        mcp.tool()(tool)  # type: ignore[arg-type]
        logger.info(f"Registered {tool.__name__} (SI units only)")

# Log information about available dependencies
from utils.import_helpers import HT_AVAILABLE, METEOSTAT_AVAILABLE


def log_server_capabilities():
    """Log server capabilities and unit support."""
    logger.info("=" * 60)
    logger.info("HEAT TRANSFER MCP SERVER STARTING")
    logger.info("=" * 60)

    # Core dependencies
    logger.info(f"HT library available: {HT_AVAILABLE}")
    logger.info(f"Meteostat library available: {METEOSTAT_AVAILABLE}")

    # Unit conversion status
    if UNIT_CONVERSION_ENABLED:
        logger.info("✓ UNIT CONVERSION: ENABLED")
        logger.info("  Supported units:")
        logger.info("    Temperature: °F, °C, K")
        logger.info("    Length: ft, in, m")
        logger.info("    Flow Rate: GPM, SCFM, lb/hr, kg/s")
        logger.info("    Pressure: psi, bar, Pa")
        logger.info("    Area: ft², m²")
        logger.info("    Power: BTU/hr, HP, W")
        logger.info("    Heat Transfer Coeff: BTU/(hr·ft²·°F), W/(m²·K)")
        logger.info("")
        logger.info("  Usage examples:")
        logger.info('    temperature="95 degF"')
        logger.info('    flow_rate="200 GPM"')
        logger.info('    pressure="14.7 psi"')
        logger.info('    dimensions={"diameter": "15 ft", "height": "26 ft"}')
    else:
        logger.info("⚠ UNIT CONVERSION: DISABLED")
        logger.info("  All parameters must be in SI units:")
        logger.info("    Temperature: K")
        logger.info("    Length: m")
        logger.info("    Flow Rate: kg/s")
        logger.info("    Pressure: Pa")
        logger.info("    Area: m²")
        logger.info("    Power: W")

    logger.info("=" * 60)
    logger.info(f"Server registered {len(OMNIBUS_TOOLS) + len(TOOLS)} tools successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    log_server_capabilities()

    # Start the server
    mcp.run()
