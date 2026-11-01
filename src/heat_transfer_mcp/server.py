"""
Enhanced MCP Server for heat transfer calculations with model-driven unit conversion.

This server registers tools through heat_transfer_mcp.ToolSpec handlers. Unit
conversion is driven by Pydantic field metadata, not stale argument-name maps.

Author: Claude AI
"""
import logging
from typing import Any, Dict
from fastmcp import FastMCP
from heat_transfer_mcp.tool_specs import ALL_TOOL_SPECS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('heat-transfer-mcp')
mcp = FastMCP('heat-transfer-calculator')
MODEL_DRIVEN_REGISTRY_ENABLED = True

def _make_mcp_tool(spec):
    """Create a FastMCP-callable wrapper around a ToolSpec handler.

    The payload parameter is annotated with the tool's Pydantic input model so
    FastMCP publishes the model's full JSON schema over MCP (issue #14). With a bare
    Dict[str, Any] annotation every tool advertised {"payload": object} and
    agents had to guess field names (wall_layers took four guesses against a
    live deployment). The handler still receives a plain dict --
    run_tool_handler owns validation, exactly as before; annotating buys
    discoverability plus an early, field-named extra_forbidden at the MCP layer.
    """

    def tool(payload: spec.input_model) -> Dict[str, Any]:
        data = payload.model_dump(exclude_unset=True) if hasattr(payload, 'model_dump') else payload
        return spec.handler(data).model_dump(exclude_none=True)
    tool.__name__ = spec.name
    tool.__doc__ = f'Model-driven handler for {spec.name}.'
    tool.__annotations__['payload'] = spec.input_model
    return tool
for spec in ALL_TOOL_SPECS:
    mcp.tool()(_make_mcp_tool(spec))
    logger.info('Registered %s with model-driven schema/unit handling', spec.name)
from heat_transfer_mcp.utils.import_helpers import HT_AVAILABLE, METEOSTAT_AVAILABLE

def log_server_capabilities():
    """Log server capabilities and unit support."""
    logger.info('=' * 60)
    logger.info('HEAT TRANSFER MCP SERVER STARTING')
    logger.info('=' * 60)
    logger.info(f'HT library available: {HT_AVAILABLE}')
    logger.info(f'Meteostat library available: {METEOSTAT_AVAILABLE}')
    logger.info('✓ MODEL-DRIVEN REGISTRY: ENABLED')
    logger.info('  Unit conversion source: Pydantic QuantityField metadata')
    logger.info('  Temperature modes: absolute degF/degC/K and delta_degF/delta_degC/K')
    logger.info('  HTC examples: BTU/(hr*ft²*F), Btu/hr-ft²-F, W/m²-K')
    logger.info('=' * 60)
    logger.info(f'Server registered {len(ALL_TOOL_SPECS)} tools successfully')
    logger.info('=' * 60)

def main():
    mcp.run(transport='stdio')
if __name__ == '__main__':
    main()
