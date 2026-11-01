"""Foundation package for the heat-transfer MCP refactor."""

from heat_transfer_mcp._engineering_utils_bridge import ensure_engineering_utils_importable

ensure_engineering_utils_importable()

from heat_transfer_mcp.registry import ToolRegistry, ToolSpec
from heat_transfer_mcp.runtime import coerce_tool_input, run_tool_handler

__all__ = ["ToolRegistry", "ToolSpec", "coerce_tool_input", "run_tool_handler"]
