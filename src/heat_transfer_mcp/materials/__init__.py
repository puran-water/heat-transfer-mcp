"""Material catalog resolver."""

from heat_transfer_mcp.materials.resolver import MaterialEntry, UnknownMaterialError, resolve_material

__all__ = ["MaterialEntry", "UnknownMaterialError", "resolve_material"]
