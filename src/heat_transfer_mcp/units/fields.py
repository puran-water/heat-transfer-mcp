"""Pydantic field helpers for quantity metadata."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field
from pydantic.fields import FieldInfo

from heat_transfer_mcp.units.kinds import QuantityKind


def QuantityField(
    default: Any = ...,
    *,
    kind: QuantityKind,
    unit: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a Pydantic field with quantity metadata.

    This replaces the old name-based unit decorator; conversion code can inspect
    field metadata instead of maintaining a parallel argument-name map.
    """
    extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    extra["quantity_kind"] = kind
    if unit:
        extra["si_unit"] = unit
    return Field(default, description=description, json_schema_extra=extra, **kwargs)


def get_quantity_kind(field: FieldInfo) -> Optional[str]:
    extra = field.json_schema_extra or {}
    if isinstance(extra, dict):
        kind = extra.get("quantity_kind")
        return str(kind) if kind else None
    return None
