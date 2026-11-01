"""Shared schema models."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class StructuredErrorModel(BaseModel):
    code: str
    message: str
    correlation_id: str = Field(default_factory=lambda: uuid4().hex)
    field: Optional[str] = None
    hint: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolOutput(BaseModel):
    """Base output envelope for schema-bound MCP tools."""

    model_config = ConfigDict(extra="allow")

    error: Optional[StructuredErrorModel] = None
    result: Dict[str, Any] = Field(...)


class ToolSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    input_model: Type[BaseModel]
    output_model: Type[ToolOutput]
