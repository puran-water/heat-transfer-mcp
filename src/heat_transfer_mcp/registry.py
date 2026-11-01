"""Tool registration primitives for the refactored server path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, Type, TypeVar

from pydantic import BaseModel

InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


@dataclass(frozen=True)
class ToolSpec(Generic[InputModelT, OutputModelT]):
    """Schema-bound handler metadata for one MCP tool."""

    name: str
    input_model: Type[InputModelT]
    output_model: Type[OutputModelT]
    handler: Callable[[InputModelT], OutputModelT]


class ToolRegistry:
    """In-memory registry used by future server wiring."""

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def all(self) -> Iterable[ToolSpec]:
        return tuple(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
