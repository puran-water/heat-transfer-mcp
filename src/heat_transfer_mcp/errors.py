"""Structured calculation errors with safe path redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Dict, Optional
from uuid import uuid4

from puran_mcp_support.engineering.safe_paths import redact_paths as _shared_redact_paths

_TMP_PATH_RE = re.compile(r"/tmp/[^\s'\"<>]+")


def redact_paths(value: Any) -> Any:
    """Redact developer-local paths using the shared safe-path normalizer."""
    return _redact_tmp_paths(_shared_redact_paths(value))


def _redact_tmp_paths(value: Any) -> Any:
    if isinstance(value, str):
        return _TMP_PATH_RE.sub("<path>", value)
    if isinstance(value, dict):
        return {
            _redact_tmp_paths(key) if isinstance(key, str) else key: _redact_tmp_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_tmp_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_tmp_paths(item) for item in value)
    return value


@dataclass
class CalculationError(Exception):
    """Typed, serializable calculation error for tool outputs."""

    code: str
    message: str
    field: Optional[str] = None
    hint: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    correlation_id: str = dataclass_field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "message": redact_paths(str(self.message)),
            "correlation_id": self.correlation_id,
        }
        if self.field:
            payload["field"] = self.field
        if self.hint:
            payload["hint"] = redact_paths(str(self.hint))
        if self.details:
            payload["details"] = redact_paths(self.details)
        return payload
