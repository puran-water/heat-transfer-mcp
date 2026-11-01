"""Runtime adapters for model-driven tool calls."""

from __future__ import annotations

import json
from types import UnionType
from typing import Any, Callable, Dict, Type, Union, get_args, get_origin

from puran_mcp_support.engineering.pydantic_base import validate_finite_recursive
from pydantic import BaseModel, ValidationError

from heat_transfer_mcp.errors import CalculationError, redact_paths
from heat_transfer_mcp.schemas.common import StructuredErrorModel, ToolOutput
from heat_transfer_mcp.units import convert_quantity, get_quantity_kind


def coerce_tool_input(input_model: Type[BaseModel], payload: Any) -> BaseModel:
    """Validate a payload after converting QuantityField-marked strings."""
    if isinstance(payload, input_model):
        return payload
    if isinstance(payload, BaseModel):
        raw = payload.model_dump()
    elif isinstance(payload, dict):
        raw = dict(payload)
    else:
        raise CalculationError(
            code="invalid_input_type",
            message=f"Expected dict or {input_model.__name__}; got {type(payload).__name__}",
        )

    try:
        converted = _convert_quantities(input_model, raw)
        return input_model.model_validate(converted)
    except ValidationError as exc:
        raise CalculationError(
            code="validation_error",
            message=str(exc),
            details={"errors": exc.errors(include_context=False, include_input=False)},
        ) from exc
    except Exception as exc:
        raise CalculationError(code="validation_error", message=str(exc)) from exc


def run_tool_handler(
    *,
    name: str,
    input_model: Type[BaseModel],
    output_model: Type[ToolOutput],
    tool_func: Callable[..., Any],
    payload: Any,
) -> ToolOutput:
    """Run a tool behind the schema, unit-conversion, and error boundary."""
    try:
        model = coerce_tool_input(input_model, payload)
        kwargs = model.model_dump(exclude_none=True)
        raw_result = tool_func(**kwargs)
        result = _normalize_result(raw_result)
        if _is_error_result(result):
            return _error_output(output_model, result)
        try:
            validate_finite_recursive(result)
        except ValueError as exc:
            error = CalculationError(code="non_finite_result", message=str(exc))
            return output_model(error=StructuredErrorModel.model_validate(error.to_dict()), result={})
        return output_model(result=redact_paths(result if isinstance(result, dict) else {"value": result}))
    except CalculationError as exc:
        return output_model(error=StructuredErrorModel.model_validate(exc.to_dict()), result={})
    except Exception as exc:  # defensive boundary for MCP-facing calls
        error = CalculationError(code="unexpected_error", message=str(exc))
        return output_model(error=StructuredErrorModel.model_validate(error.to_dict()), result={})


def _convert_quantities(input_model: Type[BaseModel], raw: Dict[str, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for name, value in raw.items():
        field = input_model.model_fields.get(name)
        if field is None:
            converted[name] = value
            continue
        kind = get_quantity_kind(field)
        if kind:
            quantity_value = _convert_quantity_field_value(value, kind, (field.json_schema_extra or {}).get("si_unit"))
            if quantity_value is not _UNCHANGED:
                converted[name] = quantity_value
                continue
        nested = _convert_nested_model_quantities(field.annotation, value)
        converted[name] = nested if nested is not _UNCHANGED else _convert_nested_quantity_like(name, value)
    return converted


_UNCHANGED = object()


def _convert_quantity_field_value(value: Any, kind: str, unit: str | None = None) -> Any:
    if isinstance(value, str):
        return convert_quantity(value, kind, target_unit=unit)
    if isinstance(value, list):
        converted_items = []
        changed = False
        for item in value:
            if isinstance(item, str):
                converted_items.append(convert_quantity(item, kind, target_unit=unit))
                changed = True
            else:
                converted_items.append(item)
        return converted_items if changed else _UNCHANGED
    return _UNCHANGED


def _convert_nested_model_quantities(annotation: Any, value: Any) -> Any:
    """Convert quantity strings inside nested Pydantic models before validation."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        for arg in get_args(annotation):
            converted = _convert_nested_model_quantities(arg, value)
            if converted is not _UNCHANGED:
                return converted
        return _UNCHANGED
    if origin in (list, tuple):
        args = get_args(annotation)
        if not args or not isinstance(value, list):
            return _UNCHANGED
        item_annotation = args[0]
        converted_items = []
        changed = False
        for item in value:
            converted = _convert_nested_model_quantities(item_annotation, item)
            if converted is _UNCHANGED:
                converted_items.append(item)
            else:
                converted_items.append(converted)
                changed = True
        return converted_items if changed else _UNCHANGED
    if isinstance(annotation, type) and issubclass(annotation, BaseModel) and isinstance(value, dict):
        return _convert_quantities(annotation, dict(value))
    return _UNCHANGED


def _convert_nested_quantity_like(name: str, value: Any) -> Any:
    """Best-effort nested conversion for common layer/dimension containers."""
    if name == "dimensions" and isinstance(value, dict):
        return {
            key: convert_quantity(item, "length") if isinstance(item, str) else item
            for key, item in value.items()
        }
    if name == "wall_layers" and isinstance(value, list):
        converted_layers = []
        for layer in value:
            if not isinstance(layer, dict):
                converted_layers.append(layer)
                continue
            item = dict(layer)
            if isinstance(item.get("thickness"), str):
                item["thickness"] = convert_quantity(item["thickness"], "length")
            for k_name in ("thermal_conductivity_k", "thermal_conductivity", "k"):
                if isinstance(item.get(k_name), str):
                    item[k_name] = convert_quantity(item[k_name], "thermal_conductivity")
            converted_layers.append(item)
        return converted_layers
    return value


def _is_error_result(result: Any) -> bool:
    return isinstance(result, dict) and ("error" in result or ("code" in result and "message" in result))


def _error_output(output_model: Type[ToolOutput], result: Dict[str, Any]) -> ToolOutput:
    code = str(result.get("code") or "calculation_error")
    message = result.get("message", result.get("error", "Tool returned an error"))
    details = _normalize_error_details(result)
    error_payload: Dict[str, Any] = {
        "code": code,
        "message": redact_paths(str(message)),
        "details": redact_paths(details),
    }
    if result.get("correlation_id"):
        error_payload["correlation_id"] = str(redact_paths(result["correlation_id"]))
    if result.get("field"):
        error_payload["field"] = str(redact_paths(result["field"]))
    if result.get("hint"):
        error_payload["hint"] = str(redact_paths(result["hint"]))
    return output_model(
        error=StructuredErrorModel.model_validate(error_payload),
        result={},
    )


def _normalize_error_details(result: Dict[str, Any]) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    raw_details = result.get("details")
    if isinstance(raw_details, dict):
        details.update(raw_details)
    elif raw_details is not None:
        details["details"] = raw_details

    envelope_keys = {"code", "message", "error", "correlation_id", "field", "hint", "details"}
    for key, value in result.items():
        if key not in envelope_keys:
            details[key] = value
    return details


def _normalize_result(raw_result: Any) -> Any:
    if isinstance(raw_result, str):
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            return {"value": raw_result}
    if isinstance(raw_result, BaseModel):
        return raw_result.model_dump()
    return raw_result
