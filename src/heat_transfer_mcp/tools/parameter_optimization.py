"""
Omnibus tool: parameter_optimization

Generic parameter sweep and constrained optimization wrapper around other tools.
Supports grid sweeps on provided parameter lists, optional constraints, and
simple objective selection (minimize/maximize a result key path).
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional

from heat_transfer_mcp.utils.validation import ValidationError, ensure_list_nonempty, set_in_path

logger = logging.getLogger("heat-transfer-mcp.parameter_optimization")


DISALLOWED_INNER_TOOLS = {"parameter_optimization"}


def _get_from_path(data: Dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def parameter_optimization(
    tool_name: str,
    base_params: Dict[str, Any],
    sweep: Dict[str, List[Any]],
    objective_key: str,
    direction: str = "minimize",  # 'minimize' or 'maximize'
    constraints: Optional[List[Dict[str, Any]]] = None,
    top_n: int = 5,
) -> str:
    """Grid-search parameter sweep for a selected tool and objective.

    Args:
        tool_name: One of {'tank_heat_loss','pipe_heat_management','size_plate_heat_exchanger','size_shell_tube_heat_exchanger'}.
        base_params: Baseline parameter dict for the tool.
        sweep: Dict of parameter->list of values to sweep (Cartesian product).
        objective_key: Dot-path key in the tool's JSON result used as the objective value.
        direction: 'minimize' or 'maximize'.
        constraints: Optional list of dicts: {'key': 'path.to.key', 'op': '<=','>=','==','<','>','!=', 'value': 123}.
        top_n: Return top N results sorted by objective.

    Returns:
        JSON with the ranked results and the full evaluated set (may be large).
    """
    from heat_transfer_mcp.tool_specs import TOOL_SPECS_BY_NAME

    supported_tools = sorted(name for name in TOOL_SPECS_BY_NAME if name not in DISALLOWED_INNER_TOOLS)
    if tool_name in DISALLOWED_INNER_TOOLS or tool_name not in TOOL_SPECS_BY_NAME:
        return json.dumps({"error": f"Unsupported tool_name: {tool_name}", "supported_tools": supported_tools})
    tool_spec = TOOL_SPECS_BY_NAME[tool_name]

    # Validate sweep structure
    try:
        if direction not in {"minimize", "maximize"}:
            raise ValidationError("direction must be minimize or maximize")
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValidationError("top_n must be a positive integer")
        for constraint in constraints or []:
            if constraint.get("op") not in {"<=", ">=", "==", "!=", "<", ">"}:
                raise ValidationError("Unsupported constraint operator")
            if not constraint.get("key") or "value" not in constraint:
                raise ValidationError("Each constraint requires key and value")
        if not isinstance(sweep, dict) or len(sweep) == 0:
            raise ValidationError("sweep must be a non-empty dict of parameter lists")
        keys = list(sweep.keys())
        values_list = []
        for k in keys:
            ensure_list_nonempty(f"sweep['{k}']", sweep[k])
            values_list.append(sweep[k])
    except ValidationError as ve:
        return json.dumps({"error": str(ve)})
    combos = list(itertools.product(*values_list))

    evaluated: List[Dict[str, Any]] = []

    def constraint_ok(res: Dict[str, Any]) -> bool:
        if not constraints:
            return True
        for c in constraints:
            key = c.get("key")
            op = c.get("op")
            val = c.get("value")
            cur = _get_from_path(res, key) if key else None
            try:
                if op == "<=" and not (cur <= val):
                    return False
                if op == ">=" and not (cur >= val):
                    return False
                if op == "==" and not (cur == val):
                    return False
                if op == "!=" and not (cur != val):
                    return False
                if op == "<" and not (cur < val):
                    return False
                if op == ">" and not (cur > val):
                    return False
            except Exception:
                return False
        return True

    for combo in combos:
        params = deepcopy(base_params)
        invalid_path = False
        for k, v in zip(keys, combo):
            # Support nested keys with dot-paths
            if "." in k:
                try:
                    set_in_path(params, k, v)
                except ValidationError as ve:
                    evaluated.append({"params": params, "error": str(ve)})
                    invalid_path = True
                    break
            else:
                params[k] = v
        if invalid_path:
            continue
        try:
            output = tool_spec.handler(params)
            if output.error is not None:
                evaluated.append({"params": params, "error": output.error.model_dump()})
                continue
            res = output.result
        except Exception as e:
            evaluated.append({"params": params, "error": str(e)})
            continue
        if "error" in res:
            evaluated.append({"params": params, "error": res["error"]})
            continue

        if not constraint_ok(res):
            evaluated.append({"params": params, "result": res, "feasible": False})
            continue

        obj_val = _get_from_path(res, objective_key)
        if isinstance(obj_val, bool) or not isinstance(obj_val, (int, float)) or not math.isfinite(obj_val):
            evaluated.append({"params": params, "error": "Objective must be a finite number"})
            continue
        evaluated.append({"params": params, "result": res, "objective": obj_val, "feasible": True})

    # Rank results
    feas = [e for e in evaluated if e.get("feasible") and e.get("objective") is not None]
    reverse = True if direction.lower() == "maximize" else False
    feas_sorted = sorted(feas, key=lambda e: e.get("objective"), reverse=reverse)
    top = feas_sorted[: max(1, int(top_n))]

    payload = {
        "tool": tool_name,
        "objective_key": objective_key,
        "direction": direction,
        "top": top,
        "evaluated_count": len(evaluated),
        "feasible_count": len(feas_sorted),
    }
    candidate_errors = [item for item in evaluated if "error" in item]
    if candidate_errors:
        payload["candidate_errors"] = candidate_errors[:5]
    if not feas_sorted:
        payload["error"] = "No feasible candidates found"
    return json.dumps(payload, allow_nan=False)
