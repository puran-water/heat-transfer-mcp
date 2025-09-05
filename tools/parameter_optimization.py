"""
Omnibus tool: parameter_optimization

Generic parameter sweep and constrained optimization wrapper around other tools.
Supports grid sweeps on provided parameter lists, optional constraints, and
simple objective selection (minimize/maximize a result key path).
"""

from __future__ import annotations

import json
import logging
import itertools
from typing import Any, Dict, List, Optional, Callable

from tools.tank_heat_loss import tank_heat_loss
from tools.heat_exchanger_design import heat_exchanger_design
from tools.pipe_heat_management import pipe_heat_management

logger = logging.getLogger("heat-transfer-mcp.parameter_optimization")


TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "tank_heat_loss": tank_heat_loss,
    "heat_exchanger_design": heat_exchanger_design,
    "pipe_heat_management": pipe_heat_management,
}


def _get_from_path(data: Dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split('.'):
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
        tool_name: One of {'tank_heat_loss','heat_exchanger_design','pipe_heat_management'}.
        base_params: Baseline parameter dict for the tool.
        sweep: Dict of parameter->list of values to sweep (Cartesian product).
        objective_key: Dot-path key in the tool's JSON result used as the objective value.
        direction: 'minimize' or 'maximize'.
        constraints: Optional list of dicts: {'key': 'path.to.key', 'op': '<=','>=','==','<','>','!=', 'value': 123}.
        top_n: Return top N results sorted by objective.

    Returns:
        JSON with the ranked results and the full evaluated set (may be large).
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unsupported tool_name: {tool_name}"})
    tool = TOOL_REGISTRY[tool_name]

    # Build Cartesian product of sweep values
    keys = list(sweep.keys())
    values_list = [sweep[k] for k in keys]
    combos = list(itertools.product(*values_list))

    evaluated: List[Dict[str, Any]] = []

    def constraint_ok(res: Dict[str, Any]) -> bool:
        if not constraints:
            return True
        for c in constraints:
            key = c.get('key')
            op = c.get('op')
            val = c.get('value')
            cur = _get_from_path(res, key) if key else None
            try:
                if op == '<=' and not (cur <= val):
                    return False
                if op == '>=' and not (cur >= val):
                    return False
                if op == '==' and not (cur == val):
                    return False
                if op == '!=' and not (cur != val):
                    return False
                if op == '<' and not (cur < val):
                    return False
                if op == '>' and not (cur > val):
                    return False
            except Exception:
                return False
        return True

    for combo in combos:
        params = dict(base_params)
        for k, v in zip(keys, combo):
            params[k] = v
        try:
            raw = tool(**params)
            res = json.loads(raw)
        except Exception as e:
            evaluated.append({"params": params, "error": str(e)})
            continue
        if 'error' in res:
            evaluated.append({"params": params, "error": res['error']})
            continue

        if not constraint_ok(res):
            evaluated.append({"params": params, "result": res, "feasible": False})
            continue

        obj_val = _get_from_path(res, objective_key)
        evaluated.append({"params": params, "result": res, "objective": obj_val, "feasible": True})

    # Rank results
    feas = [e for e in evaluated if e.get("feasible") and e.get("objective") is not None]
    reverse = True if direction.lower() == 'maximize' else False
    feas_sorted = sorted(feas, key=lambda e: e.get("objective"), reverse=reverse)
    top = feas_sorted[: max(1, int(top_n))]

    return json.dumps({
        "tool": tool_name,
        "objective_key": objective_key,
        "direction": direction,
        "top": top,
        "evaluated_count": len(evaluated),
        "feasible_count": len(feas_sorted),
    })

