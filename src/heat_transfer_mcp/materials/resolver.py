"""Canonical material catalog resolver."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class MaterialEntry:
    name: str
    canonical_id: str
    aliases: List[str]
    k_W_mK: float
    rho_kg_m3: Optional[float]
    cp_J_kgK: Optional[float]
    emissivity: Any
    source: str
    applicable_temp_range_C: Optional[List[float]]
    requested_temperature_C: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


class UnknownMaterialError(ValueError):
    """Raised when a material name cannot be resolved."""

    def __init__(self, query: str, alternatives: List[str]) -> None:
        self.query = query
        self.alternatives = alternatives
        alt_text = ", ".join(alternatives) if alternatives else "none"
        super().__init__(f"Unknown material '{query}'. Closest alternatives: {alt_text}")


def resolve_material(name_or_alias: str, *, temperature_C: Optional[float] = None) -> MaterialEntry:
    """Resolve a material name or alias to a canonical catalog entry."""
    if not isinstance(name_or_alias, str) or not name_or_alias.strip():
        raise UnknownMaterialError(str(name_or_alias), _alternatives(str(name_or_alias)))

    catalog = _catalog()
    norm = _normalize(name_or_alias)
    raw = catalog["lookup"].get(norm)
    if raw is None:
        raise UnknownMaterialError(name_or_alias, _alternatives(name_or_alias))

    temp_range = raw.get("applicable_temp_range_C")
    warning_messages: List[str] = []
    if temperature_C is not None and temp_range:
        low, high = float(temp_range[0]), float(temp_range[1])
        if temperature_C < low or temperature_C > high:
            message = (
                f"Requested temperature {temperature_C:g} C is outside applicable range "
                f"{low:g} to {high:g} C for {raw['canonical_id']}"
            )
            raise ValueError(message)

    return MaterialEntry(
        name=str(raw["name"]),
        canonical_id=str(raw["canonical_id"]),
        aliases=list(raw.get("aliases") or []),
        k_W_mK=_resolve_k(raw.get("k_W_mK"), temperature_C),
        rho_kg_m3=_optional_float(raw.get("rho_kg_m3")),
        cp_J_kgK=_optional_float(raw.get("cp_J_kgK")),
        emissivity=raw.get("emissivity"),
        source=str(raw.get("source") or "materials.yaml"),
        applicable_temp_range_C=[float(temp_range[0]), float(temp_range[1])] if temp_range else None,
        requested_temperature_C=temperature_C,
        warnings=warning_messages,
    )


@lru_cache(maxsize=1)
def _catalog() -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "data" / "materials.yaml"
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    materials = payload.get("materials") or []
    lookup: Dict[str, Dict[str, Any]] = {}
    display_names: List[str] = []
    for entry in materials:
        names = [entry.get("canonical_id"), entry.get("name")]
        names.extend(entry.get("aliases") or [])
        display_names.extend(str(name) for name in names if name)
        for name in names:
            if name:
                lookup[_normalize(str(name))] = entry
    return {"materials": materials, "lookup": lookup, "display_names": display_names}


def _resolve_k(k_spec: Any, temperature_C: Optional[float]) -> float:
    if isinstance(k_spec, (int, float)):
        return float(k_spec)
    if isinstance(k_spec, dict):
        temps = k_spec.get("temperature_C")
        values = k_spec.get("values")
        if isinstance(temps, list) and isinstance(values, list) and len(temps) == len(values) and temps:
            if temperature_C is None:
                # Use the first tabulated value as nominal room-temperature value.
                return float(values[0])
            return _interpolate(float(temperature_C), [float(t) for t in temps], [float(v) for v in values])
    raise ValueError(f"Invalid k_W_mK specification: {k_spec!r}")


def _interpolate(x: float, xs: List[float], ys: List[float]) -> float:
    if x < xs[0] or x > xs[-1]:
        raise ValueError(f"Conductivity temperature {x} C outside tabulated range {xs[0]} to {xs[-1]} C")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def _alternatives(query: str) -> List[str]:
    catalog = _catalog()
    normalized_to_display: Dict[str, str] = {}
    for name in catalog["display_names"]:
        normalized_to_display.setdefault(_normalize(name), name)
    close = difflib.get_close_matches(_normalize(query), list(normalized_to_display), n=5, cutoff=0.35)
    return [normalized_to_display[item] for item in close]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _optional_float(value: Any) -> Optional[float]:
    return float(value) if value is not None else None
