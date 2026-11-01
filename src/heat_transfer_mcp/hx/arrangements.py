"""Heat-exchanger flow arrangement normalization and ht-library adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional


class FlowArrangement(str, Enum):
    COUNTERFLOW = "counterflow"
    PARALLEL = "parallel"
    CROSSFLOW = "crossflow"
    CROSSFLOW_MIXED_CMAX = "crossflow_mixed_cmax"
    CROSSFLOW_MIXED_CMIN = "crossflow_mixed_cmin"
    SHELL_TUBE = "shell_tube"


_ALIASES: Dict[str, FlowArrangement] = {
    "counterflow": FlowArrangement.COUNTERFLOW,
    "counter": FlowArrangement.COUNTERFLOW,
    "parallel": FlowArrangement.PARALLEL,
    "parallelflow": FlowArrangement.PARALLEL,
    "co_current": FlowArrangement.PARALLEL,
    "cocurrent": FlowArrangement.PARALLEL,
    "crossflow": FlowArrangement.CROSSFLOW,
    "crossflow_unmixed": FlowArrangement.CROSSFLOW,
    "crossflow_mixed_cmax": FlowArrangement.CROSSFLOW_MIXED_CMAX,
    "crossflow_mixed_cmin": FlowArrangement.CROSSFLOW_MIXED_CMIN,
    "shell_tube": FlowArrangement.SHELL_TUBE,
    "shell-and-tube": FlowArrangement.SHELL_TUBE,
    "shell_and_tube": FlowArrangement.SHELL_TUBE,
    "s&t": FlowArrangement.SHELL_TUBE,
    "tema_e": FlowArrangement.SHELL_TUBE,
}


@dataclass(frozen=True)
class HtArrangement:
    arrangement: FlowArrangement
    ht_subtype: str
    n_shell_tube: Optional[int] = None
    counterflow_lmtd: bool = True


_HT_SUBTYPES = {
    FlowArrangement.COUNTERFLOW: ("counterflow", True),
    FlowArrangement.PARALLEL: ("parallel", False),
    FlowArrangement.CROSSFLOW: ("crossflow", False),
    FlowArrangement.CROSSFLOW_MIXED_CMAX: ("crossflow, mixed Cmax", False),
    FlowArrangement.CROSSFLOW_MIXED_CMIN: ("crossflow, mixed Cmin", False),
    # Verified against the installed ht.hx.effectiveness_NTU_method API.
    FlowArrangement.SHELL_TUBE: ("S&T", True),
}


def supported_arrangements() -> Iterable[str]:
    return [item.value for item in FlowArrangement]


def normalize_arrangement(value: str) -> FlowArrangement:
    key = (value or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    supported = ", ".join(supported_arrangements())
    raise ValueError(f"Unsupported flow_arrangement '{value}'. Supported arrangements: {supported}")


def arrangement_for_ht(value: str, *, n_shell_passes: int = 1) -> HtArrangement:
    arrangement = normalize_arrangement(value)
    subtype, counterflow_lmtd = _HT_SUBTYPES[arrangement]
    n_shell_tube = int(n_shell_passes) if arrangement == FlowArrangement.SHELL_TUBE else None
    return HtArrangement(
        arrangement=arrangement,
        ht_subtype=subtype,
        n_shell_tube=n_shell_tube,
        counterflow_lmtd=counterflow_lmtd,
    )
