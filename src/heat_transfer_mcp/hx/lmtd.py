"""LMTD, F-factor, and effective mean-temperature-difference helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

from heat_transfer_mcp.hx.arrangements import FlowArrangement, normalize_arrangement


@dataclass(frozen=True)
class LMTDAnalysis:
    terminal_deltas_K: Dict[str, float]
    lmtd_K: Optional[float]
    f_correction: Optional[float]
    corrected_lmtd_K: Optional[float]
    effective_mtd_K: Optional[float]
    method: str
    note: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "terminal_deltas_K": self.terminal_deltas_K,
            "LMTD_K": self.lmtd_K,
            "F_correction": self.f_correction,
            "corrected_LMTD_K": self.corrected_lmtd_K,
            "effective_MTD_K": self.effective_mtd_K,
            "method": self.method,
            "note": self.note,
        }


def terminal_deltas(
    *,
    Thi: float,
    Tho: float,
    Tci: float,
    Tco: float,
    counterflow: bool,
) -> Dict[str, float]:
    if counterflow:
        return {
            "hot_in_minus_cold_out": Thi - Tco,
            "hot_out_minus_cold_in": Tho - Tci,
        }
    return {
        "hot_in_minus_cold_in": Thi - Tci,
        "hot_out_minus_cold_out": Tho - Tco,
    }


def true_lmtd(delta_1: float, delta_2: float) -> Optional[float]:
    if delta_1 <= 0 or delta_2 <= 0:
        return None
    if math.isclose(delta_1, delta_2, rel_tol=1e-12, abs_tol=1e-12):
        return delta_1
    # the end-ratio log as log1p of the relative difference (utils.hx_common.LMTD_LOG1P_RULE)
    return (delta_1 - delta_2) / math.log1p((delta_1 - delta_2) / delta_2)


def shell_tube_f_factor(*, Thi: float, Tho: float, Tci: float, Tco: float, shells: int = 1) -> Optional[float]:
    try:
        from ht.hx import F_LMTD_Fakheri

        value = float(F_LMTD_Fakheri(Tci=Tci, Tco=Tco, Thi=Thi, Tho=Tho, shells=shells))
        if math.isfinite(value) and 0.0 < value <= 1.0:
            return value
    except Exception:
        return None
    return None


def analyze_lmtd(
    arrangement: Union[str, FlowArrangement],
    *,
    Thi: float,
    Tho: float,
    Tci: float,
    Tco: float,
    Q: Optional[float] = None,
    UA: Optional[float] = None,
    shells: int = 1,
) -> LMTDAnalysis:
    arr = normalize_arrangement(arrangement.value if isinstance(arrangement, FlowArrangement) else arrangement)
    counterflow = arr in {FlowArrangement.COUNTERFLOW, FlowArrangement.SHELL_TUBE}
    deltas = terminal_deltas(Thi=Thi, Tho=Tho, Tci=Tci, Tco=Tco, counterflow=counterflow)
    lmtd = true_lmtd(*list(deltas.values()))
    effective = abs(Q) / UA if Q is not None and UA and UA > 0 else None

    if arr == FlowArrangement.SHELL_TUBE:
        f_correction = shell_tube_f_factor(Thi=Thi, Tho=Tho, Tci=Tci, Tco=Tco, shells=shells)
        corrected = lmtd * f_correction if lmtd is not None and f_correction is not None else None
        return LMTDAnalysis(
            terminal_deltas_K=deltas,
            lmtd_K=lmtd,
            f_correction=f_correction,
            corrected_lmtd_K=corrected,
            effective_mtd_K=effective or corrected,
            method="counterflow_lmtd_with_shell_tube_f_factor",
            note=None if f_correction is not None else "F-factor unavailable for this temperature program",
        )

    if arr in {FlowArrangement.COUNTERFLOW, FlowArrangement.PARALLEL}:
        return LMTDAnalysis(
            terminal_deltas_K=deltas,
            lmtd_K=lmtd,
            f_correction=1.0 if lmtd is not None else None,
            corrected_lmtd_K=lmtd,
            effective_mtd_K=effective or lmtd,
            method=f"{arr.value}_lmtd",
        )

    # Crossflow charts/correlations are not valid for every outlet temperature
    # pair produced by epsilon-NTU rating. Use Q/UA as the thermodynamically
    # consistent MTD when a chart correction is unsupported.
    return LMTDAnalysis(
        terminal_deltas_K=deltas,
        lmtd_K=lmtd,
        f_correction=None,
        corrected_lmtd_K=None,
        effective_mtd_K=effective,
        method="effective_mtd_from_Q_over_UA",
        note="Crossflow F-factor lookup unsupported for this rating result; using Q/UA effective MTD",
    )
