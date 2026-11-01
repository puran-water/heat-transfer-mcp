"""Sizing semantics shared by shell-tube tools."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ShellTubeSizingVerification:
    Q_design_W: float
    A_required_m2: float
    A_available_m2: float
    Q_available_W: float
    overdesign_ratio: float
    balance_satisfied: bool
    design_balance_error_pct: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "Q_design_W": self.Q_design_W,
            "Q_design_kW": self.Q_design_W / 1000.0,
            "A_required_m2": self.A_required_m2,
            "A_available_m2": self.A_available_m2,
            "Q_available_W": self.Q_available_W,
            "Q_available_kW": self.Q_available_W / 1000.0,
            "rated_capacity_W": self.Q_available_W,
            "rated_capacity_kW": self.Q_available_W / 1000.0,
            "overdesign_ratio": self.overdesign_ratio,
            "balance_satisfied": self.balance_satisfied,
            "design_balance_error_pct": self.design_balance_error_pct,
        }


def required_area(*, Q_design_W: float, U_W_m2K: float, lmtd_K: float, f_correction: float = 1.0) -> float:
    denominator = U_W_m2K * lmtd_K * f_correction
    if denominator <= 0:
        raise ValueError("U, LMTD, and F-factor must produce a positive thermal driving force")
    return abs(Q_design_W) / denominator


def available_capacity(*, U_W_m2K: float, A_available_m2: float, lmtd_K: float, f_correction: float = 1.0) -> float:
    return U_W_m2K * A_available_m2 * lmtd_K * f_correction


def verify_shell_tube_sizing(
    *,
    Q_design_W: float,
    U_W_m2K: float,
    A_available_m2: float,
    lmtd_K: float,
    f_correction: float,
    Q_hot_W: Optional[float],
    Q_cold_W: Optional[float],
    tolerance_pct: float = 5.0,
) -> ShellTubeSizingVerification:
    A_required = required_area(Q_design_W=Q_design_W, U_W_m2K=U_W_m2K, lmtd_K=lmtd_K, f_correction=f_correction)
    Q_available = available_capacity(U_W_m2K=U_W_m2K, A_available_m2=A_available_m2, lmtd_K=lmtd_K, f_correction=f_correction)
    overdesign = A_available_m2 / A_required if A_required > 0 else math.inf

    candidates = [abs(Q_design_W)]
    if Q_hot_W is not None:
        candidates.append(abs(Q_hot_W))
    if Q_cold_W is not None:
        candidates.append(abs(Q_cold_W))
    max_error = max(abs(candidate - abs(Q_design_W)) for candidate in candidates)
    error_pct = max_error / max(abs(Q_design_W), 1.0) * 100.0

    return ShellTubeSizingVerification(
        Q_design_W=abs(Q_design_W),
        A_required_m2=A_required,
        A_available_m2=A_available_m2,
        Q_available_W=Q_available,
        overdesign_ratio=overdesign,
        balance_satisfied=error_pct <= tolerance_pct,
        design_balance_error_pct=error_pct,
    )
