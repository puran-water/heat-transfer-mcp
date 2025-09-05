"""
Omnibus tool: heat_exchanger_design

Unified heat exchanger sizing and performance for process heating/cooling.
Integrates tank_heat_loss for total duty, feed heating requirements, and
provides area sizing with LMTD/effectiveness methods and optional physical
dimension estimates for shell-and-tube.

Consolidates: heat_exchanger + size_heat_exchanger_area + estimate_hx_physical_dims
+ hx_shell_side_h_kern + heat_duty
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Optional

from utils.import_helpers import HT_AVAILABLE

from tools.fluid_properties import get_fluid_properties
from tools.estimate_hx_physical_dims import estimate_hx_physical_dims
from tools.size_heat_exchanger_area import size_heat_exchanger_area
from tools.tank_heat_loss import tank_heat_loss

logger = logging.getLogger("heat-transfer-mcp.heat_exchanger_design")


def _cp_for(fluid: str, T_K: float, P_Pa: float = 101325.0) -> Optional[float]:
    try:
        props = json.loads(get_fluid_properties(fluid, T_K, P_Pa))
        if "error" in props:
            return None
        return props.get("specific_heat_cp")
    except Exception:
        return None


def _lmtd(Thi: float, Tho: float, Tci: float, Tco: float) -> float:
    dT1 = Thi - Tco
    dT2 = Tho - Tci
    if abs(dT1 - dT2) < 1e-9:
        return dT1  # equal temp differences
    # Avoid log of negative if user inputs inverted streams
    if dT1 <= 0 or dT2 <= 0:
        return float("nan")
    return (dT1 - dT2) / math.log(dT1 / dT2)


def heat_exchanger_design(
    # Integration with tank heat loss
    include_tank_loss: bool = False,
    tank_params: Optional[Dict[str, Any]] = None,
    # Feed/process heating requirements
    process_fluid: str = "water",
    process_mass_flow_kg_s: Optional[float] = None,
    process_inlet_temp_K: Optional[float] = None,
    process_target_temp_K: Optional[float] = None,
    # Heating medium settings
    heating_fluid: str = "hot_water",
    heating_inlet_temp_K: Optional[float] = None,
    heating_outlet_temp_K: Optional[float] = None,
    # Sizing inputs
    required_total_duty_W: Optional[float] = None,
    overall_U_W_m2K: Optional[float] = None,
    hx_type: str = "shell_tube",
    shells: int = 1,
    flow_arrangement: str = "counterflow",
    # Physical estimates
    estimate_physical: bool = False,
) -> str:
    """Size a heat exchanger for maintaining tank temperature and/or heating a feed.

    Args:
        include_tank_loss: If True, compute tank heat loss from tank_params via tools.tank_heat_loss and add to duty.
        tank_params: Dict of parameters accepted by tank_heat_loss.

        process_fluid: Name of the process fluid to be heated/cooled.
        process_mass_flow_kg_s: Mass flow rate of process fluid (kg/s).
        process_inlet_temp_K: Process fluid inlet temperature (K).
        process_target_temp_K: Desired outlet temp (K).

        heating_fluid: Name of heating fluid (informational).
        heating_inlet_temp_K: Heating fluid inlet temp (K).
        heating_outlet_temp_K: Heating fluid outlet temp (K) assumed/target.

        required_total_duty_W: If provided, use directly. Otherwise computed from tank loss + process heating.
        overall_U_W_m2K: If provided, use to size area via LMTD method. Otherwise default typical value.
        hx_type: 'shell_tube', 'plate', 'coil', 'double_pipe' (used for reporting; correlations TBD).
        shells: Number of shell passes (for LMTD correction when needed).
        flow_arrangement: 'counterflow' (default) or 'parallel'.

        estimate_physical: If True, call estimate_hx_physical_dims to provide rough mechanical dimensions.

    Returns:
        JSON with computed duty, UA/area sizing, and optional physical estimates.
    """
    try:
        # 1) Determine total required duty
        duty_W = 0.0
        tank_detail = None
        if include_tank_loss:
            if not tank_params or not isinstance(tank_params, dict):
                return json.dumps({"error": "tank_params dict is required when include_tank_loss=True"})
            tank_json = tank_heat_loss(**tank_params)
            tank_detail = json.loads(tank_json)
            if "error" in tank_detail:
                return json.dumps({"error": f"tank_heat_loss failed: {tank_detail['error']}"})
            duty_W += float(tank_detail.get("total_heat_loss_w", 0.0))

        # Process heating component Q = m*Cp*(T_out - T_in)
        process_Q = 0.0
        if (
            process_mass_flow_kg_s is not None
            and process_inlet_temp_K is not None
            and process_target_temp_K is not None
        ):
            cp = _cp_for(process_fluid, (process_inlet_temp_K + process_target_temp_K) / 2.0) or 4180.0
            process_Q = float(process_mass_flow_kg_s) * cp * (float(process_target_temp_K) - float(process_inlet_temp_K))
            duty_W += process_Q

        if required_total_duty_W is not None:
            duty_W = float(required_total_duty_W)

        # 2) Size area using LMTD with correction factor if possible
        Thi = float(heating_inlet_temp_K) if heating_inlet_temp_K is not None else None
        Tho = float(heating_outlet_temp_K) if heating_outlet_temp_K is not None else None
        Tci = float(process_inlet_temp_K) if process_inlet_temp_K is not None else None
        Tco = float(process_target_temp_K) if process_target_temp_K is not None else None

        lmt_d = None
        Ft = 1.0
        if None not in (Thi, Tho, Tci, Tco):
            lmt_d = _lmtd(Thi, Tho, Tci, Tco)
            if HT_AVAILABLE and not math.isnan(lmt_d):
                try:
                    import ht
                    from ht import F_LMTD_Fakheri
                    Ft = F_LMTD_Fakheri(Tci=Tci, Tco=Tco, Thi=Thi, Tho=Tho, shells=shells)
                except Exception as e:
                    logger.debug(f"F_LMTD correction fallback: {e}")
                    Ft = 1.0

        # Default U if not provided (typical clean water-water)
        U = float(overall_U_W_m2K) if overall_U_W_m2K is not None else 1000.0

        area_m2 = None
        UA = None
        if lmt_d is not None and lmt_d > 0.0:
            UA = duty_W / (lmt_d * Ft) if duty_W and lmt_d else None
            area_m2 = UA / U if UA is not None and U > 0 else None

        # 3) Optional mechanical estimate via existing tool
        mech = None
        if estimate_physical and area_m2:
            try:
                est_json = estimate_hx_physical_dims(required_area=area_m2, hx_type=hx_type)
                mech = json.loads(est_json)
            except Exception as e:
                mech = {"error": str(e)}

        result = {
            "total_required_duty_w": duty_W,
            "components": {
                "tank_loss_w": tank_detail.get("total_heat_loss_w") if tank_detail else 0.0,
                "process_heating_w": process_Q,
            },
            "lmt_d_k": lmt_d,
            "F_correction": Ft,
            "overall_U_w_m2k": U,
            "UA_w_k": UA,
            "required_area_m2": area_m2,
            "hx_type": hx_type,
            "shells": shells,
            "flow_arrangement": flow_arrangement,
            "mechanical_estimate": mech,
            "calculation_methods": {
                "LMTD": "Classical LMTD with Fakheri correction when ht available",
                "U_assumption": "Default 1000 W/m2-K if not provided",
                "tank_loss_integration": "tools.tank_heat_loss",
            },
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"heat_exchanger_design failed: {e}", exc_info=True)
        return json.dumps({"error": str(e)})

