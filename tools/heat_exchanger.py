"""
Heat exchanger performance/rating tool to analyze heat exchanger scenarios.

This module provides functionality to rate heat exchangers using
the effectiveness-NTU method via ht.hx.effectiveness_NTU_method.

Rating mode: Given HX geometry (U, A) and inlet conditions, find outlet temps and duty.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

from utils.import_helpers import HT_AVAILABLE
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.heat_exchanger")


def calculate_heat_exchanger_performance(
    hx_type: str = "shell_tube",
    flow_arrangement: str = None,
    hot_fluid_name: str = None,
    hot_fluid_flow_rate: float = None,
    hot_fluid_inlet_temp: float = None,
    cold_fluid_name: str = None,
    cold_fluid_flow_rate: float = None,
    cold_fluid_inlet_temp: float = None,
    overall_heat_transfer_coefficient_U: float = None,
    heat_transfer_area: float = None,
    hot_fluid_pressure: float = 101325.0,
    cold_fluid_pressure: float = 101325.0,
    n_shell_passes: int = 1,
) -> str:
    """Rate a heat exchanger using the effectiveness-NTU method.

    Uses ht.hx.effectiveness_NTU_method from the CalebBell/ht library for
    accurate calculations across multiple flow configurations.

    Rating mode: Given HX geometry (U, A) and inlet conditions, calculates
    outlet temperatures and actual heat duty.

    Args:
        hx_type: Type of heat exchanger (e.g., 'shell_tube', 'double_pipe', 'plate')
        flow_arrangement: Flow arrangement - one of:
            - 'counterflow': Pure counterflow
            - 'parallelflow' or 'parallel': Co-current flow
            - 'crossflow': Crossflow, both fluids unmixed
            - 'crossflow_mixed_Cmax': Crossflow, Cmax mixed
            - 'crossflow_mixed_Cmin': Crossflow, Cmin mixed
            - 'shell_tube': Shell-and-tube (TEMA E)
        hot_fluid_name: Name of the hot fluid
        hot_fluid_flow_rate: Mass flow rate of the hot fluid (kg/s)
        hot_fluid_inlet_temp: Inlet temperature of the hot fluid (K)
        cold_fluid_name: Name of the cold fluid
        cold_fluid_flow_rate: Mass flow rate of the cold fluid (kg/s)
        cold_fluid_inlet_temp: Inlet temperature of the cold fluid (K)
        overall_heat_transfer_coefficient_U: Overall heat transfer coefficient (W/m²K)
        heat_transfer_area: Total heat transfer area (m²)
        hot_fluid_pressure: Pressure of hot fluid (Pa)
        cold_fluid_pressure: Pressure of cold fluid (Pa)
        n_shell_passes: Number of shell passes for shell-tube HX (default 1)

    Returns:
        JSON string with heat exchanger performance results

    Raises:
        ImportError: If ht library is not available
    """
    try:
        # Require ht library for rating calculations
        if not HT_AVAILABLE:
            raise ImportError("ht library required for heat exchanger rating")

        from ht.hx import effectiveness_NTU_method, effectiveness_from_NTU
        from ht.core import LMTD

        # Validate required inputs
        required_inputs = [
            flow_arrangement,
            hot_fluid_name,
            hot_fluid_flow_rate,
            hot_fluid_inlet_temp,
            cold_fluid_name,
            cold_fluid_flow_rate,
            cold_fluid_inlet_temp,
            overall_heat_transfer_coefficient_U,
            heat_transfer_area,
        ]

        if None in required_inputs:
            return json.dumps(
                {
                    "error": "All required inputs must be provided for heat exchanger analysis.",
                    "required": [
                        "flow_arrangement",
                        "hot_fluid_name",
                        "hot_fluid_flow_rate",
                        "hot_fluid_inlet_temp",
                        "cold_fluid_name",
                        "cold_fluid_flow_rate",
                        "cold_fluid_inlet_temp",
                        "overall_heat_transfer_coefficient_U",
                        "heat_transfer_area",
                    ],
                }
            )

        # 1. Get fluid properties for hot fluid
        hot_fluid_props_json = get_fluid_properties(hot_fluid_name, hot_fluid_inlet_temp, hot_fluid_pressure)
        hot_fluid_props = json.loads(hot_fluid_props_json)

        if "error" in hot_fluid_props:
            return json.dumps({"error": f"Failed to get hot fluid properties: {hot_fluid_props['error']}"})

        # 2. Get fluid properties for cold fluid
        cold_fluid_props_json = get_fluid_properties(cold_fluid_name, cold_fluid_inlet_temp, cold_fluid_pressure)
        cold_fluid_props = json.loads(cold_fluid_props_json)

        if "error" in cold_fluid_props:
            return json.dumps({"error": f"Failed to get cold fluid properties: {cold_fluid_props['error']}"})

        # 3. Extract specific heat capacities
        hot_fluid_cp = hot_fluid_props.get("specific_heat_cp")
        cold_fluid_cp = cold_fluid_props.get("specific_heat_cp")

        if hot_fluid_cp is None or cold_fluid_cp is None:
            return json.dumps({"error": "Could not determine specific heat capacity for one or both fluids."})

        # 4. Map flow arrangement to ht.hx subtype
        flow_lower = flow_arrangement.lower()
        subtype_map = {
            "counterflow": "counterflow",
            "parallelflow": "parallel",
            "parallel": "parallel",
            "cocurrent": "parallel",
            "crossflow": "crossflow",
            "crossflow_unmixed": "crossflow",
            "crossflow_mixed_cmax": "crossflow, mixed Cmax",
            "crossflow_mixed_cmin": "crossflow, mixed Cmin",
            "shell_tube": "TEMA E",
            "tema_e": "TEMA E",
            "tema_j": "TEMA J",
            "tema_h": "TEMA H",
            "tema_g": "TEMA G",
        }
        subtype = subtype_map.get(flow_lower, "counterflow")

        # 5. Use ht.hx.effectiveness_NTU_method for rating calculation
        # This function solves for outlet temperatures when UA is known
        UA = overall_heat_transfer_coefficient_U * heat_transfer_area

        try:
            hx_results = effectiveness_NTU_method(
                mh=hot_fluid_flow_rate,
                mc=cold_fluid_flow_rate,
                Cph=hot_fluid_cp,
                Cpc=cold_fluid_cp,
                Thi=hot_fluid_inlet_temp,
                Tci=cold_fluid_inlet_temp,
                UA=UA,
                subtype=subtype,
                n_shell_tube=n_shell_passes if "TEMA" in subtype else None,
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": f"effectiveness_NTU_method failed: {str(e)}",
                    "hint": "Check that flow arrangement and inputs are valid",
                }
            )

        # Extract results from ht.hx solver
        heat_duty = hx_results.get("Q", 0)
        effectiveness = hx_results.get("effectiveness", 0)
        NTU = hx_results.get("NTU", 0)
        hot_fluid_outlet_temp = hx_results.get("Tho", hot_fluid_inlet_temp)
        cold_fluid_outlet_temp = hx_results.get("Tco", cold_fluid_inlet_temp)
        C_hot = hx_results.get("Ch", hot_fluid_flow_rate * hot_fluid_cp)
        C_cold = hx_results.get("Cc", cold_fluid_flow_rate * cold_fluid_cp)
        C_min = hx_results.get("Cmin", min(C_hot, C_cold))
        C_max = hx_results.get("Cmax", max(C_hot, C_cold))
        C_r = hx_results.get("Cr", C_min / C_max if C_max > 0 else 0)

        # 6. Calculate LMTD using ht.core.LMTD for verification
        lmtd = None
        try:
            is_counterflow = flow_lower in ("counterflow", "shell_tube", "tema_e", "tema_j", "tema_h", "tema_g")
            lmtd = LMTD(
                Thi=hot_fluid_inlet_temp,
                Tho=hot_fluid_outlet_temp,
                Tci=cold_fluid_inlet_temp,
                Tco=cold_fluid_outlet_temp,
                counterflow=is_counterflow,
            )
        except Exception as lmtd_error:
            logger.debug(f"Could not calculate LMTD: {lmtd_error}")

        # 7. Heat balance verification
        Q_from_UA = UA * lmtd if lmtd else None
        heat_balance_error_pct = None
        if Q_from_UA and heat_duty > 0:
            heat_balance_error_pct = abs(Q_from_UA - heat_duty) / heat_duty * 100

        # Create result
        result = {
            "heat_duty_W": heat_duty,
            "heat_duty_kW": heat_duty / 1000,
            "effectiveness": effectiveness,
            "NTU": NTU,
            "capacity_ratio_Cr": C_r,
            "C_hot_W_K": C_hot,
            "C_cold_W_K": C_cold,
            "C_min_W_K": C_min,
            "C_max_W_K": C_max,
            "LMTD_K": lmtd,
            "UA_W_K": UA,
            "temperatures": {
                "hot_inlet_K": hot_fluid_inlet_temp,
                "hot_inlet_C": hot_fluid_inlet_temp - 273.15,
                "hot_outlet_K": hot_fluid_outlet_temp,
                "hot_outlet_C": hot_fluid_outlet_temp - 273.15,
                "cold_inlet_K": cold_fluid_inlet_temp,
                "cold_inlet_C": cold_fluid_inlet_temp - 273.15,
                "cold_outlet_K": cold_fluid_outlet_temp,
                "cold_outlet_C": cold_fluid_outlet_temp - 273.15,
            },
            "hot_fluid": {
                "name": hot_fluid_name,
                "flow_rate_kg_s": hot_fluid_flow_rate,
                "specific_heat_J_kgK": hot_fluid_cp,
                "pressure_Pa": hot_fluid_pressure,
            },
            "cold_fluid": {
                "name": cold_fluid_name,
                "flow_rate_kg_s": cold_fluid_flow_rate,
                "specific_heat_J_kgK": cold_fluid_cp,
                "pressure_Pa": cold_fluid_pressure,
            },
            "heat_exchanger": {
                "type": hx_type,
                "flow_arrangement": flow_arrangement,
                "subtype_used": subtype,
                "U_W_m2K": overall_heat_transfer_coefficient_U,
                "area_m2": heat_transfer_area,
                "n_shell_passes": n_shell_passes if "TEMA" in subtype else None,
            },
            "verification": {
                "Q_from_LMTD_W": Q_from_UA,
                "heat_balance_error_pct": heat_balance_error_pct,
                "method": "ht.hx.effectiveness_NTU_method",
            },
        }

        return json.dumps(result)

    except ImportError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error in calculate_heat_exchanger_performance: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
