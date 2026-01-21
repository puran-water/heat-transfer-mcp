"""
Dedicated Plate Heat Exchanger (PHE) sizing tool with thermal-hydraulic coupling.

This module provides accurate PHE sizing using upstream Caleb Bell ht/fluids libraries
with proper coupling between thermal performance (U-value) and hydraulic performance
(pressure drop) through velocity and Reynolds number dependencies.

Key features:
- Uses PlateExchanger geometry class for consistent derived values
- Enforces odd plate counts for symmetric channel allocation
- Pairs Nu and friction correlations consistently
- Includes port/manifold losses in pressure drop
- Provides iterative solver for thermal-hydraulic optimization
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Union

from utils.import_helpers import HT_AVAILABLE, FLUIDS_AVAILABLE
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.plate_heat_exchanger_sizing")


def size_plate_heat_exchanger(
    # Duty specification (provide heat_duty_W OR temperatures to calculate)
    heat_duty_W: Optional[float] = None,
    # Temperature specs (used if heat_duty_W not provided)
    hot_inlet_temp_K: Optional[float] = None,
    hot_outlet_temp_K: Optional[float] = None,
    cold_inlet_temp_K: Optional[float] = None,
    cold_outlet_temp_K: Optional[float] = None,
    # Flow rates (required)
    hot_mass_flow_kg_s: Optional[float] = None,
    cold_mass_flow_kg_s: Optional[float] = None,
    # Fluids
    hot_fluid: str = "water",
    cold_fluid: str = "water",
    hot_fluid_pressure_Pa: float = 101325.0,
    cold_fluid_pressure_Pa: float = 101325.0,
    # PHE geometry specification
    plate_amplitude_m: Optional[float] = None,
    plate_wavelength_m: Optional[float] = None,
    plate_width_m: Optional[float] = None,
    plate_length_m: Optional[float] = None,
    port_diameter_m: Optional[float] = None,
    plate_thickness_m: float = 0.0005,  # 0.5 mm default
    plate_conductivity_W_mK: float = 16.0,  # Stainless steel default
    chevron_angle_deg: float = 45.0,
    # Target plate count (if None, will be solved)
    n_plates: Optional[int] = None,
    # Configuration
    passes_hot: int = 1,
    passes_cold: int = 1,
    flow_arrangement: str = "counterflow",
    # Constraints for iterative solver
    max_pressure_drop_kPa: Optional[float] = None,
    min_plates: int = 5,
    max_plates: int = 201,
    # Fouling
    fouling_factor_hot_m2K_W: float = 0.0,
    fouling_factor_cold_m2K_W: float = 0.0,
    # Correlation selection (must pair Nu and friction consistently)
    correlation: str = "Martin_VDI",
    # Parameter sweep for sensitivity analysis
    sweep_max_pressure_drop_kPa: Optional[List[float]] = None,
    # Multi-parameter auto-optimization
    auto_optimize: bool = False,
    optimize_for: str = "area",  # "area", "cost", "dP" - objective to minimize
    strict: bool = False,
) -> str:
    """Size a plate heat exchanger with coupled thermal-hydraulic design.

    This tool provides accurate PHE sizing using upstream ht/fluids libraries
    with proper thermal-hydraulic coupling. The area, U-value, and pressure drop
    are all interdependent through velocity and Reynolds number.

    Args:
        heat_duty_W: Target heat transfer rate (W). If not provided, calculated from temps.

        hot_inlet_temp_K: Hot fluid inlet temperature (K)
        hot_outlet_temp_K: Hot fluid outlet temperature (K)
        cold_inlet_temp_K: Cold fluid inlet temperature (K)
        cold_outlet_temp_K: Cold fluid outlet temperature (K)

        hot_mass_flow_kg_s: Hot fluid mass flow rate (kg/s)
        cold_mass_flow_kg_s: Cold fluid mass flow rate (kg/s)

        hot_fluid: Hot fluid name (default 'water')
        cold_fluid: Cold fluid name (default 'water')
        hot_fluid_pressure_Pa: Hot fluid pressure (Pa)
        cold_fluid_pressure_Pa: Cold fluid pressure (Pa)

        plate_amplitude_m: Half wave height of corrugations (m). Typical: 1-3 mm.
        plate_wavelength_m: Distance between ridges (m). Typical: 6-12 mm.
        plate_width_m: Plate width between gaskets (m)
        plate_length_m: Port-to-port length (m)
        port_diameter_m: Port diameter (m)
        plate_thickness_m: Plate metal thickness (m). Default 0.5 mm.
        plate_conductivity_W_mK: Plate material thermal conductivity (W/m-K). Default 16 (SS).
        chevron_angle_deg: Chevron angle (degrees). Typical: 30-65.

        n_plates: Target number of plates. If None, solver finds optimal.

        passes_hot: Number of hot-side passes (default 1)
        passes_cold: Number of cold-side passes (default 1)
        flow_arrangement: 'counterflow' (default) or 'parallel'

        max_pressure_drop_kPa: Maximum allowable pressure drop (kPa). Used by solver.
        min_plates: Minimum plates for solver (default 5)
        max_plates: Maximum plates for solver (default 201)

        fouling_factor_hot_m2K_W: Hot side fouling resistance (m²K/W)
        fouling_factor_cold_m2K_W: Cold side fouling resistance (m²K/W)

        correlation: Correlation to use. Options:
            - 'Kumar': APV data-based (chevron 30-65°, Re 0.1-10000)
            - 'Martin_1999': Martin 1999 (chevron 0-80°, Re 200-10000)
            - 'Martin_VDI': Martin VDI revision (default)
            - 'Muley_Manglik': Includes enlargement factor (chevron 30-60°, Re > 1000)
        sweep_max_pressure_drop_kPa: List of max dP values for parameter sweep.
            When provided, returns sizing results for each dP constraint to show
            the trade-off between allowable pressure drop and exchanger size/cost.
            Example: [20, 30, 50, 75, 100] to see impact of dP on plates/area.
        auto_optimize: If True, search across chevron angles, pass configs, and correlations
            to find optimal feasible design. Returns best configuration and ranked candidates.
        optimize_for: Objective to minimize when auto_optimize=True. Options:
            - 'area': Minimize heat transfer area (default)
            - 'cost': Minimize area * dP^0.5 (proxy for cost)
            - 'dP': Minimize maximum pressure drop
        strict: If True, fail if ht/fluids libraries unavailable

    Returns:
        JSON with comprehensive sizing results including:
        - duty_kW, LMTD_K, effectiveness, NTU
        - geometry: plates, channels, area, hydraulic diameter
        - thermal: U, h_hot, h_cold, Re_hot, Re_cold, correlation
        - hydraulic: velocity, pressure_drop, friction_factor for both sides
        - temperatures: inlet/outlet for both sides, terminal temp differences
    """
    try:
        # Validate library availability
        if not FLUIDS_AVAILABLE:
            if strict:
                return json.dumps({"error": "fluids library required with strict=True"})
            return json.dumps(
                {
                    "error": "fluids library not available for PHE sizing",
                    "suggestion": "Install with: pip install fluids>=1.0.0",
                }
            )

        if not HT_AVAILABLE:
            if strict:
                return json.dumps({"error": "ht library required with strict=True"})
            return json.dumps(
                {"error": "ht library not available for PHE sizing", "suggestion": "Install with: pip install ht>=1.2.0"}
            )

        # Handle auto-optimization: search across multiple design parameters
        if auto_optimize:
            # Search space for PHE optimization
            search_chevron_angles = [30, 45, 60]
            search_pass_configs = [(1, 1), (1, 2), (2, 1), (2, 2)]
            search_correlations = ["Martin_VDI", "Kumar"]

            all_candidates = []
            for chevron in search_chevron_angles:
                for passes_h, passes_c in search_pass_configs:
                    for corr in search_correlations:
                        try:
                            result_json = size_plate_heat_exchanger(
                                heat_duty_W=heat_duty_W,
                                hot_inlet_temp_K=hot_inlet_temp_K,
                                hot_outlet_temp_K=hot_outlet_temp_K,
                                cold_inlet_temp_K=cold_inlet_temp_K,
                                cold_outlet_temp_K=cold_outlet_temp_K,
                                hot_mass_flow_kg_s=hot_mass_flow_kg_s,
                                cold_mass_flow_kg_s=cold_mass_flow_kg_s,
                                hot_fluid=hot_fluid,
                                cold_fluid=cold_fluid,
                                hot_fluid_pressure_Pa=hot_fluid_pressure_Pa,
                                cold_fluid_pressure_Pa=cold_fluid_pressure_Pa,
                                plate_amplitude_m=plate_amplitude_m,
                                plate_wavelength_m=plate_wavelength_m,
                                plate_width_m=plate_width_m,
                                plate_length_m=plate_length_m,
                                port_diameter_m=port_diameter_m,
                                plate_thickness_m=plate_thickness_m,
                                plate_conductivity_W_mK=plate_conductivity_W_mK,
                                chevron_angle_deg=chevron,
                                n_plates=n_plates,
                                passes_hot=passes_h,
                                passes_cold=passes_c,
                                flow_arrangement=flow_arrangement,
                                max_pressure_drop_kPa=max_pressure_drop_kPa,
                                min_plates=min_plates,
                                max_plates=max_plates,
                                fouling_factor_hot_m2K_W=fouling_factor_hot_m2K_W,
                                fouling_factor_cold_m2K_W=fouling_factor_cold_m2K_W,
                                correlation=corr,
                                sweep_max_pressure_drop_kPa=None,
                                auto_optimize=False,  # Don't recurse
                                optimize_for=optimize_for,
                                strict=strict,
                            )
                            result = json.loads(result_json)

                            if "error" not in result:
                                # Calculate score based on objective
                                area = result["geometry"]["area_required_m2"]
                                dP_max = max(
                                    result["hydraulic"]["pressure_drop_hot_kPa"], result["hydraulic"]["pressure_drop_cold_kPa"]
                                )

                                if optimize_for == "area":
                                    score = area
                                elif optimize_for == "cost":
                                    score = area * math.sqrt(dP_max + 1)
                                elif optimize_for == "dP":
                                    score = dP_max
                                else:
                                    score = area

                                all_candidates.append(
                                    {
                                        "config": {
                                            "chevron_angle_deg": chevron,
                                            "passes_hot": passes_h,
                                            "passes_cold": passes_c,
                                            "correlation": corr,
                                        },
                                        "score": score,
                                        "plates": result["geometry"]["plates"],
                                        "area_required_m2": result["geometry"]["area_required_m2"],
                                        "U_W_m2K": result["thermal"]["U_W_m2K"],
                                        "Re_hot": result["thermal"]["Re_hot"],
                                        "Re_cold": result["thermal"]["Re_cold"],
                                        "dP_hot_kPa": result["hydraulic"]["pressure_drop_hot_kPa"],
                                        "dP_cold_kPa": result["hydraulic"]["pressure_drop_cold_kPa"],
                                        "full_result": result,
                                    }
                                )
                        except Exception as e:
                            logger.debug(
                                f"Optimization candidate failed: chevron={chevron}, passes=({passes_h},{passes_c}), corr={corr}: {e}"
                            )
                            continue

            if all_candidates:
                # Sort by score (minimize)
                all_candidates.sort(key=lambda x: x["score"])
                best = all_candidates[0]

                return json.dumps(
                    {
                        "optimization_mode": True,
                        "optimize_for": optimize_for,
                        "best_configuration": best["config"],
                        "best_result": best["full_result"],
                        "candidates_evaluated": len(all_candidates),
                        "top_5_candidates": [
                            {
                                "config": c["config"],
                                "score": c["score"],
                                "plates": c["plates"],
                                "area_m2": c["area_required_m2"],
                                "U_W_m2K": c["U_W_m2K"],
                                "Re_hot": c["Re_hot"],
                                "Re_cold": c["Re_cold"],
                                "dP_hot_kPa": c["dP_hot_kPa"],
                                "dP_cold_kPa": c["dP_cold_kPa"],
                            }
                            for c in all_candidates[:5]
                        ],
                        "search_space": {
                            "chevron_angle_deg": search_chevron_angles,
                            "pass_configs": search_pass_configs,
                            "correlations": search_correlations,
                        },
                    }
                )
            else:
                return json.dumps(
                    {
                        "error": "No feasible configuration found in optimization search",
                        "search_space": {
                            "chevron_angle_deg": search_chevron_angles,
                            "pass_configs": search_pass_configs,
                            "correlations": search_correlations,
                        },
                        "suggestion": "Try relaxing max_pressure_drop_kPa constraint or adjusting plate geometry",
                    }
                )

        # Handle parameter sweep if requested
        if sweep_max_pressure_drop_kPa is not None and len(sweep_max_pressure_drop_kPa) > 0:
            sweep_results = []
            for dP_limit in sorted(sweep_max_pressure_drop_kPa):
                # Recursive call with single dP value
                single_result_json = size_plate_heat_exchanger(
                    heat_duty_W=heat_duty_W,
                    hot_inlet_temp_K=hot_inlet_temp_K,
                    hot_outlet_temp_K=hot_outlet_temp_K,
                    cold_inlet_temp_K=cold_inlet_temp_K,
                    cold_outlet_temp_K=cold_outlet_temp_K,
                    hot_mass_flow_kg_s=hot_mass_flow_kg_s,
                    cold_mass_flow_kg_s=cold_mass_flow_kg_s,
                    hot_fluid=hot_fluid,
                    cold_fluid=cold_fluid,
                    hot_fluid_pressure_Pa=hot_fluid_pressure_Pa,
                    cold_fluid_pressure_Pa=cold_fluid_pressure_Pa,
                    plate_amplitude_m=plate_amplitude_m,
                    plate_wavelength_m=plate_wavelength_m,
                    plate_width_m=plate_width_m,
                    plate_length_m=plate_length_m,
                    port_diameter_m=port_diameter_m,
                    plate_thickness_m=plate_thickness_m,
                    plate_conductivity_W_mK=plate_conductivity_W_mK,
                    chevron_angle_deg=chevron_angle_deg,
                    n_plates=n_plates,
                    passes_hot=passes_hot,
                    passes_cold=passes_cold,
                    flow_arrangement=flow_arrangement,
                    max_pressure_drop_kPa=dP_limit,
                    min_plates=min_plates,
                    max_plates=max_plates,
                    fouling_factor_hot_m2K_W=fouling_factor_hot_m2K_W,
                    fouling_factor_cold_m2K_W=fouling_factor_cold_m2K_W,
                    correlation=correlation,
                    sweep_max_pressure_drop_kPa=None,  # Don't recurse
                    strict=strict,
                )
                single_result = json.loads(single_result_json)

                # Extract key metrics for sweep summary
                if "error" not in single_result:
                    sweep_results.append(
                        {
                            "max_dP_constraint_kPa": dP_limit,
                            "plates": single_result["geometry"]["plates"],
                            "area_available_m2": single_result["geometry"]["total_area_m2"],
                            "area_required_m2": single_result["geometry"]["area_required_m2"],
                            "area_margin_pct": single_result["geometry"]["area_margin_pct"],
                            "U_W_m2K": single_result["thermal"]["U_W_m2K"],
                            "dP_hot_kPa": single_result["hydraulic"]["pressure_drop_hot_kPa"],
                            "dP_cold_kPa": single_result["hydraulic"]["pressure_drop_cold_kPa"],
                            "velocity_hot_m_s": single_result["hydraulic"]["velocity_hot_m_s"],
                            "velocity_cold_m_s": single_result["hydraulic"]["velocity_cold_m_s"],
                            "Re_hot": single_result["thermal"]["Re_hot"],
                            "Re_cold": single_result["thermal"]["Re_cold"],
                        }
                    )
                else:
                    sweep_results.append(
                        {
                            "max_dP_constraint_kPa": dP_limit,
                            "error": single_result["error"],
                        }
                    )

            # Also run without constraint to show minimum thermal requirement
            unconstrained_json = size_plate_heat_exchanger(
                heat_duty_W=heat_duty_W,
                hot_inlet_temp_K=hot_inlet_temp_K,
                hot_outlet_temp_K=hot_outlet_temp_K,
                cold_inlet_temp_K=cold_inlet_temp_K,
                cold_outlet_temp_K=cold_outlet_temp_K,
                hot_mass_flow_kg_s=hot_mass_flow_kg_s,
                cold_mass_flow_kg_s=cold_mass_flow_kg_s,
                hot_fluid=hot_fluid,
                cold_fluid=cold_fluid,
                hot_fluid_pressure_Pa=hot_fluid_pressure_Pa,
                cold_fluid_pressure_Pa=cold_fluid_pressure_Pa,
                plate_amplitude_m=plate_amplitude_m,
                plate_wavelength_m=plate_wavelength_m,
                plate_width_m=plate_width_m,
                plate_length_m=plate_length_m,
                port_diameter_m=port_diameter_m,
                plate_thickness_m=plate_thickness_m,
                plate_conductivity_W_mK=plate_conductivity_W_mK,
                chevron_angle_deg=chevron_angle_deg,
                n_plates=n_plates,
                passes_hot=passes_hot,
                passes_cold=passes_cold,
                flow_arrangement=flow_arrangement,
                max_pressure_drop_kPa=None,  # No constraint
                min_plates=min_plates,
                max_plates=max_plates,
                fouling_factor_hot_m2K_W=fouling_factor_hot_m2K_W,
                fouling_factor_cold_m2K_W=fouling_factor_cold_m2K_W,
                correlation=correlation,
                sweep_max_pressure_drop_kPa=None,
                strict=strict,
            )
            unconstrained = json.loads(unconstrained_json)

            return json.dumps(
                {
                    "sweep_type": "max_pressure_drop_kPa",
                    "sweep_values": sorted(sweep_max_pressure_drop_kPa),
                    "results": sweep_results,
                    "unconstrained_minimum": (
                        {
                            "plates": unconstrained.get("geometry", {}).get("plates"),
                            "area_required_m2": unconstrained.get("geometry", {}).get("area_required_m2"),
                            "U_W_m2K": unconstrained.get("thermal", {}).get("U_W_m2K"),
                            "dP_hot_kPa": unconstrained.get("hydraulic", {}).get("pressure_drop_hot_kPa"),
                            "dP_cold_kPa": unconstrained.get("hydraulic", {}).get("pressure_drop_cold_kPa"),
                            "note": "Minimum plates for thermal duty (no dP constraint)",
                        }
                        if "error" not in unconstrained
                        else {"error": unconstrained["error"]}
                    ),
                    "analysis_notes": [
                        "Lower dP constraint -> more plates -> lower velocity -> lower U -> larger area required",
                        "Higher dP constraint -> fewer plates -> higher velocity -> higher U -> smaller area required",
                        "Q = U * A_required * LMTD is satisfied for all cases",
                        "A_available may exceed A_required due to discrete plate counts",
                    ],
                }
            )

        # Validate required inputs
        if hot_mass_flow_kg_s is None or hot_mass_flow_kg_s <= 0:
            return json.dumps({"error": "hot_mass_flow_kg_s must be positive"})
        if cold_mass_flow_kg_s is None or cold_mass_flow_kg_s <= 0:
            return json.dumps({"error": "cold_mass_flow_kg_s must be positive"})

        # Validate correlation
        valid_correlations = ["Kumar", "Martin_1999", "Martin_VDI", "Muley_Manglik"]
        if correlation not in valid_correlations:
            return json.dumps({"error": f"Invalid correlation: {correlation}. Valid options: {valid_correlations}"})

        # Validate plate geometry
        if plate_amplitude_m is None or plate_amplitude_m <= 0:
            return json.dumps({"error": "plate_amplitude_m must be positive (typical: 1-3 mm)"})
        if plate_wavelength_m is None or plate_wavelength_m <= 0:
            return json.dumps({"error": "plate_wavelength_m must be positive (typical: 6-12 mm)"})
        if plate_width_m is None or plate_width_m <= 0:
            return json.dumps({"error": "plate_width_m must be positive"})
        if plate_length_m is None or plate_length_m <= 0:
            return json.dumps({"error": "plate_length_m must be positive"})

        # Validate temperatures and determine duty
        temps_provided = [hot_inlet_temp_K, hot_outlet_temp_K, cold_inlet_temp_K, cold_outlet_temp_K]
        temps_count = sum(1 for t in temps_provided if t is not None)

        if heat_duty_W is None:
            # Need all 4 temperatures to calculate duty
            if temps_count < 4:
                return json.dumps({"error": "Either heat_duty_W or all four temperatures must be provided"})
        else:
            # With duty provided, need at least 2 inlet temps (can calculate both outlets from Q = m*Cp*dT)
            if hot_inlet_temp_K is None or cold_inlet_temp_K is None:
                return json.dumps({"error": "Both inlet temperatures required when heat_duty_W is provided"})

        # Get fluid properties
        hot_bulk_temp = (hot_inlet_temp_K + (hot_outlet_temp_K or hot_inlet_temp_K)) / 2
        cold_bulk_temp = (cold_inlet_temp_K + (cold_outlet_temp_K or cold_inlet_temp_K)) / 2

        hot_props_json = get_fluid_properties(hot_fluid, hot_bulk_temp, hot_fluid_pressure_Pa, strict=strict)
        hot_props = json.loads(hot_props_json)
        if "error" in hot_props:
            return json.dumps({"error": f"Hot fluid properties error: {hot_props['error']}"})

        cold_props_json = get_fluid_properties(cold_fluid, cold_bulk_temp, cold_fluid_pressure_Pa, strict=strict)
        cold_props = json.loads(cold_props_json)
        if "error" in cold_props:
            return json.dumps({"error": f"Cold fluid properties error: {cold_props['error']}"})

        # Extract properties
        rho_hot = hot_props.get("density")
        mu_hot = hot_props.get("dynamic_viscosity")
        k_hot = hot_props.get("thermal_conductivity")
        cp_hot = hot_props.get("specific_heat_cp")
        Pr_hot = hot_props.get("prandtl_number")

        rho_cold = cold_props.get("density")
        mu_cold = cold_props.get("dynamic_viscosity")
        k_cold = cold_props.get("thermal_conductivity")
        cp_cold = cold_props.get("specific_heat_cp")
        Pr_cold = cold_props.get("prandtl_number")

        if None in [rho_hot, mu_hot, k_hot, cp_hot, rho_cold, mu_cold, k_cold, cp_cold]:
            return json.dumps({"error": "Missing critical fluid properties"})

        # Calculate Prandtl if not available
        if Pr_hot is None:
            Pr_hot = mu_hot * cp_hot / k_hot
        if Pr_cold is None:
            Pr_cold = mu_cold * cp_cold / k_cold

        # Calculate or verify heat duty
        if heat_duty_W is None:
            # Calculate from hot side
            heat_duty_W = hot_mass_flow_kg_s * cp_hot * (hot_inlet_temp_K - hot_outlet_temp_K)
            # Verify cold side is consistent
            Q_cold = cold_mass_flow_kg_s * cp_cold * (cold_outlet_temp_K - cold_inlet_temp_K)
            if abs(heat_duty_W - Q_cold) / max(abs(heat_duty_W), 1) > 0.05:
                logger.warning(f"Energy imbalance: Q_hot={heat_duty_W:.1f}W, Q_cold={Q_cold:.1f}W")
        else:
            # Calculate missing outlet temp
            if hot_outlet_temp_K is None:
                hot_outlet_temp_K = hot_inlet_temp_K - heat_duty_W / (hot_mass_flow_kg_s * cp_hot)
            if cold_outlet_temp_K is None:
                cold_outlet_temp_K = cold_inlet_temp_K + heat_duty_W / (cold_mass_flow_kg_s * cp_cold)

        # Calculate LMTD using ht.core.LMTD
        from ht.core import LMTD as ht_LMTD

        # Determine if counterflow or parallel
        is_counterflow = flow_arrangement.lower() == "counterflow"

        # Check for temperature crossover before calling library
        if is_counterflow:
            dT1 = hot_inlet_temp_K - cold_outlet_temp_K
            dT2 = hot_outlet_temp_K - cold_inlet_temp_K
        else:  # parallel
            dT1 = hot_inlet_temp_K - cold_inlet_temp_K
            dT2 = hot_outlet_temp_K - cold_outlet_temp_K

        if dT1 <= 0 or dT2 <= 0:
            return json.dumps({"error": "Temperature crossover: LMTD undefined", "details": {"dT1": dT1, "dT2": dT2}})

        LMTD = ht_LMTD(
            Thi=hot_inlet_temp_K,
            Tho=hot_outlet_temp_K,
            Tci=cold_inlet_temp_K,
            Tco=cold_outlet_temp_K,
            counterflow=is_counterflow,
        )

        if LMTD < 0.5:
            logger.warning(f"Very small LMTD ({LMTD:.2f} K) - pinch point may be too tight")

        # Import upstream libraries
        from fluids.geometry import PlateExchanger
        from ht.conv_plate import (
            Nu_plate_Kumar,
            Nu_plate_Martin,
            Nu_plate_Muley_Manglik,
        )
        from fluids.friction import (
            friction_plate_Kumar,
            friction_plate_Martin_1999,
            friction_plate_Martin_VDI,
            friction_plate_Muley_Manglik,
        )

        # Determine target plates to evaluate
        if n_plates is not None:
            # User specified plate count
            if n_plates % 2 == 0:
                n_plates += 1  # Force odd for symmetric allocation
            plates_to_try = [n_plates]
        else:
            # Solver mode: try range of odd plate counts
            plates_to_try = list(range(min_plates if min_plates % 2 == 1 else min_plates + 1, max_plates + 1, 2))

        best_result = None
        all_results = []

        for n_pl in plates_to_try:
            try:
                # Create PlateExchanger geometry instance
                phe = PlateExchanger(
                    amplitude=plate_amplitude_m,
                    wavelength=plate_wavelength_m,
                    chevron_angle=chevron_angle_deg,
                    width=plate_width_m,
                    length=plate_length_m,
                    d_port=port_diameter_m,
                    plates=n_pl,
                    thickness=plate_thickness_m,
                )

                # Get geometry from class
                D_h = phe.D_hydraulic
                phi = phe.plate_enlargement_factor
                A_channel = phe.A_channel_flow
                A_plate = getattr(phe, "A_plate_surface", plate_width_m * plate_length_m * phi)
                n_channels = phe.channels  # = plates - 1
                channels_per_fluid = n_channels // 2

                # Validate pass configuration
                if channels_per_fluid % passes_hot != 0:
                    continue  # Invalid pass/plate combination
                if channels_per_fluid % passes_cold != 0:
                    continue

                channels_per_pass_hot = channels_per_fluid // passes_hot
                channels_per_pass_cold = channels_per_fluid // passes_cold

                # Calculate velocities (per channel)
                v_hot = hot_mass_flow_kg_s / (rho_hot * channels_per_pass_hot * A_channel)
                v_cold = cold_mass_flow_kg_s / (rho_cold * channels_per_pass_cold * A_channel)

                # Calculate Reynolds numbers
                Re_hot = rho_hot * v_hot * D_h / mu_hot
                Re_cold = rho_cold * v_cold * D_h / mu_cold

                # Calculate Nusselt numbers using selected correlation
                warnings_list = []
                paired_friction = None

                if correlation == "Kumar":
                    # Kumar correlation supports viscosity correction (mu/mu_wall)^0.17
                    # Estimate plate temperature as average of bulk temps
                    T_plate = (hot_bulk_temp + cold_bulk_temp) / 2
                    # Wall temp for each side is average of bulk and plate temp
                    T_wall_hot = (hot_bulk_temp + T_plate) / 2
                    T_wall_cold = (cold_bulk_temp + T_plate) / 2

                    # Get viscosity at wall temperature for correction
                    hot_wall_props = json.loads(
                        get_fluid_properties(hot_fluid, T_wall_hot, hot_fluid_pressure_Pa, strict=strict)
                    )
                    cold_wall_props = json.loads(
                        get_fluid_properties(cold_fluid, T_wall_cold, cold_fluid_pressure_Pa, strict=strict)
                    )
                    mu_wall_hot = hot_wall_props.get("dynamic_viscosity", mu_hot)
                    mu_wall_cold = cold_wall_props.get("dynamic_viscosity", mu_cold)

                    Nu_hot = Nu_plate_Kumar(
                        Re=Re_hot, Pr=Pr_hot, chevron_angle=chevron_angle_deg, mu=mu_hot, mu_wall=mu_wall_hot
                    )
                    Nu_cold = Nu_plate_Kumar(
                        Re=Re_cold, Pr=Pr_cold, chevron_angle=chevron_angle_deg, mu=mu_cold, mu_wall=mu_wall_cold
                    )
                    f_hot = friction_plate_Kumar(Re_hot, chevron_angle_deg)
                    f_cold = friction_plate_Kumar(Re_cold, chevron_angle_deg)
                    paired_friction = "friction_plate_Kumar"

                    if chevron_angle_deg < 30 or chevron_angle_deg > 65:
                        warnings_list.append(f"Kumar valid for chevron 30-65°, got {chevron_angle_deg}°")

                elif correlation == "Martin_1999":
                    Nu_hot = Nu_plate_Martin(Re=Re_hot, Pr=Pr_hot, chevron_angle=chevron_angle_deg, variant="1999")
                    Nu_cold = Nu_plate_Martin(Re=Re_cold, Pr=Pr_cold, chevron_angle=chevron_angle_deg, variant="1999")
                    f_hot = friction_plate_Martin_1999(Re_hot, chevron_angle_deg)
                    f_cold = friction_plate_Martin_1999(Re_cold, chevron_angle_deg)
                    paired_friction = "friction_plate_Martin_1999"

                elif correlation == "Martin_VDI":
                    Nu_hot = Nu_plate_Martin(Re=Re_hot, Pr=Pr_hot, chevron_angle=chevron_angle_deg, variant="VDI")
                    Nu_cold = Nu_plate_Martin(Re=Re_cold, Pr=Pr_cold, chevron_angle=chevron_angle_deg, variant="VDI")
                    f_hot = friction_plate_Martin_VDI(Re_hot, chevron_angle_deg)
                    f_cold = friction_plate_Martin_VDI(Re_cold, chevron_angle_deg)
                    paired_friction = "friction_plate_Martin_VDI"

                elif correlation == "Muley_Manglik":
                    Nu_hot = Nu_plate_Muley_Manglik(
                        Re=Re_hot, Pr=Pr_hot, chevron_angle=chevron_angle_deg, plate_enlargement_factor=phi
                    )
                    Nu_cold = Nu_plate_Muley_Manglik(
                        Re=Re_cold, Pr=Pr_cold, chevron_angle=chevron_angle_deg, plate_enlargement_factor=phi
                    )
                    f_hot = friction_plate_Muley_Manglik(Re_hot, chevron_angle_deg, phi)
                    f_cold = friction_plate_Muley_Manglik(Re_cold, chevron_angle_deg, phi)
                    paired_friction = "friction_plate_Muley_Manglik"

                    if Re_hot < 1000 or Re_cold < 1000:
                        warnings_list.append(f"Muley_Manglik valid for Re > 1000")

                # Calculate heat transfer coefficients
                h_hot = Nu_hot * k_hot / D_h
                h_cold = Nu_cold * k_cold / D_h

                # Calculate overall U (based on plate surface area)
                R_wall = plate_thickness_m / plate_conductivity_W_mK
                R_total = 1 / h_hot + fouling_factor_hot_m2K_W + R_wall + fouling_factor_cold_m2K_W + 1 / h_cold
                U = 1 / R_total

                # Calculate available area
                # A_heat_transfer = (plates - 2) * A_plate_surface
                A_available = (n_pl - 2) * A_plate

                # Calculate required area
                A_required = abs(heat_duty_W) / (U * LMTD)

                # Calculate pressure drops
                # Frictional: dP_channel = f * (L/D_h) * (rho * v^2 / 2)
                dP_channel_hot = f_hot * (plate_length_m / D_h) * (rho_hot * v_hot**2 / 2)
                dP_channel_cold = f_cold * (plate_length_m / D_h) * (rho_cold * v_cold**2 / 2)

                # Total frictional = per_channel * n_passes
                dP_friction_hot = dP_channel_hot * passes_hot
                dP_friction_cold = dP_channel_cold * passes_cold

                # Port losses (if port diameter provided)
                dP_port_hot = 0.0
                dP_port_cold = 0.0
                v_port_hot = 0.0
                v_port_cold = 0.0

                if port_diameter_m is not None and port_diameter_m > 0:
                    A_port = math.pi * port_diameter_m**2 / 4
                    v_port_hot = hot_mass_flow_kg_s / (rho_hot * A_port)
                    v_port_cold = cold_mass_flow_kg_s / (rho_cold * A_port)
                    # ~1.4 velocity heads per port pair (inlet + outlet)
                    # Velocity head = 0.5 * rho * v^2, so total = 1.4 * 0.5 * rho * v^2
                    dP_port_hot = 1.4 * 0.5 * rho_hot * v_port_hot**2
                    dP_port_cold = 1.4 * 0.5 * rho_cold * v_port_cold**2

                dP_total_hot = dP_friction_hot + dP_port_hot
                dP_total_cold = dP_friction_cold + dP_port_cold

                # Check constraints
                thermal_satisfied = A_available >= A_required * 0.99  # 1% tolerance
                hydraulic_satisfied = True
                if max_pressure_drop_kPa is not None:
                    if dP_total_hot / 1000 > max_pressure_drop_kPa:
                        hydraulic_satisfied = False
                    if dP_total_cold / 1000 > max_pressure_drop_kPa:
                        hydraulic_satisfied = False

                # Calculate effectiveness and NTU
                C_hot = hot_mass_flow_kg_s * cp_hot
                C_cold = cold_mass_flow_kg_s * cp_cold
                C_min = min(C_hot, C_cold)
                C_max = max(C_hot, C_cold)
                C_r = C_min / C_max if C_max > 0 else 0

                Q_max = C_min * (hot_inlet_temp_K - cold_inlet_temp_K)
                effectiveness = abs(heat_duty_W) / Q_max if Q_max > 0 else 0
                NTU = U * A_available / C_min if C_min > 0 else 0

                # Build result for this plate count
                result = {
                    "plates": n_pl,
                    "thermal_satisfied": thermal_satisfied,
                    "hydraulic_satisfied": hydraulic_satisfied,
                    "area_available_m2": A_available,
                    "area_required_m2": A_required,
                    "area_margin_pct": (A_available / A_required - 1) * 100 if A_required > 0 else 0,
                    "U_W_m2K": U,
                    "dP_hot_kPa": dP_total_hot / 1000,
                    "dP_cold_kPa": dP_total_cold / 1000,
                }

                all_results.append(result)

                # Select best result (first that satisfies both constraints)
                if best_result is None and thermal_satisfied and hydraulic_satisfied:
                    # Build comprehensive result
                    best_result = {
                        "duty_W": abs(heat_duty_W),
                        "duty_kW": abs(heat_duty_W) / 1000,
                        "LMTD_K": LMTD,
                        "effectiveness": effectiveness,
                        "NTU": NTU,
                        "geometry": {
                            "plates": n_pl,
                            "channels_total": n_channels,
                            "channels_hot": channels_per_fluid,
                            "channels_cold": channels_per_fluid,
                            "channels_per_pass_hot": channels_per_pass_hot,
                            "channels_per_pass_cold": channels_per_pass_cold,
                            "area_per_plate_m2": A_plate,
                            "total_area_m2": A_available,
                            "area_required_m2": A_required,
                            "area_margin_pct": (A_available / A_required - 1) * 100 if A_required > 0 else 0,
                            "hydraulic_diameter_m": D_h,
                            "plate_enlargement_factor": phi,
                            "plate_dimensions": {
                                "width_m": plate_width_m,
                                "length_m": plate_length_m,
                                "amplitude_m": plate_amplitude_m,
                                "wavelength_m": plate_wavelength_m,
                                "thickness_m": plate_thickness_m,
                            },
                            "chevron_angle_deg": chevron_angle_deg,
                        },
                        "thermal": {
                            "U_W_m2K": U,
                            "h_hot_W_m2K": h_hot,
                            "h_cold_W_m2K": h_cold,
                            "Nu_hot": Nu_hot,
                            "Nu_cold": Nu_cold,
                            "Re_hot": Re_hot,
                            "Re_cold": Re_cold,
                            "Pr_hot": Pr_hot,
                            "Pr_cold": Pr_cold,
                            "correlation": correlation,
                            "R_wall_m2K_W": R_wall,
                            "fouling_hot_m2K_W": fouling_factor_hot_m2K_W,
                            "fouling_cold_m2K_W": fouling_factor_cold_m2K_W,
                        },
                        "hydraulic": {
                            "velocity_hot_m_s": v_hot,
                            "velocity_cold_m_s": v_cold,
                            "pressure_drop_hot_Pa": dP_total_hot,
                            "pressure_drop_hot_kPa": dP_total_hot / 1000,
                            "pressure_drop_cold_Pa": dP_total_cold,
                            "pressure_drop_cold_kPa": dP_total_cold / 1000,
                            "friction_factor_hot": f_hot,
                            "friction_factor_cold": f_cold,
                            "correlation": paired_friction,
                            "components_hot": {
                                "frictional_kPa": dP_friction_hot / 1000,
                                "port_losses_kPa": dP_port_hot / 1000,
                            },
                            "components_cold": {
                                "frictional_kPa": dP_friction_cold / 1000,
                                "port_losses_kPa": dP_port_cold / 1000,
                            },
                        },
                        "temperatures": {
                            "hot_inlet_K": hot_inlet_temp_K,
                            "hot_inlet_C": hot_inlet_temp_K - 273.15,
                            "hot_outlet_K": hot_outlet_temp_K,
                            "hot_outlet_C": hot_outlet_temp_K - 273.15,
                            "cold_inlet_K": cold_inlet_temp_K,
                            "cold_inlet_C": cold_inlet_temp_K - 273.15,
                            "cold_outlet_K": cold_outlet_temp_K,
                            "cold_outlet_C": cold_outlet_temp_K - 273.15,
                            # Terminal temperature differences (correct terminology)
                            "terminal_temp_diff_min_K": min(
                                hot_inlet_temp_K - cold_outlet_temp_K, hot_outlet_temp_K - cold_inlet_temp_K
                            ),
                            "terminal_temp_diff_max_K": max(
                                hot_inlet_temp_K - cold_outlet_temp_K, hot_outlet_temp_K - cold_inlet_temp_K
                            ),
                        },
                        "fluids": {
                            "hot": {
                                "name": hot_fluid,
                                "mass_flow_kg_s": hot_mass_flow_kg_s,
                                "density_kg_m3": rho_hot,
                                "viscosity_Pa_s": mu_hot,
                                "conductivity_W_mK": k_hot,
                                "specific_heat_J_kgK": cp_hot,
                            },
                            "cold": {
                                "name": cold_fluid,
                                "mass_flow_kg_s": cold_mass_flow_kg_s,
                                "density_kg_m3": rho_cold,
                                "viscosity_Pa_s": mu_cold,
                                "conductivity_W_mK": k_cold,
                                "specific_heat_J_kgK": cp_cold,
                            },
                        },
                        "configuration": {
                            "flow_arrangement": flow_arrangement,
                            "passes_hot": passes_hot,
                            "passes_cold": passes_cold,
                        },
                    }

                    if port_diameter_m:
                        best_result["hydraulic"]["port_velocity_hot_m_s"] = v_port_hot
                        best_result["hydraulic"]["port_velocity_cold_m_s"] = v_port_cold
                        best_result["geometry"]["port_diameter_m"] = port_diameter_m

                    if warnings_list:
                        best_result["warnings"] = warnings_list

            except Exception as calc_error:
                logger.debug(f"Calculation failed for {n_pl} plates: {calc_error}")
                continue

        # Return results
        if best_result is not None:
            return json.dumps(best_result)
        elif all_results:
            # No solution found that satisfies constraints
            return json.dumps(
                {
                    "error": "No plate count satisfies both thermal and hydraulic constraints",
                    "suggestion": "Try relaxing max_pressure_drop_kPa or adjusting plate geometry",
                    "evaluated_configurations": all_results[:10],  # Show first 10
                }
            )
        else:
            return json.dumps(
                {
                    "error": "Could not evaluate any plate configurations",
                    "suggestion": "Check plate geometry parameters and pass configuration",
                }
            )

    except ImportError as e:
        return json.dumps(
            {
                "error": f"Required library import failed: {e}",
                "suggestion": "Install with: pip install ht>=1.2.0 fluids>=1.0.0",
            }
        )
    except Exception as e:
        logger.error(f"Error in size_plate_heat_exchanger: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
