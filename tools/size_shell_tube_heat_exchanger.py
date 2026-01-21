"""
Dedicated Shell-Tube Heat Exchanger sizing tool with thermal-hydraulic coupling.

This module provides accurate shell-tube HX sizing using upstream Caleb Bell ht/fluids
libraries with proper coupling between thermal performance (U-value) and hydraulic
performance (pressure drop) through velocity and Reynolds number dependencies.

Key features:
- Uses TEMA tube standards and bundle layout functions from ht.hx
- Tube-side convection via Nu_conv_internal (auto-selects best correlation)
- Shell-side convection via Kern or Zukauskas tube-bank methods
- Pressure drop via dP_Kern (shell) and one_phase_dP (tube)
- LMTD correction factor via F_LMTD_Fakheri for multi-pass configurations
- Iterative solver to find tube count satisfying thermal + hydraulic constraints
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

from utils.import_helpers import HT_AVAILABLE, FLUIDS_AVAILABLE
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.size_shell_tube_heat_exchanger")


def size_shell_tube_heat_exchanger(
    # Duty specification (provide heat_duty_W OR temperatures to calculate)
    heat_duty_W: Optional[float] = None,
    # Temperature specs
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
    # Shell geometry
    shell_inner_diameter_m: Optional[float] = None,
    baffle_spacing_m: Optional[float] = None,
    baffle_cut_fraction: float = 0.25,
    n_baffles: Optional[int] = None,
    # Tube geometry
    tube_outer_diameter_m: float = 0.019,  # 3/4" OD default
    tube_inner_diameter_m: float = 0.016,  # ~16mm ID (14 BWG)
    tube_pitch_m: float = 0.025,  # 25mm pitch (1.25 * OD typical)
    tube_layout_angle: int = 30,  # 30, 45, 60, or 90 degrees
    tube_length_m: float = 3.0,  # 3m tube length default
    tube_roughness_m: float = 0.00004,  # ~40 micron for steel
    tube_material_conductivity_W_mK: float = 45.0,  # Carbon steel default
    # Configuration
    n_tube_passes: int = 2,
    n_shell_passes: int = 1,
    tube_side_fluid: str = "cold",  # Which fluid in tubes: "hot" or "cold"
    # Constraints for solver
    max_pressure_drop_tube_kPa: Optional[float] = None,
    max_pressure_drop_shell_kPa: Optional[float] = None,
    max_velocity_tube_m_s: float = 3.0,
    max_velocity_shell_m_s: float = 1.5,
    min_velocity_tube_m_s: float = 0.3,  # Minimum to ensure reasonable Re
    min_Re_tube: float = 2500.0,  # Minimum for turbulent flow (Re > 2300)
    # Fouling
    fouling_factor_tube_m2K_W: float = 0.0001,
    fouling_factor_shell_m2K_W: float = 0.0002,
    # Correlation selection
    shell_side_method: str = "Kern",  # "Kern" or "Zukauskas"
    tube_side_method: str = "auto",  # "auto", "Gnielinski", "Dittus_Boelter"
    # Solver options
    solve_for: str = "n_tubes",  # "n_tubes", "tube_length"
    min_tubes: int = 10,
    max_tubes: int = 2000,
    # Sweep/optimization options
    sweep_n_tube_passes: Optional[List[int]] = None,
    auto_optimize: bool = False,  # Enable automatic multi-parameter search
    optimize_for: str = "area",  # "area", "cost", "dP" - objective to minimize
    strict: bool = False,
) -> str:
    """Size a shell-tube heat exchanger with coupled thermal-hydraulic design.

    This tool provides accurate shell-tube HX sizing using upstream ht/fluids libraries
    with proper thermal-hydraulic coupling. The tube count, U-value, and pressure drops
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

        shell_inner_diameter_m: Shell ID (m). If None, estimated from tube count.
            When specified, uses ht.hx.Ntubes_Phadkeb to compute the maximum
            tube count that fits within the shell, constraining the search space.
        baffle_spacing_m: Baffle spacing (m). If None, defaults to 0.4 * shell_diameter.
        baffle_cut_fraction: Baffle cut as fraction (0.25 = 25%). Default 0.25.
        n_baffles: Number of baffles. If None, calculated from tube_length / baffle_spacing.

        tube_outer_diameter_m: Tube OD (m). Default 0.019 (3/4").
        tube_inner_diameter_m: Tube ID (m). Default 0.016.
        tube_pitch_m: Tube pitch (m). Default 0.025 (25mm).
        tube_layout_angle: Layout angle (30, 45, 60, 90). Default 30 (triangular).
        tube_length_m: Tube length (m). Default 3.0.
        tube_roughness_m: Tube roughness (m). Default 0.00004.
        tube_material_conductivity_W_mK: Tube material k (W/m-K). Default 45.

        n_tube_passes: Number of tube passes. Default 2.
        n_shell_passes: Number of shell passes. Default 1.
        tube_side_fluid: Which fluid on tube side ("hot" or "cold"). Default "cold".

        max_pressure_drop_tube_kPa: Maximum tube-side dP (kPa).
        max_pressure_drop_shell_kPa: Maximum shell-side dP (kPa).
        max_velocity_tube_m_s: Maximum tube velocity (m/s). Default 3.0.
        max_velocity_shell_m_s: Maximum shell velocity (m/s). Default 1.5.
        min_velocity_tube_m_s: Minimum tube velocity (m/s) for adequate heat transfer. Default 0.5.
        min_Re_tube: Minimum tube-side Reynolds number for turbulent flow. Default 5000.

        fouling_factor_tube_m2K_W: Tube-side fouling (m²K/W). Default 0.0001.
        fouling_factor_shell_m2K_W: Shell-side fouling (m²K/W). Default 0.0002.

        shell_side_method: "Kern" or "Zukauskas". Default "Kern".
        tube_side_method: "auto", "Gnielinski", or "Dittus_Boelter". Default "auto".

        solve_for: What to optimize: "n_tubes" or "tube_length". Default "n_tubes".
        min_tubes: Minimum tubes for solver (default 10).
        max_tubes: Maximum tubes for solver (default 2000).

        sweep_n_tube_passes: List of pass counts for parameter sweep.
        strict: If True, fail if libraries unavailable.

    Returns:
        JSON with comprehensive sizing results including:
        - duty_kW, LMTD_K, F_correction, effectiveness, NTU
        - geometry: n_tubes, tube_length, shell_diameter, bundle_diameter, n_baffles
        - thermal: U, h_tube, h_shell, Re, Pr, correlation
        - hydraulic: velocity, pressure_drop, friction_factor for both sides
        - temperatures: inlet/outlet for both sides
        - heat_balance_verification
    """
    try:
        # Validate library availability
        if not HT_AVAILABLE:
            if strict:
                return json.dumps({"error": "ht library required with strict=True"})
            return json.dumps(
                {
                    "error": "ht library not available for shell-tube sizing",
                    "suggestion": "Install with: pip install ht>=1.2.0",
                }
            )

        if not FLUIDS_AVAILABLE:
            if strict:
                return json.dumps({"error": "fluids library required with strict=True"})
            return json.dumps(
                {
                    "error": "fluids library not available for shell-tube sizing",
                    "suggestion": "Install with: pip install fluids>=1.0.0",
                }
            )

        # Handle automatic optimization if requested
        if auto_optimize:
            # Search across multiple design parameters to find feasible configurations
            search_passes = [1, 2, 4, 6, 8]
            search_lengths = [2.0, 3.0, 4.0, 5.0, 6.0]
            search_angles = [30, 45]

            all_candidates = []
            for passes in search_passes:
                for length in search_lengths:
                    for angle in search_angles:
                        try:
                            result_json = size_shell_tube_heat_exchanger(
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
                                shell_inner_diameter_m=shell_inner_diameter_m,
                                baffle_spacing_m=baffle_spacing_m,
                                baffle_cut_fraction=baffle_cut_fraction,
                                n_baffles=n_baffles,
                                tube_outer_diameter_m=tube_outer_diameter_m,
                                tube_inner_diameter_m=tube_inner_diameter_m,
                                tube_pitch_m=tube_pitch_m,
                                tube_layout_angle=angle,
                                tube_length_m=length,
                                tube_roughness_m=tube_roughness_m,
                                tube_material_conductivity_W_mK=tube_material_conductivity_W_mK,
                                n_tube_passes=passes,
                                n_shell_passes=n_shell_passes,
                                tube_side_fluid=tube_side_fluid,
                                max_pressure_drop_tube_kPa=max_pressure_drop_tube_kPa,
                                max_pressure_drop_shell_kPa=max_pressure_drop_shell_kPa,
                                max_velocity_tube_m_s=max_velocity_tube_m_s,
                                max_velocity_shell_m_s=max_velocity_shell_m_s,
                                min_velocity_tube_m_s=min_velocity_tube_m_s,
                                min_Re_tube=min_Re_tube,
                                fouling_factor_tube_m2K_W=fouling_factor_tube_m2K_W,
                                fouling_factor_shell_m2K_W=fouling_factor_shell_m2K_W,
                                shell_side_method=shell_side_method,
                                tube_side_method=tube_side_method,
                                solve_for=solve_for,
                                min_tubes=min_tubes,
                                max_tubes=max_tubes,
                                auto_optimize=False,  # Don't recurse
                                strict=strict,
                            )
                            result = json.loads(result_json)
                            if "error" not in result:
                                # Calculate objective score
                                if optimize_for == "area":
                                    score = result["geometry"]["total_area_m2"]
                                elif optimize_for == "dP":
                                    score = (
                                        result["hydraulic"]["pressure_drop_tube_kPa"]
                                        + result["hydraulic"]["pressure_drop_shell_kPa"]
                                    )
                                else:  # "cost" - approximate based on area and tube count
                                    score = result["geometry"]["total_area_m2"] * (1 + 0.01 * result["geometry"]["n_tubes"])

                                all_candidates.append(
                                    {
                                        "config": {
                                            "n_tube_passes": passes,
                                            "tube_length_m": length,
                                            "tube_layout_angle": angle,
                                        },
                                        "score": score,
                                        "n_tubes": result["geometry"]["n_tubes"],
                                        "total_area_m2": result["geometry"]["total_area_m2"],
                                        "U_W_m2K": result["thermal"]["U_W_m2K"],
                                        "Re_tube": result["thermal"]["Re_tube"],
                                        "dP_tube_kPa": result["hydraulic"]["pressure_drop_tube_kPa"],
                                        "dP_shell_kPa": result["hydraulic"]["pressure_drop_shell_kPa"],
                                        "full_result": result,
                                    }
                                )
                        except Exception as e:
                            logger.debug(
                                f"Optimization candidate failed: passes={passes}, length={length}, angle={angle}: {e}"
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
                                "n_tubes": c["n_tubes"],
                                "area_m2": c["total_area_m2"],
                                "U_W_m2K": c["U_W_m2K"],
                                "Re_tube": c["Re_tube"],
                                "dP_tube_kPa": c["dP_tube_kPa"],
                                "dP_shell_kPa": c["dP_shell_kPa"],
                            }
                            for c in all_candidates[:5]
                        ],
                        "search_space": {
                            "n_tube_passes": search_passes,
                            "tube_length_m": search_lengths,
                            "tube_layout_angle": search_angles,
                        },
                    }
                )
            else:
                return json.dumps(
                    {
                        "error": "No feasible configuration found in optimization search",
                        "search_space": {
                            "n_tube_passes": search_passes,
                            "tube_length_m": search_lengths,
                            "tube_layout_angle": search_angles,
                        },
                        "suggestion": "Try relaxing constraints (max_pressure_drop, min_Re_tube) or expanding search ranges",
                    }
                )

        # Handle parameter sweep if requested
        if sweep_n_tube_passes is not None and len(sweep_n_tube_passes) > 0:
            sweep_results = []
            for passes in sorted(sweep_n_tube_passes):
                single_result_json = size_shell_tube_heat_exchanger(
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
                    shell_inner_diameter_m=shell_inner_diameter_m,
                    baffle_spacing_m=baffle_spacing_m,
                    baffle_cut_fraction=baffle_cut_fraction,
                    n_baffles=n_baffles,
                    tube_outer_diameter_m=tube_outer_diameter_m,
                    tube_inner_diameter_m=tube_inner_diameter_m,
                    tube_pitch_m=tube_pitch_m,
                    tube_layout_angle=tube_layout_angle,
                    tube_length_m=tube_length_m,
                    tube_roughness_m=tube_roughness_m,
                    tube_material_conductivity_W_mK=tube_material_conductivity_W_mK,
                    n_tube_passes=passes,
                    n_shell_passes=n_shell_passes,
                    tube_side_fluid=tube_side_fluid,
                    max_pressure_drop_tube_kPa=max_pressure_drop_tube_kPa,
                    max_pressure_drop_shell_kPa=max_pressure_drop_shell_kPa,
                    max_velocity_tube_m_s=max_velocity_tube_m_s,
                    max_velocity_shell_m_s=max_velocity_shell_m_s,
                    min_velocity_tube_m_s=min_velocity_tube_m_s,
                    min_Re_tube=min_Re_tube,
                    fouling_factor_tube_m2K_W=fouling_factor_tube_m2K_W,
                    fouling_factor_shell_m2K_W=fouling_factor_shell_m2K_W,
                    shell_side_method=shell_side_method,
                    tube_side_method=tube_side_method,
                    solve_for=solve_for,
                    min_tubes=min_tubes,
                    max_tubes=max_tubes,
                    sweep_n_tube_passes=None,
                    strict=strict,
                )
                single_result = json.loads(single_result_json)

                if "error" not in single_result:
                    sweep_results.append(
                        {
                            "n_tube_passes": passes,
                            "n_tubes": single_result["geometry"]["n_tubes"],
                            "total_area_m2": single_result["geometry"]["total_area_m2"],
                            "U_W_m2K": single_result["thermal"]["U_W_m2K"],
                            "F_correction": single_result["F_correction"],
                            "dP_tube_kPa": single_result["hydraulic"]["pressure_drop_tube_kPa"],
                            "dP_shell_kPa": single_result["hydraulic"]["pressure_drop_shell_kPa"],
                            "Re_tube": single_result["thermal"]["Re_tube"],
                            "Re_shell": single_result["thermal"]["Re_shell"],
                        }
                    )
                else:
                    sweep_results.append(
                        {
                            "n_tube_passes": passes,
                            "error": single_result["error"],
                        }
                    )

            return json.dumps(
                {
                    "sweep_type": "n_tube_passes",
                    "sweep_values": sorted(sweep_n_tube_passes),
                    "results": sweep_results,
                    "analysis_notes": [
                        "More tube passes -> higher tube velocity -> higher h_tube -> higher U",
                        "More tube passes -> higher tube-side dP (proportional to passes)",
                        "F correction factor decreases with more passes for given NTU",
                        "Optimal passes balance U improvement vs dP increase",
                    ],
                }
            )

        # Validate required inputs
        if hot_mass_flow_kg_s is None or hot_mass_flow_kg_s <= 0:
            return json.dumps({"error": "hot_mass_flow_kg_s must be positive"})
        if cold_mass_flow_kg_s is None or cold_mass_flow_kg_s <= 0:
            return json.dumps({"error": "cold_mass_flow_kg_s must be positive"})

        # Validate temperatures and determine duty
        temps_provided = [hot_inlet_temp_K, hot_outlet_temp_K, cold_inlet_temp_K, cold_outlet_temp_K]
        temps_count = sum(1 for t in temps_provided if t is not None)

        if heat_duty_W is None:
            if temps_count < 4:
                return json.dumps({"error": "Either heat_duty_W or all four temperatures must be provided"})
        else:
            if hot_inlet_temp_K is None or cold_inlet_temp_K is None:
                return json.dumps({"error": "Both inlet temperatures required when heat_duty_W is provided"})

        # Validate tube geometry
        if tube_pitch_m <= tube_outer_diameter_m:
            return json.dumps(
                {"error": f"tube_pitch_m ({tube_pitch_m}) must be > tube_outer_diameter_m ({tube_outer_diameter_m})"}
            )
        if tube_inner_diameter_m >= tube_outer_diameter_m:
            return json.dumps(
                {
                    "error": f"tube_inner_diameter_m ({tube_inner_diameter_m}) must be < tube_outer_diameter_m ({tube_outer_diameter_m})"
                }
            )
        if tube_layout_angle not in [30, 45, 60, 90]:
            return json.dumps({"error": f"tube_layout_angle must be 30, 45, 60, or 90 degrees, got {tube_layout_angle}"})

        # Assign fluids to tube/shell sides
        if tube_side_fluid.lower() == "cold":
            tube_fluid = cold_fluid
            tube_flow = cold_mass_flow_kg_s
            tube_pressure = cold_fluid_pressure_Pa
            shell_fluid = hot_fluid
            shell_flow = hot_mass_flow_kg_s
            shell_pressure = hot_fluid_pressure_Pa
        else:
            tube_fluid = hot_fluid
            tube_flow = hot_mass_flow_kg_s
            tube_pressure = hot_fluid_pressure_Pa
            shell_fluid = cold_fluid
            shell_flow = cold_mass_flow_kg_s
            shell_pressure = cold_fluid_pressure_Pa

        # Get fluid properties at bulk temperatures
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

        # Assign properties to tube/shell sides
        if tube_side_fluid.lower() == "cold":
            rho_tube, mu_tube, k_tube, cp_tube, Pr_tube = rho_cold, mu_cold, k_cold, cp_cold, Pr_cold
            rho_shell, mu_shell, k_shell, cp_shell, Pr_shell = rho_hot, mu_hot, k_hot, cp_hot, Pr_hot
        else:
            rho_tube, mu_tube, k_tube, cp_tube, Pr_tube = rho_hot, mu_hot, k_hot, cp_hot, Pr_hot
            rho_shell, mu_shell, k_shell, cp_shell, Pr_shell = rho_cold, mu_cold, k_cold, cp_cold, Pr_cold

        # Calculate or verify heat duty
        if heat_duty_W is None:
            heat_duty_W = hot_mass_flow_kg_s * cp_hot * (hot_inlet_temp_K - hot_outlet_temp_K)
            Q_cold = cold_mass_flow_kg_s * cp_cold * (cold_outlet_temp_K - cold_inlet_temp_K)
            if abs(heat_duty_W - Q_cold) / max(abs(heat_duty_W), 1) > 0.05:
                logger.warning(f"Energy imbalance: Q_hot={heat_duty_W:.1f}W, Q_cold={Q_cold:.1f}W")
        else:
            if hot_outlet_temp_K is None:
                hot_outlet_temp_K = hot_inlet_temp_K - heat_duty_W / (hot_mass_flow_kg_s * cp_hot)
            if cold_outlet_temp_K is None:
                cold_outlet_temp_K = cold_inlet_temp_K + heat_duty_W / (cold_mass_flow_kg_s * cp_cold)

        # Calculate LMTD using ht.core.LMTD (counterflow assumed for base calculation)
        from ht.core import LMTD as ht_LMTD

        # Check for temperature crossover before calling library
        dT1 = hot_inlet_temp_K - cold_outlet_temp_K
        dT2 = hot_outlet_temp_K - cold_inlet_temp_K

        if dT1 <= 0 or dT2 <= 0:
            return json.dumps(
                {
                    "error": "Temperature crossover: LMTD undefined",
                    "details": {
                        "dT1_hot_in_minus_cold_out": dT1,
                        "dT2_hot_out_minus_cold_in": dT2,
                        "hint": "Hot fluid must be hotter than cold fluid at both ends",
                    },
                }
            )

        LMTD = ht_LMTD(
            Thi=hot_inlet_temp_K, Tho=hot_outlet_temp_K, Tci=cold_inlet_temp_K, Tco=cold_outlet_temp_K, counterflow=True
        )

        # Calculate LMTD correction factor for multi-pass
        # Note: F_LMTD_Fakheri is for shell-and-tube HX with 2N tube passes per shell pass
        # Standard configurations: 1 shell + 2/4/6 tube passes, 2 shells + 4/8/12 tube passes, etc.
        F_correction = 1.0
        try:
            from ht.hx import F_LMTD_Fakheri

            # Validate tube passes for Fakheri correlation
            # The correlation assumes tube passes = 2 * shells * N for some integer N
            # Warn if using non-standard configurations
            if n_tube_passes < 2 * n_shell_passes:
                logger.warning(
                    f"F_LMTD_Fakheri: n_tube_passes ({n_tube_passes}) < 2 * n_shell_passes ({n_shell_passes}). "
                    "This is a non-standard configuration; F correction may be inaccurate."
                )
            elif n_tube_passes % (2 * n_shell_passes) != 0:
                logger.warning(
                    f"F_LMTD_Fakheri: n_tube_passes ({n_tube_passes}) is not a multiple of 2 * n_shell_passes ({2 * n_shell_passes}). "
                    "Standard configurations are 2, 4, 6... tube passes per shell. F correction may be inaccurate."
                )

            F_correction = F_LMTD_Fakheri(
                Tci=cold_inlet_temp_K,
                Tco=cold_outlet_temp_K,
                Thi=hot_inlet_temp_K,
                Tho=hot_outlet_temp_K,
                shells=n_shell_passes,
            )
            if F_correction < 0.75:
                logger.warning(f"Low F correction ({F_correction:.3f}) suggests temperature approach is tight")
        except Exception as e:
            logger.debug(f"F_LMTD correction fallback to 1.0: {e}")
            F_correction = 1.0

        # Import required functions from upstream ht/fluids libraries
        # Philosophy: Use library functions exclusively, fail loudly if unavailable
        from ht.conv_internal import Nu_conv_internal
        from ht.conv_tube_bank import Nu_Zukauskas_Bejan, dP_Kern, dP_Zukauskas
        from fluids.friction import one_phase_dP, friction_factor
        from ht.hx import DBundle_for_Ntubes_Phadkeb, Ntubes_Phadkeb

        # Tube geometry
        A_tube_inner = math.pi * tube_inner_diameter_m**2 / 4
        tube_wall_thickness = (tube_outer_diameter_m - tube_inner_diameter_m) / 2

        # If shell diameter is specified, use Ntubes_Phadkeb to compute max tube count
        # This constrains the search space based on physical shell dimensions
        max_tubes_from_shell = None
        if shell_inner_diameter_m is not None:
            # Bundle diameter must be less than shell ID minus clearance
            # Typical clearance is 10-25mm for pull-through bundles
            clearance_bundle_to_shell = 0.020  # 20mm default clearance
            max_bundle_diameter = shell_inner_diameter_m - clearance_bundle_to_shell

            if max_bundle_diameter > 2 * tube_pitch_m:  # Sanity check
                try:
                    max_tubes_from_shell = int(
                        Ntubes_Phadkeb(
                            DBundle=max_bundle_diameter,
                            Do=tube_outer_diameter_m,
                            pitch=tube_pitch_m,
                            Ntp=n_tube_passes,
                            angle=tube_layout_angle,
                        )
                    )
                    # Constrain max_tubes to what fits in the shell
                    if max_tubes_from_shell < max_tubes:
                        logger.info(
                            f"Shell ID {shell_inner_diameter_m*1000:.0f}mm constrains "
                            f"max tubes to {max_tubes_from_shell} (was {max_tubes})"
                        )
                        max_tubes = max_tubes_from_shell
                except Exception as e:
                    logger.warning(f"Ntubes_Phadkeb failed for shell constraint: {e}")

        # Iterate to find tube count that satisfies constraints
        best_result = None
        all_results = []

        tubes_to_try = list(range(min_tubes, max_tubes + 1, max(1, (max_tubes - min_tubes) // 200)))
        # Add fine resolution around likely solution
        if len(tubes_to_try) < 100:
            tubes_to_try = list(range(min_tubes, max_tubes + 1))

        for n_tubes in tubes_to_try:
            try:
                # Tubes per pass
                if n_tubes % n_tube_passes != 0:
                    continue  # Skip invalid combinations
                tubes_per_pass = n_tubes // n_tube_passes

                # Tube-side velocity
                v_tube = tube_flow / (rho_tube * tubes_per_pass * A_tube_inner)
                if v_tube > max_velocity_tube_m_s * 1.5:
                    continue  # Way too high, skip
                if v_tube < min_velocity_tube_m_s * 0.5:
                    continue  # Way too low, skip (would give laminar flow)

                # Tube-side Reynolds
                Re_tube = rho_tube * v_tube * tube_inner_diameter_m / mu_tube
                if Re_tube < min_Re_tube * 0.5:
                    continue  # Reynolds too low, skip (laminar flow gives poor h)

                # Tube-side Nusselt (auto-select best correlation)
                # eD = relative roughness = roughness / diameter
                eD_tube = tube_roughness_m / tube_inner_diameter_m

                # Use ht library functions exclusively - no fallbacks
                if tube_side_method == "auto":
                    Nu_tube = Nu_conv_internal(
                        Re=Re_tube,
                        Pr=Pr_tube,
                        eD=eD_tube,
                    )
                else:
                    # Force specific correlation - need friction factor first
                    from ht.conv_internal import turbulent_Gnielinski, turbulent_Dittus_Boelter

                    fd = friction_factor(Re=Re_tube, eD=eD_tube)
                    if tube_side_method == "Gnielinski":
                        Nu_tube = turbulent_Gnielinski(Re_tube, Pr_tube, fd=fd)
                    else:  # Dittus_Boelter
                        Nu_tube = turbulent_Dittus_Boelter(Re_tube, Pr_tube)

                h_tube = Nu_tube * k_tube / tube_inner_diameter_m

                # Estimate shell diameter from tube count using ht library
                D_bundle = DBundle_for_Ntubes_Phadkeb(
                    Ntubes=n_tubes, Do=tube_outer_diameter_m, pitch=tube_pitch_m, Ntp=n_tube_passes, angle=tube_layout_angle
                )

                # Shell diameter = bundle + clearance (typical 10-25mm for pull-through)
                D_shell = shell_inner_diameter_m if shell_inner_diameter_m else D_bundle + 0.020

                # Baffle spacing
                B = baffle_spacing_m if baffle_spacing_m else 0.4 * D_shell
                B = max(B, D_shell / 5)  # Minimum for flow distribution
                B = min(B, D_shell)  # Maximum for support

                # Number of baffles
                N_baffles = n_baffles if n_baffles else max(1, int(tube_length_m / B) - 1)

                # Shell-side cross-flow area
                clearance = tube_pitch_m - tube_outer_diameter_m
                A_shell = D_shell * clearance * B / tube_pitch_m

                # Shell-side velocity
                v_shell = shell_flow / (rho_shell * A_shell)
                if v_shell > max_velocity_shell_m_s * 1.5:
                    continue

                # Shell-side equivalent diameter
                if tube_layout_angle in [30, 60]:
                    De_shell = (
                        4.0 * ((math.sqrt(3) / 4.0) * tube_pitch_m**2 - (math.pi / 8.0) * tube_outer_diameter_m**2)
                    ) / ((math.pi / 2.0) * tube_outer_diameter_m)
                else:
                    De_shell = (4.0 * (tube_pitch_m**2 - (math.pi / 4.0) * tube_outer_diameter_m**2)) / (
                        math.pi * tube_outer_diameter_m
                    )

                # Shell-side mass velocity
                Gs_shell = shell_flow / A_shell

                # Shell-side Nusselt - use ht library correlations exclusively
                tube_rows = max(5, int(math.sqrt(n_tubes)))
                if shell_side_method == "Zukauskas":
                    # Zukauskas/tube-bank correlations use tube OD for Re and h
                    # V_max is the velocity through the minimum cross-sectional area
                    # For crossflow over tube banks, V_max = V_approach * Pt / (Pt - Do)
                    V_approach = shell_flow / (rho_shell * A_shell)
                    V_max = V_approach * tube_pitch_m / clearance if clearance > 0 else V_approach
                    Re_shell_tb = rho_shell * V_max * tube_outer_diameter_m / mu_shell
                    Re_shell = Re_shell_tb  # Alias for consistent use in dP_Zukauskas
                    Nu_shell = Nu_Zukauskas_Bejan(
                        Re=Re_shell_tb, Pr=Pr_shell, tube_rows=tube_rows, pitch_parallel=tube_pitch_m, pitch_normal=tube_pitch_m
                    )
                    # For tube-bank correlations, h = Nu * k / Do (tube outer diameter)
                    h_shell = Nu_shell * k_shell / tube_outer_diameter_m
                else:  # Kern - standard textbook correlation (no ht equivalent)
                    # Kern's method: Nu = 0.36 * Re^0.55 * Pr^(1/3) * (mu/mu_w)^0.14
                    # Re is based on De (shell equivalent diameter)
                    Re_shell = De_shell * Gs_shell / mu_shell
                    # Wall viscosity correction omitted as approximation
                    Nu_shell = 0.36 * Re_shell**0.55 * Pr_shell ** (1 / 3)
                    # For Kern method, h = Nu * k / De
                    h_shell = Nu_shell * k_shell / De_shell

                # Overall U (referenced to tube OD)
                R_tube_inner = 1 / h_tube * (tube_outer_diameter_m / tube_inner_diameter_m)
                R_wall = (
                    tube_outer_diameter_m
                    * math.log(tube_outer_diameter_m / tube_inner_diameter_m)
                    / (2 * tube_material_conductivity_W_mK)
                )
                R_shell = 1 / h_shell
                R_fouling_tube = fouling_factor_tube_m2K_W * (tube_outer_diameter_m / tube_inner_diameter_m)
                R_fouling_shell = fouling_factor_shell_m2K_W

                R_total = R_tube_inner + R_fouling_tube + R_wall + R_fouling_shell + R_shell
                U = 1 / R_total

                # Available area (based on tube OD)
                A_available = n_tubes * math.pi * tube_outer_diameter_m * tube_length_m

                # Required area
                A_required = abs(heat_duty_W) / (U * LMTD * F_correction)

                # Pressure drops - use fluids/ht library functions exclusively
                # Tube-side: using fluids library (no fallback)
                # Total tube length = tube_length * n_passes
                # Note: one_phase_dP expects mass flow through a SINGLE tube, not total
                dP_tube = one_phase_dP(
                    m=tube_flow / tubes_per_pass,  # Per-tube mass flow
                    rho=rho_tube,
                    mu=mu_tube,
                    D=tube_inner_diameter_m,
                    roughness=tube_roughness_m,
                    L=tube_length_m * n_tube_passes,
                )
                # Add entrance/exit losses (~1.5 velocity heads per pass)
                dP_tube += 1.5 * n_tube_passes * rho_tube * v_tube**2 / 2

                # Shell-side: use matching pressure drop correlation
                if shell_side_method == "Zukauskas":
                    # Use dP_Zukauskas to match Nu_Zukauskas_Bejan
                    # V_max was computed earlier for Re_shell calculation
                    dP_shell = dP_Zukauskas(
                        Re=Re_shell,
                        n=tube_rows,
                        ST=tube_pitch_m,  # Transverse pitch
                        SL=tube_pitch_m,  # Longitudinal pitch
                        D=tube_outer_diameter_m,
                        rho=rho_shell,
                        Vmax=V_max,  # Use V_max computed for tube bank, not v_shell
                    )
                else:  # Kern method
                    # Use dP_Kern to match Kern Nusselt correlation
                    dP_shell = dP_Kern(
                        m=shell_flow,
                        rho=rho_shell,
                        mu=mu_shell,
                        DShell=D_shell,
                        LSpacing=B,
                        pitch=tube_pitch_m,
                        Do=tube_outer_diameter_m,
                        NBaffles=N_baffles,
                    )

                # Check constraints
                thermal_satisfied = A_available >= A_required * 0.99
                hydraulic_satisfied = True

                if v_tube > max_velocity_tube_m_s:
                    hydraulic_satisfied = False
                if v_tube < min_velocity_tube_m_s:
                    hydraulic_satisfied = False
                if Re_tube < min_Re_tube:
                    hydraulic_satisfied = False
                if v_shell > max_velocity_shell_m_s:
                    hydraulic_satisfied = False
                if max_pressure_drop_tube_kPa and dP_tube / 1000 > max_pressure_drop_tube_kPa:
                    hydraulic_satisfied = False
                if max_pressure_drop_shell_kPa and dP_shell / 1000 > max_pressure_drop_shell_kPa:
                    hydraulic_satisfied = False

                # Calculate effectiveness and NTU
                C_hot = hot_mass_flow_kg_s * cp_hot
                C_cold = cold_mass_flow_kg_s * cp_cold
                C_min = min(C_hot, C_cold)
                C_max = max(C_hot, C_cold)
                Q_max = C_min * (hot_inlet_temp_K - cold_inlet_temp_K)
                effectiveness = abs(heat_duty_W) / Q_max if Q_max > 0 else 0
                NTU = U * A_available / C_min if C_min > 0 else 0

                result_summary = {
                    "n_tubes": n_tubes,
                    "thermal_satisfied": thermal_satisfied,
                    "hydraulic_satisfied": hydraulic_satisfied,
                    "area_available_m2": A_available,
                    "area_required_m2": A_required,
                    "area_margin_pct": (A_available / A_required - 1) * 100 if A_required > 0 else 0,
                    "U_W_m2K": U,
                    "dP_tube_kPa": dP_tube / 1000,
                    "dP_shell_kPa": dP_shell / 1000,
                    "v_tube_m_s": v_tube,
                    "v_shell_m_s": v_shell,
                }
                all_results.append(result_summary)

                # Select first result that satisfies both constraints
                if best_result is None and thermal_satisfied and hydraulic_satisfied:
                    # Build comprehensive result
                    best_result = {
                        "duty_W": abs(heat_duty_W),
                        "duty_kW": abs(heat_duty_W) / 1000,
                        "LMTD_K": LMTD,
                        "F_correction": F_correction,
                        "effectiveness": effectiveness,
                        "NTU": NTU,
                        "geometry": {
                            "type": "shell_tube",
                            "n_tubes": n_tubes,
                            "tubes_per_pass": tubes_per_pass,
                            "tube_length_m": tube_length_m,
                            "shell_diameter_m": D_shell,
                            "shell_diameter_specified": shell_inner_diameter_m is not None,
                            "max_tubes_from_shell": max_tubes_from_shell,
                            "bundle_diameter_m": D_bundle,
                            "n_baffles": N_baffles,
                            "baffle_spacing_m": B,
                            "baffle_cut_fraction": baffle_cut_fraction,
                            "tube_pitch_m": tube_pitch_m,
                            "tube_layout_angle": tube_layout_angle,
                            "area_per_tube_m2": math.pi * tube_outer_diameter_m * tube_length_m,
                            "total_area_m2": A_available,
                            "area_required_m2": A_required,
                            "area_margin_pct": (A_available / A_required - 1) * 100 if A_required > 0 else 0,
                            "tube_dimensions": {
                                "outer_diameter_m": tube_outer_diameter_m,
                                "inner_diameter_m": tube_inner_diameter_m,
                                "wall_thickness_m": tube_wall_thickness,
                            },
                        },
                        "thermal": {
                            "U_W_m2K": U,
                            "h_tube_W_m2K": h_tube,
                            "h_shell_W_m2K": h_shell,
                            "Nu_tube": Nu_tube,
                            "Nu_shell": Nu_shell,
                            "Re_tube": Re_tube,
                            "Re_shell": Re_shell,
                            "Pr_tube": Pr_tube,
                            "Pr_shell": Pr_shell,
                            "correlation_tube": tube_side_method,
                            "correlation_shell": shell_side_method,
                            "R_wall_m2K_W": R_wall,
                            "fouling_tube_m2K_W": fouling_factor_tube_m2K_W,
                            "fouling_shell_m2K_W": fouling_factor_shell_m2K_W,
                        },
                        "hydraulic": {
                            "velocity_tube_m_s": v_tube,
                            "velocity_shell_m_s": v_shell,
                            "pressure_drop_tube_Pa": dP_tube,
                            "pressure_drop_tube_kPa": dP_tube / 1000,
                            "pressure_drop_shell_Pa": dP_shell,
                            "pressure_drop_shell_kPa": dP_shell / 1000,
                            "correlation_tube_dP": "fluids.one_phase_dP",
                            "correlation_shell_dP": "ht.conv_tube_bank.dP_Zukauskas" if shell_side_method == "Zukauskas" else "ht.conv_tube_bank.dP_Kern",
                            "shell_cross_flow_area_m2": A_shell,
                            "equivalent_diameter_shell_m": De_shell,
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
                            "terminal_temp_diff_min_K": min(dT1, dT2),
                            "terminal_temp_diff_max_K": max(dT1, dT2),
                        },
                        "configuration": {
                            "n_tube_passes": n_tube_passes,
                            "n_shell_passes": n_shell_passes,
                            "tube_side_fluid": tube_side_fluid,
                            "flow_arrangement": f"{n_shell_passes}-{n_tube_passes} (shell-tube passes)",
                        },
                        "fluids": {
                            "tube_side": {
                                "name": tube_fluid,
                                "mass_flow_kg_s": tube_flow,
                                "density_kg_m3": rho_tube,
                                "viscosity_Pa_s": mu_tube,
                                "conductivity_W_mK": k_tube,
                                "specific_heat_J_kgK": cp_tube,
                            },
                            "shell_side": {
                                "name": shell_fluid,
                                "mass_flow_kg_s": shell_flow,
                                "density_kg_m3": rho_shell,
                                "viscosity_Pa_s": mu_shell,
                                "conductivity_W_mK": k_shell,
                                "specific_heat_J_kgK": cp_shell,
                            },
                        },
                        "heat_balance_verification": {
                            "Q_from_LMTD_kW": U * A_available * LMTD * F_correction / 1000,
                            "Q_from_hot_side_kW": hot_mass_flow_kg_s * cp_hot * (hot_inlet_temp_K - hot_outlet_temp_K) / 1000,
                            "Q_from_cold_side_kW": cold_mass_flow_kg_s
                            * cp_cold
                            * (cold_outlet_temp_K - cold_inlet_temp_K)
                            / 1000,
                            "balance_satisfied": abs(heat_duty_W - U * A_required * LMTD * F_correction)
                            / max(abs(heat_duty_W), 1)
                            < 0.05,
                        },
                    }

            except Exception as calc_error:
                logger.debug(f"Calculation failed for {n_tubes} tubes: {calc_error}")
                continue

        # Return results
        if best_result is not None:
            return json.dumps(best_result)
        elif all_results:
            # No solution found - provide diagnostic info
            # Find closest solutions
            thermal_ok = [r for r in all_results if r["thermal_satisfied"]]
            if thermal_ok:
                closest = min(thermal_ok, key=lambda r: r["dP_tube_kPa"] + r["dP_shell_kPa"])
                return json.dumps(
                    {
                        "error": "No configuration satisfies both thermal AND hydraulic constraints",
                        "closest_thermal_solution": closest,
                        "suggestion": "Try increasing max_pressure_drop_*_kPa or max_velocity_*_m_s",
                        "alternatives": [
                            "Increase tube length to reduce velocity",
                            "Use more tube passes to improve h_tube",
                            "Use larger shell diameter to reduce shell-side velocity",
                        ],
                    }
                )
            else:
                closest = min(all_results, key=lambda r: r["area_required_m2"] - r["area_available_m2"])
                return json.dumps(
                    {
                        "error": "No configuration satisfies thermal requirement within tube count range",
                        "closest_solution": closest,
                        "suggestion": "Try increasing max_tubes or tube_length_m",
                    }
                )
        else:
            return json.dumps(
                {
                    "error": "Could not evaluate any tube configurations",
                    "suggestion": "Check tube geometry parameters and pass configuration",
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
        logger.error(f"Error in size_shell_tube_heat_exchanger: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
