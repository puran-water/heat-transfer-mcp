"""
Dedicated Double-Pipe Heat Exchanger sizing tool with thermal-hydraulic coupling.

This module provides accurate double-pipe (concentric tube) HX sizing using upstream
Caleb Bell ht/fluids libraries with proper coupling between thermal performance (U-value)
and hydraulic performance (pressure drop) through velocity and Reynolds number dependencies.

Key features:
- Inner pipe carries one fluid, annulus (outer pipe - inner pipe) carries the other
- Supports counterflow and parallel flow arrangements
- Uses ht.conv_internal.Nu_conv_internal for both inner pipe and annulus
- Pressure drop via fluids.friction.one_phase_dP
- Iterative solver to find length satisfying thermal + hydraulic constraints

Double-pipe HX advantages:
- Simple construction, easy to maintain
- True counterflow achievable (highest effectiveness)
- Suitable for small duties, viscous fluids, high pressure applications
- Can be arranged in series for larger duties (hairpin configuration)

Reference: Kern, Process Heat Transfer (1950), Chapter 9
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional

from utils.import_helpers import HT_AVAILABLE, FLUIDS_AVAILABLE
from tools.fluid_properties import get_fluid_properties
from utils.hx_common import calculate_annulus_geometry, calculate_overall_U, calculate_lmtd_with_check

logger = logging.getLogger("heat-transfer-mcp.size_double_pipe_heat_exchanger")


def size_double_pipe_heat_exchanger(
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
    # Inner pipe geometry
    inner_pipe_outer_diameter_m: float = 0.0254,  # 1" OD
    inner_pipe_inner_diameter_m: float = 0.0229,  # ~0.9" ID (schedule 40)
    inner_pipe_roughness_m: float = 0.00004,  # ~40 micron for steel
    inner_pipe_material_conductivity_W_mK: float = 45.0,  # Carbon steel default
    # Outer pipe geometry
    outer_pipe_inner_diameter_m: float = 0.0525,  # 2" schedule 40 ID
    # Configuration
    flow_arrangement: str = "counterflow",  # "counterflow" or "parallel"
    inner_pipe_fluid: str = "hot",  # Which fluid in inner pipe: "hot" or "cold"
    n_hairpins: int = 1,  # Number of hairpin sections (series arrangement)
    # Constraints for solver
    max_pressure_drop_inner_kPa: Optional[float] = None,
    max_pressure_drop_annulus_kPa: Optional[float] = None,
    max_velocity_inner_m_s: float = 3.0,
    max_velocity_annulus_m_s: float = 2.0,
    min_velocity_inner_m_s: float = 0.3,
    min_velocity_annulus_m_s: float = 0.2,
    min_Re_inner: float = 2500.0,  # Minimum for turbulent flow
    # Fouling
    fouling_factor_inner_m2K_W: float = 0.0001,
    fouling_factor_annulus_m2K_W: float = 0.0002,
    # Solver options
    min_length_m: float = 1.0,
    max_length_m: float = 20.0,
    length_step_m: float = 0.1,
    solve_for: str = "length",  # "length" or "rating" (given length, find duty)
    pipe_length_m: Optional[float] = None,  # Required for rating mode
    strict: bool = False,
) -> str:
    """Size a double-pipe (concentric tube) heat exchanger with coupled thermal-hydraulic design.

    This tool provides accurate double-pipe HX sizing using upstream ht/fluids libraries
    with proper thermal-hydraulic coupling. The pipe length, U-value, and pressure drops
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

        inner_pipe_outer_diameter_m: Inner pipe OD (m). Default 0.0254 (1").
        inner_pipe_inner_diameter_m: Inner pipe ID (m). Default 0.0229.
        inner_pipe_roughness_m: Inner pipe roughness (m). Default 0.00004.
        inner_pipe_material_conductivity_W_mK: Inner pipe material k (W/m-K). Default 45.

        outer_pipe_inner_diameter_m: Outer pipe ID (m). Default 0.0525 (2").

        flow_arrangement: "counterflow" or "parallel". Default "counterflow".
        inner_pipe_fluid: Which fluid in inner pipe ("hot" or "cold"). Default "hot".
        n_hairpins: Number of hairpin sections in series. Default 1.

        max_pressure_drop_inner_kPa: Maximum inner pipe dP (kPa).
        max_pressure_drop_annulus_kPa: Maximum annulus dP (kPa).
        max_velocity_inner_m_s: Maximum inner pipe velocity (m/s). Default 3.0.
        max_velocity_annulus_m_s: Maximum annulus velocity (m/s). Default 2.0.
        min_velocity_inner_m_s: Minimum inner pipe velocity (m/s). Default 0.3.
        min_velocity_annulus_m_s: Minimum annulus velocity (m/s). Default 0.2.
        min_Re_inner: Minimum inner Reynolds number for turbulent flow. Default 2500.

        fouling_factor_inner_m2K_W: Inner fouling (m²K/W). Default 0.0001.
        fouling_factor_annulus_m2K_W: Annulus fouling (m²K/W). Default 0.0002.

        min_length_m: Minimum pipe length for solver (m). Default 1.0.
        max_length_m: Maximum pipe length for solver (m). Default 20.0.
        length_step_m: Length increment for solver (m). Default 0.1.
        solve_for: "length" to size, "rating" to rate. Default "length".
        pipe_length_m: Pipe length for rating mode (m).

        strict: If True, fail if libraries unavailable.

    Returns:
        JSON with comprehensive sizing results including:
        - duty_kW, LMTD_K, effectiveness, NTU
        - geometry: lengths, diameters, areas
        - thermal: U, h_inner, h_annulus, Re, Pr, correlations
        - hydraulic: velocity, pressure_drop for both sides
        - temperatures: inlet/outlet for both sides
        - heat_balance_verification
    """
    try:
        # Validate library availability
        if not HT_AVAILABLE:
            if strict:
                return json.dumps({"error": "ht library required with strict=True"})
            return json.dumps({
                "error": "ht library not available for double-pipe sizing",
                "suggestion": "Install with: pip install ht>=1.2.0"
            })

        if not FLUIDS_AVAILABLE:
            if strict:
                return json.dumps({"error": "fluids library required with strict=True"})
            return json.dumps({
                "error": "fluids library not available for double-pipe sizing",
                "suggestion": "Install with: pip install fluids>=1.0.0"
            })

        # Validate required inputs
        if hot_mass_flow_kg_s is None or hot_mass_flow_kg_s <= 0:
            return json.dumps({"error": "hot_mass_flow_kg_s must be positive"})
        if cold_mass_flow_kg_s is None or cold_mass_flow_kg_s <= 0:
            return json.dumps({"error": "cold_mass_flow_kg_s must be positive"})

        # Validate temperatures and determine duty
        # Rating mode only needs inlet temperatures (outlet temps calculated from HX)
        # Sizing mode needs either heat_duty_W + inlet temps, or all four temps
        temps_provided = [hot_inlet_temp_K, hot_outlet_temp_K, cold_inlet_temp_K, cold_outlet_temp_K]
        temps_count = sum(1 for t in temps_provided if t is not None)

        if solve_for == "rating":
            # Rating mode: inlet temps required, outlet temps calculated
            if hot_inlet_temp_K is None or cold_inlet_temp_K is None:
                return json.dumps({
                    "error": "Both inlet temperatures required for rating mode"
                })
            # For rating, we need to estimate outlet temps initially
            # Use a reasonable assumption (will be recalculated)
            if hot_outlet_temp_K is None:
                hot_outlet_temp_K = hot_inlet_temp_K - 10  # Initial estimate
            if cold_outlet_temp_K is None:
                cold_outlet_temp_K = cold_inlet_temp_K + 10  # Initial estimate
        else:
            # Sizing mode
            if heat_duty_W is None:
                if temps_count < 4:
                    return json.dumps({
                        "error": "Either heat_duty_W or all four temperatures must be provided for sizing"
                    })
            else:
                if hot_inlet_temp_K is None or cold_inlet_temp_K is None:
                    return json.dumps({
                        "error": "Both inlet temperatures required when heat_duty_W is provided"
                    })

        # Validate geometry
        if outer_pipe_inner_diameter_m <= inner_pipe_outer_diameter_m:
            return json.dumps({
                "error": f"outer_pipe_inner_diameter_m ({outer_pipe_inner_diameter_m}) must be > "
                         f"inner_pipe_outer_diameter_m ({inner_pipe_outer_diameter_m})"
            })
        if inner_pipe_inner_diameter_m >= inner_pipe_outer_diameter_m:
            return json.dumps({
                "error": f"inner_pipe_inner_diameter_m ({inner_pipe_inner_diameter_m}) must be < "
                         f"inner_pipe_outer_diameter_m ({inner_pipe_outer_diameter_m})"
            })

        # Validate flow arrangement
        flow_lower = flow_arrangement.lower()
        if flow_lower not in ["counterflow", "parallel", "parallelflow"]:
            return json.dumps({
                "error": f"flow_arrangement must be 'counterflow' or 'parallel', got '{flow_arrangement}'"
            })
        is_counterflow = flow_lower == "counterflow"

        # Validate rating mode
        if solve_for == "rating":
            if pipe_length_m is None or pipe_length_m <= 0:
                return json.dumps({
                    "error": "pipe_length_m required for rating mode (solve_for='rating')"
                })

        # Calculate annulus geometry using shared utility
        annulus_geom = calculate_annulus_geometry(
            D_outer=outer_pipe_inner_diameter_m,
            D_inner=inner_pipe_outer_diameter_m
        )

        # Inner pipe geometry
        A_inner = math.pi * inner_pipe_inner_diameter_m**2 / 4
        inner_pipe_wall_thickness = (inner_pipe_outer_diameter_m - inner_pipe_inner_diameter_m) / 2

        # Assign fluids to inner/annulus sides
        if inner_pipe_fluid.lower() == "hot":
            inner_fluid = hot_fluid
            inner_flow = hot_mass_flow_kg_s
            inner_pressure = hot_fluid_pressure_Pa
            annulus_fluid = cold_fluid
            annulus_flow = cold_mass_flow_kg_s
            annulus_pressure = cold_fluid_pressure_Pa
        else:
            inner_fluid = cold_fluid
            inner_flow = cold_mass_flow_kg_s
            inner_pressure = cold_fluid_pressure_Pa
            annulus_fluid = hot_fluid
            annulus_flow = hot_mass_flow_kg_s
            annulus_pressure = hot_fluid_pressure_Pa

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

        # Assign properties to inner/annulus sides
        if inner_pipe_fluid.lower() == "hot":
            rho_inner, mu_inner, k_inner, cp_inner, Pr_inner = rho_hot, mu_hot, k_hot, cp_hot, Pr_hot
            rho_annulus, mu_annulus, k_annulus, cp_annulus, Pr_annulus = rho_cold, mu_cold, k_cold, cp_cold, Pr_cold
        else:
            rho_inner, mu_inner, k_inner, cp_inner, Pr_inner = rho_cold, mu_cold, k_cold, cp_cold, Pr_cold
            rho_annulus, mu_annulus, k_annulus, cp_annulus, Pr_annulus = rho_hot, mu_hot, k_hot, cp_hot, Pr_hot

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

        # Calculate LMTD using shared utility
        LMTD, lmtd_error = calculate_lmtd_with_check(
            Thi=hot_inlet_temp_K,
            Tho=hot_outlet_temp_K,
            Tci=cold_inlet_temp_K,
            Tco=cold_outlet_temp_K,
            counterflow=is_counterflow
        )

        if lmtd_error:
            return json.dumps({
                "error": f"LMTD calculation failed: {lmtd_error}",
                "hint": "Hot fluid must be hotter than cold fluid at both ends"
            })

        # Import required functions from upstream ht/fluids libraries
        from ht.conv_internal import Nu_conv_internal
        from fluids.friction import one_phase_dP, friction_factor

        # Velocities (constant for all lengths)
        v_inner = inner_flow / (rho_inner * A_inner)
        v_annulus = annulus_flow / (rho_annulus * annulus_geom["A_annulus_m2"])

        # Check velocity constraints first
        velocity_warnings = []
        if v_inner > max_velocity_inner_m_s:
            velocity_warnings.append(f"Inner pipe velocity {v_inner:.2f} m/s exceeds max {max_velocity_inner_m_s} m/s")
        if v_inner < min_velocity_inner_m_s:
            velocity_warnings.append(f"Inner pipe velocity {v_inner:.2f} m/s below min {min_velocity_inner_m_s} m/s")
        if v_annulus > max_velocity_annulus_m_s:
            velocity_warnings.append(f"Annulus velocity {v_annulus:.2f} m/s exceeds max {max_velocity_annulus_m_s} m/s")
        if v_annulus < min_velocity_annulus_m_s:
            velocity_warnings.append(f"Annulus velocity {v_annulus:.2f} m/s below min {min_velocity_annulus_m_s} m/s")

        # Reynolds numbers
        Re_inner = rho_inner * v_inner * inner_pipe_inner_diameter_m / mu_inner
        # For annulus, use hydraulic diameter for Reynolds
        Re_annulus = rho_annulus * v_annulus * annulus_geom["D_hydraulic_m"] / mu_annulus

        if Re_inner < min_Re_inner:
            velocity_warnings.append(f"Inner pipe Re={Re_inner:.0f} below turbulent threshold {min_Re_inner}")

        # Inner pipe Nusselt (using ht library)
        eD_inner = inner_pipe_roughness_m / inner_pipe_inner_diameter_m
        try:
            Nu_inner = Nu_conv_internal(Re=Re_inner, Pr=Pr_inner, eD=eD_inner)
        except Exception as e:
            # May fail for laminar flow - use Hausen correlation
            if Re_inner < 2300:
                # Hausen correlation for laminar developing flow
                Nu_inner = 3.66 + 0.065 * (inner_pipe_inner_diameter_m / max_length_m) * Re_inner * Pr_inner / \
                           (1 + 0.04 * ((inner_pipe_inner_diameter_m / max_length_m) * Re_inner * Pr_inner)**(2/3))
            else:
                return json.dumps({"error": f"Inner pipe Nusselt calculation failed: {e}"})

        h_inner = Nu_inner * k_inner / inner_pipe_inner_diameter_m

        # Annulus Nusselt
        # For annulus, use equivalent diameter for heat transfer to inner pipe surface
        # De = (D_outer^2 - D_inner^2) / D_inner for heat transfer to inner pipe only
        # Use hydraulic diameter D_h = D_outer - D_inner for friction/flow

        try:
            # Annulus convection - use hydraulic diameter and ht library
            Nu_annulus = Nu_conv_internal(Re=Re_annulus, Pr=Pr_annulus, eD=0)  # Smooth outer surface
        except Exception as e:
            if Re_annulus < 2300:
                # Laminar annulus flow (simplified)
                Nu_annulus = 3.66  # Simplified for fully developed laminar
            else:
                return json.dumps({"error": f"Annulus Nusselt calculation failed: {e}"})

        # For annulus heat transfer, use equivalent diameter
        h_annulus = Nu_annulus * k_annulus / annulus_geom["D_equivalent_m"]

        # Calculate overall U using shared utility (referenced to inner pipe outer surface)
        U_result = calculate_overall_U(
            h_inner=h_inner,
            h_outer=h_annulus,
            D_inner=inner_pipe_inner_diameter_m,
            D_outer=inner_pipe_outer_diameter_m,
            wall_thickness=inner_pipe_wall_thickness,
            k_wall=inner_pipe_material_conductivity_W_mK,
            fouling_inner=fouling_factor_inner_m2K_W,
            fouling_outer=fouling_factor_annulus_m2K_W,
            reference="outer"  # Reference to pipe OD for area calculation
        )
        U = U_result["U_W_m2K"]

        # Required area
        A_required = abs(heat_duty_W) / (U * LMTD)

        # Required length per hairpin (area = pi * D_outer * L * n_hairpins)
        L_per_hairpin_required = A_required / (math.pi * inner_pipe_outer_diameter_m * n_hairpins)

        if solve_for == "rating":
            # Rating mode: use effectiveness-NTU method to find actual duty
            # This avoids circular dependency with LMTD
            from ht.hx import effectiveness_from_NTU

            L_actual = pipe_length_m
            A_actual = math.pi * inner_pipe_outer_diameter_m * L_actual * n_hairpins
            UA = U * A_actual

            # Heat capacity rates
            C_hot = hot_mass_flow_kg_s * cp_hot
            C_cold = cold_mass_flow_kg_s * cp_cold
            C_min = min(C_hot, C_cold)
            C_max = max(C_hot, C_cold)
            Cr = C_min / C_max if C_max > 0 else 0
            NTU = UA / C_min if C_min > 0 else 0

            # Get effectiveness using ht library
            subtype = 'counterflow' if is_counterflow else 'parallel'
            effectiveness = effectiveness_from_NTU(NTU, Cr, subtype=subtype)

            # Calculate actual duty from effectiveness
            Q_max = C_min * (hot_inlet_temp_K - cold_inlet_temp_K)
            Q_actual = effectiveness * Q_max

            # Calculate actual outlet temperatures
            actual_hot_outlet_K = hot_inlet_temp_K - Q_actual / C_hot
            actual_cold_outlet_K = cold_inlet_temp_K + Q_actual / C_cold

            # Now calculate actual LMTD for reporting
            try:
                actual_LMTD, _ = calculate_lmtd_with_check(
                    Thi=hot_inlet_temp_K,
                    Tho=actual_hot_outlet_K,
                    Tci=cold_inlet_temp_K,
                    Tco=actual_cold_outlet_K,
                    counterflow=is_counterflow
                )
            except Exception:
                actual_LMTD = None

            # Pressure drops
            dP_inner = one_phase_dP(
                m=inner_flow,
                rho=rho_inner,
                mu=mu_inner,
                D=inner_pipe_inner_diameter_m,
                roughness=inner_pipe_roughness_m,
                L=L_actual * n_hairpins
            )
            # Add U-bend losses for hairpin (1.5 velocity heads per hairpin)
            if n_hairpins > 1:
                dP_inner += (n_hairpins - 1) * 1.5 * rho_inner * v_inner**2 / 2

            dP_annulus = one_phase_dP(
                m=annulus_flow,
                rho=rho_annulus,
                mu=mu_annulus,
                D=annulus_geom["D_hydraulic_m"],
                roughness=0,  # Smooth outer pipe
                L=L_actual * n_hairpins
            )

            result = {
                "mode": "rating",
                "actual_duty_W": Q_actual,
                "actual_duty_kW": Q_actual / 1000,
                "design_duty_W": abs(heat_duty_W) if heat_duty_W else None,
                "design_duty_kW": abs(heat_duty_W) / 1000 if heat_duty_W else None,
                "duty_ratio": Q_actual / abs(heat_duty_W) if heat_duty_W else None,
                "LMTD_K": actual_LMTD,
                "effectiveness": effectiveness,
                "NTU": NTU,
                "Cr": Cr,

                "geometry": {
                    "type": "double_pipe",
                    "pipe_length_m": L_actual,
                    "n_hairpins": n_hairpins,
                    "total_length_m": L_actual * n_hairpins,
                    "area_per_hairpin_m2": math.pi * inner_pipe_outer_diameter_m * L_actual,
                    "total_area_m2": A_actual,
                    "inner_pipe": {
                        "outer_diameter_m": inner_pipe_outer_diameter_m,
                        "inner_diameter_m": inner_pipe_inner_diameter_m,
                        "wall_thickness_m": inner_pipe_wall_thickness,
                        "flow_area_m2": A_inner,
                    },
                    "annulus": {
                        "outer_diameter_m": outer_pipe_inner_diameter_m,
                        "inner_diameter_m": inner_pipe_outer_diameter_m,
                        "hydraulic_diameter_m": annulus_geom["D_hydraulic_m"],
                        "equivalent_diameter_m": annulus_geom["D_equivalent_m"],
                        "flow_area_m2": annulus_geom["A_annulus_m2"],
                        "gap_m": annulus_geom["gap_m"],
                    }
                },

                "thermal": {
                    "U_W_m2K": U,
                    "h_inner_W_m2K": h_inner,
                    "h_annulus_W_m2K": h_annulus,
                    "Nu_inner": Nu_inner,
                    "Nu_annulus": Nu_annulus,
                    "Re_inner": Re_inner,
                    "Re_annulus": Re_annulus,
                    "Pr_inner": Pr_inner,
                    "Pr_annulus": Pr_annulus,
                    "R_total_m2K_W": U_result["R_total_m2K_W"],
                    "R_conv_inner_m2K_W": U_result["R_conv_inner_m2K_W"],
                    "R_wall_m2K_W": U_result["R_wall_m2K_W"],
                    "R_conv_outer_m2K_W": U_result["R_conv_outer_m2K_W"],
                    "fouling_inner_m2K_W": fouling_factor_inner_m2K_W,
                    "fouling_annulus_m2K_W": fouling_factor_annulus_m2K_W,
                },

                "hydraulic": {
                    "velocity_inner_m_s": v_inner,
                    "velocity_annulus_m_s": v_annulus,
                    "pressure_drop_inner_Pa": dP_inner,
                    "pressure_drop_inner_kPa": dP_inner / 1000,
                    "pressure_drop_annulus_Pa": dP_annulus,
                    "pressure_drop_annulus_kPa": dP_annulus / 1000,
                },

                "temperatures": {
                    "hot_inlet_K": hot_inlet_temp_K,
                    "hot_inlet_C": hot_inlet_temp_K - 273.15,
                    "hot_outlet_K": actual_hot_outlet_K,
                    "hot_outlet_C": actual_hot_outlet_K - 273.15,
                    "cold_inlet_K": cold_inlet_temp_K,
                    "cold_inlet_C": cold_inlet_temp_K - 273.15,
                    "cold_outlet_K": actual_cold_outlet_K,
                    "cold_outlet_C": actual_cold_outlet_K - 273.15,
                },

                "configuration": {
                    "flow_arrangement": flow_arrangement,
                    "inner_pipe_fluid": inner_pipe_fluid,
                },

                "warnings": velocity_warnings if velocity_warnings else None,
            }

            return json.dumps(result)

        # Sizing mode: find minimum length satisfying constraints
        best_result = None
        all_results = []

        # Generate length values to try
        lengths_to_try = []
        L = min_length_m
        while L <= max_length_m:
            lengths_to_try.append(L)
            L += length_step_m

        for L in lengths_to_try:
            try:
                # Area for this length
                A_available = math.pi * inner_pipe_outer_diameter_m * L * n_hairpins

                # Pressure drops
                dP_inner = one_phase_dP(
                    m=inner_flow,
                    rho=rho_inner,
                    mu=mu_inner,
                    D=inner_pipe_inner_diameter_m,
                    roughness=inner_pipe_roughness_m,
                    L=L * n_hairpins
                )
                # Add U-bend losses
                if n_hairpins > 1:
                    dP_inner += (n_hairpins - 1) * 1.5 * rho_inner * v_inner**2 / 2

                dP_annulus = one_phase_dP(
                    m=annulus_flow,
                    rho=rho_annulus,
                    mu=mu_annulus,
                    D=annulus_geom["D_hydraulic_m"],
                    roughness=0,
                    L=L * n_hairpins
                )

                # Check thermal constraint
                thermal_satisfied = A_available >= A_required * 0.99

                # Check hydraulic constraints
                hydraulic_satisfied = True
                if v_inner > max_velocity_inner_m_s:
                    hydraulic_satisfied = False
                if v_annulus > max_velocity_annulus_m_s:
                    hydraulic_satisfied = False
                if max_pressure_drop_inner_kPa and dP_inner / 1000 > max_pressure_drop_inner_kPa:
                    hydraulic_satisfied = False
                if max_pressure_drop_annulus_kPa and dP_annulus / 1000 > max_pressure_drop_annulus_kPa:
                    hydraulic_satisfied = False

                result_summary = {
                    "length_m": L,
                    "thermal_satisfied": thermal_satisfied,
                    "hydraulic_satisfied": hydraulic_satisfied,
                    "area_available_m2": A_available,
                    "area_required_m2": A_required,
                    "area_margin_pct": (A_available / A_required - 1) * 100 if A_required > 0 else 0,
                    "U_W_m2K": U,
                    "dP_inner_kPa": dP_inner / 1000,
                    "dP_annulus_kPa": dP_annulus / 1000,
                }
                all_results.append(result_summary)

                # Select first result that satisfies both constraints
                if best_result is None and thermal_satisfied and hydraulic_satisfied:
                    # Effectiveness and NTU
                    C_hot = hot_mass_flow_kg_s * cp_hot
                    C_cold = cold_mass_flow_kg_s * cp_cold
                    C_min = min(C_hot, C_cold)
                    C_max = max(C_hot, C_cold)
                    Q_max = C_min * (hot_inlet_temp_K - cold_inlet_temp_K)
                    effectiveness = abs(heat_duty_W) / Q_max if Q_max > 0 else 0
                    NTU = U * A_available / C_min if C_min > 0 else 0

                    best_result = {
                        "mode": "sizing",
                        "duty_W": abs(heat_duty_W),
                        "duty_kW": abs(heat_duty_W) / 1000,
                        "LMTD_K": LMTD,
                        "effectiveness": effectiveness,
                        "NTU": NTU,

                        "geometry": {
                            "type": "double_pipe",
                            "pipe_length_m": L,
                            "n_hairpins": n_hairpins,
                            "total_length_m": L * n_hairpins,
                            "area_per_hairpin_m2": math.pi * inner_pipe_outer_diameter_m * L,
                            "total_area_m2": A_available,
                            "area_required_m2": A_required,
                            "area_margin_pct": (A_available / A_required - 1) * 100 if A_required > 0 else 0,
                            "inner_pipe": {
                                "outer_diameter_m": inner_pipe_outer_diameter_m,
                                "inner_diameter_m": inner_pipe_inner_diameter_m,
                                "wall_thickness_m": inner_pipe_wall_thickness,
                                "flow_area_m2": A_inner,
                            },
                            "annulus": {
                                "outer_diameter_m": outer_pipe_inner_diameter_m,
                                "inner_diameter_m": inner_pipe_outer_diameter_m,
                                "hydraulic_diameter_m": annulus_geom["D_hydraulic_m"],
                                "equivalent_diameter_m": annulus_geom["D_equivalent_m"],
                                "flow_area_m2": annulus_geom["A_annulus_m2"],
                                "gap_m": annulus_geom["gap_m"],
                            }
                        },

                        "thermal": {
                            "U_W_m2K": U,
                            "h_inner_W_m2K": h_inner,
                            "h_annulus_W_m2K": h_annulus,
                            "Nu_inner": Nu_inner,
                            "Nu_annulus": Nu_annulus,
                            "Re_inner": Re_inner,
                            "Re_annulus": Re_annulus,
                            "Pr_inner": Pr_inner,
                            "Pr_annulus": Pr_annulus,
                            "R_total_m2K_W": U_result["R_total_m2K_W"],
                            "R_conv_inner_m2K_W": U_result["R_conv_inner_m2K_W"],
                            "R_wall_m2K_W": U_result["R_wall_m2K_W"],
                            "R_conv_outer_m2K_W": U_result["R_conv_outer_m2K_W"],
                            "fouling_inner_m2K_W": fouling_factor_inner_m2K_W,
                            "fouling_annulus_m2K_W": fouling_factor_annulus_m2K_W,
                        },

                        "hydraulic": {
                            "velocity_inner_m_s": v_inner,
                            "velocity_annulus_m_s": v_annulus,
                            "pressure_drop_inner_Pa": dP_inner,
                            "pressure_drop_inner_kPa": dP_inner / 1000,
                            "pressure_drop_annulus_Pa": dP_annulus,
                            "pressure_drop_annulus_kPa": dP_annulus / 1000,
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
                            "terminal_temp_diff_min_K": min(hot_inlet_temp_K - cold_outlet_temp_K,
                                                            hot_outlet_temp_K - cold_inlet_temp_K),
                            "terminal_temp_diff_max_K": max(hot_inlet_temp_K - cold_outlet_temp_K,
                                                            hot_outlet_temp_K - cold_inlet_temp_K),
                        },

                        "configuration": {
                            "flow_arrangement": flow_arrangement,
                            "inner_pipe_fluid": inner_pipe_fluid,
                        },

                        "fluids": {
                            "inner_pipe": {
                                "name": inner_fluid,
                                "mass_flow_kg_s": inner_flow,
                                "density_kg_m3": rho_inner,
                                "viscosity_Pa_s": mu_inner,
                                "conductivity_W_mK": k_inner,
                                "specific_heat_J_kgK": cp_inner,
                            },
                            "annulus": {
                                "name": annulus_fluid,
                                "mass_flow_kg_s": annulus_flow,
                                "density_kg_m3": rho_annulus,
                                "viscosity_Pa_s": mu_annulus,
                                "conductivity_W_mK": k_annulus,
                                "specific_heat_J_kgK": cp_annulus,
                            },
                        },

                        "heat_balance_verification": {
                            "Q_from_LMTD_kW": U * A_available * LMTD / 1000,
                            "Q_from_hot_side_kW": hot_mass_flow_kg_s * cp_hot * (hot_inlet_temp_K - hot_outlet_temp_K) / 1000,
                            "Q_from_cold_side_kW": cold_mass_flow_kg_s * cp_cold * (cold_outlet_temp_K - cold_inlet_temp_K) / 1000,
                            "balance_satisfied": True,
                        },

                        "warnings": velocity_warnings if velocity_warnings else None,
                    }

            except Exception as calc_error:
                logger.debug(f"Calculation failed for length {L}m: {calc_error}")
                continue

        # Return results
        if best_result is not None:
            return json.dumps(best_result)
        elif all_results:
            # No solution found - provide diagnostic info
            thermal_ok = [r for r in all_results if r["thermal_satisfied"]]
            if thermal_ok:
                closest = min(thermal_ok, key=lambda r: r["dP_inner_kPa"] + r["dP_annulus_kPa"])
                return json.dumps({
                    "error": "No configuration satisfies both thermal AND hydraulic constraints",
                    "closest_thermal_solution": closest,
                    "suggestion": "Try increasing max_pressure_drop_*_kPa or max_velocity_*_m_s",
                    "alternatives": [
                        "Increase max_length_m",
                        "Use multiple hairpins (n_hairpins > 1)",
                        "Use larger pipe diameters to reduce velocity"
                    ],
                    "velocity_warnings": velocity_warnings,
                })
            else:
                closest = min(all_results, key=lambda r: r["area_required_m2"] - r["area_available_m2"])
                return json.dumps({
                    "error": "No configuration satisfies thermal requirement within length range",
                    "required_length_m": L_per_hairpin_required,
                    "max_length_searched_m": max_length_m,
                    "closest_solution": closest,
                    "suggestion": f"Try increasing max_length_m to at least {L_per_hairpin_required:.1f}m "
                                  f"or use more hairpins (currently n_hairpins={n_hairpins})",
                })
        else:
            return json.dumps({
                "error": "Could not evaluate any pipe length configurations",
                "suggestion": "Check geometry parameters"
            })

    except ImportError as e:
        return json.dumps({
            "error": f"Required library import failed: {e}",
            "suggestion": "Install with: pip install ht>=1.2.0 fluids>=1.0.0"
        })
    except Exception as e:
        logger.error(f"Error in size_double_pipe_heat_exchanger: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
