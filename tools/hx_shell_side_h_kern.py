"""
Heat exchanger shell-side heat transfer coefficient calculation tool.

This module provides functionality to estimate the shell-side heat transfer coefficient (ho)
for a Segmental Baffled Shell-and-Tube Heat Exchanger using Kern's method.

Also provides TEMA standard functions for tube sizing and bundle layout.
"""

import json
import logging
import math
from typing import Dict, Optional, Any, Tuple

from utils.import_helpers import HT_AVAILABLE
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.hx_shell_side_h_kern")


# ========================== TEMA Helper Functions ========================== #


def validate_tema_tubing(nps: float, bwg: int) -> Dict[str, Any]:
    """
    Validate if a tube size is TEMA-compliant using ht.hx functions.

    Args:
        nps: Nominal Pipe Size (e.g., 0.75 for 3/4")
        bwg: Birmingham Wire Gauge (e.g., 14, 16, 18)

    Returns:
        Dict with validation result and tube dimensions if valid
    """
    if not HT_AVAILABLE:
        return {"error": "ht library required for TEMA validation", "fallback": "Cannot validate without ht library"}

    try:
        from ht.hx import check_tubing_TEMA, get_tube_TEMA

        is_valid = check_tubing_TEMA(nps, bwg)

        if is_valid:
            # Get tube dimensions
            tube_data = get_tube_TEMA(NPS=nps, BWG=bwg)
            return {
                "tema_compliant": True,
                "nps": nps,
                "bwg": bwg,
                "outer_diameter_m": tube_data[2],  # Do
                "inner_diameter_m": tube_data[3],  # Di
                "wall_thickness_m": tube_data[4],  # t
                "source": "ht.hx.get_tube_TEMA",
            }
        else:
            return {
                "tema_compliant": False,
                "nps": nps,
                "bwg": bwg,
                "message": f"NPS={nps}, BWG={bwg} is not a standard TEMA tube size",
            }
    except Exception as e:
        return {"error": f"TEMA validation failed: {e}"}


def get_tema_tube_by_od(target_od_m: float, min_thickness_m: Optional[float] = None) -> Dict[str, Any]:
    """
    Find a TEMA-compliant tube close to a target outer diameter.

    Args:
        target_od_m: Target outer diameter in meters
        min_thickness_m: Minimum wall thickness in meters (optional)

    Returns:
        Dict with closest matching TEMA tube specifications
    """
    if not HT_AVAILABLE:
        return {"error": "ht library required for TEMA tube lookup"}

    try:
        from ht.hx import get_tube_TEMA

        # get_tube_TEMA can do fuzzy matching with Do and tmin
        if min_thickness_m:
            tube_data = get_tube_TEMA(Do=target_od_m, tmin=min_thickness_m)
        else:
            tube_data = get_tube_TEMA(Do=target_od_m)

        return {
            "nps": tube_data[0],
            "bwg": tube_data[1],
            "outer_diameter_m": tube_data[2],
            "inner_diameter_m": tube_data[3],
            "wall_thickness_m": tube_data[4],
            "source": "ht.hx.get_tube_TEMA (fuzzy match)",
        }
    except Exception as e:
        return {"error": f"TEMA tube lookup failed: {e}"}


def get_minimum_shell_diameter(tube_od_m: float) -> Dict[str, Any]:
    """
    Get minimum shell diameter for a given tube outer diameter.

    Uses ht.hx.DBundle_min for initial sizing when only tube diameter is known.

    Args:
        tube_od_m: Tube outer diameter in meters

    Returns:
        Dict with minimum shell/bundle diameter
    """
    if not HT_AVAILABLE:
        # Fallback: Rule of thumb - minimum 4 tubes across
        return {
            "min_shell_diameter_m": tube_od_m * 6,
            "source": "fallback (6x tube OD)",
            "note": "Install ht library for accurate TEMA sizing",
        }

    try:
        from ht.hx import DBundle_min

        d_min = DBundle_min(tube_od_m)
        return {"min_shell_diameter_m": d_min, "tube_od_m": tube_od_m, "source": "ht.hx.DBundle_min"}
    except Exception as e:
        return {"error": f"Minimum shell diameter calculation failed: {e}"}


def get_shell_bundle_clearance(bundle_diameter_m: float, shell_diameter_m: float) -> Dict[str, Any]:
    """
    Get shell-to-bundle clearance for different head types.

    Args:
        bundle_diameter_m: Tube bundle outer diameter in meters
        shell_diameter_m: Shell inner diameter in meters

    Returns:
        Dict with clearance information
    """
    if not HT_AVAILABLE:
        clearance = shell_diameter_m - bundle_diameter_m
        return {
            "clearance_m": clearance,
            "bundle_diameter_m": bundle_diameter_m,
            "shell_diameter_m": shell_diameter_m,
            "source": "direct calculation",
            "note": "Install ht library for TEMA-specific clearances",
        }

    try:
        from ht.hx import shell_clearance

        # Get clearances for different head types
        clearance_fixed = shell_clearance(bundle_diameter_m, shell_diameter_m, fixed=True)
        clearance_floating = shell_clearance(bundle_diameter_m, shell_diameter_m, fixed=False)

        return {
            "bundle_diameter_m": bundle_diameter_m,
            "shell_diameter_m": shell_diameter_m,
            "clearance_fixed_tubesheet_m": clearance_fixed,
            "clearance_floating_head_m": clearance_floating,
            "source": "ht.hx.shell_clearance",
        }
    except Exception as e:
        return {"error": f"Shell clearance calculation failed: {e}"}


def get_baffle_thickness(shell_diameter_m: float, unsupported_length_m: float, service: str = "normal") -> Dict[str, Any]:
    """
    Get recommended baffle thickness per TEMA standards.

    Args:
        shell_diameter_m: Shell inner diameter in meters
        unsupported_length_m: Unsupported tube span in meters
        service: Service type ('normal' or 'severe')

    Returns:
        Dict with recommended baffle thickness
    """
    if not HT_AVAILABLE:
        # Fallback: Approximate based on shell diameter
        if shell_diameter_m < 0.4:
            thickness = 0.00318  # 1/8" for small shells
        elif shell_diameter_m < 1.0:
            thickness = 0.00476  # 3/16" for medium shells
        else:
            thickness = 0.00635  # 1/4" for large shells

        return {
            "baffle_thickness_m": thickness,
            "source": "fallback approximation",
            "note": "Install ht library for TEMA-specific thickness",
        }

    try:
        from ht.hx import baffle_thickness

        t = baffle_thickness(shell_diameter_m, unsupported_length_m, service=service)
        return {
            "baffle_thickness_m": t,
            "shell_diameter_m": shell_diameter_m,
            "unsupported_length_m": unsupported_length_m,
            "service": service,
            "source": "ht.hx.baffle_thickness",
        }
    except Exception as e:
        return {"error": f"Baffle thickness calculation failed: {e}"}


def calculate_hx_shell_side_h_kern(
    shell_inner_diameter: float,
    tube_outer_diameter: float,
    tube_pitch: float,
    tube_layout_angle: int,
    baffle_spacing: float,
    baffle_cut_percent: float,
    shell_fluid_name: str,
    shell_fluid_flow_rate: float,
    shell_fluid_bulk_temp: float,
    shell_fluid_pressure: float = 101325.0,
    tube_wall_temp: Optional[float] = None,
    tube_rows: Optional[int] = None,
    pitch_parallel: Optional[float] = None,
    pitch_normal: Optional[float] = None,
    n_baffles: Optional[int] = None,
    include_pressure_drop: bool = True,
    strict: bool = False,
) -> str:
    """Estimates shell-side h and pressure drop using Kern's method.

    Args:
        shell_inner_diameter: Inner diameter of the shell (Ds) in meters
        tube_outer_diameter: Outer diameter of the tubes (do) in meters
        tube_pitch: Center-to-center distance between tubes (Pt) in meters
        tube_layout_angle: Tube layout angle (30, 45, 60, 90 degrees)
        baffle_spacing: Center-to-center spacing between baffles (B) in meters
        baffle_cut_percent: Baffle cut as a percentage of shell diameter (e.g., 25 for 25%)
        shell_fluid_name: Name of the fluid on the shell side
        shell_fluid_flow_rate: Mass flow rate of the shell-side fluid (m_shell) in kg/s
        shell_fluid_bulk_temp: Average bulk temperature of the shell-side fluid in Kelvin (K)
        shell_fluid_pressure: Pressure of shell-side fluid (Pa)
        tube_wall_temp: Estimated average temperature of the tube outer wall in Kelvin (K)
        tube_rows: Number of tube rows (for Zukauskas correlation)
        pitch_parallel: Parallel pitch (for Zukauskas correlation)
        pitch_normal: Normal pitch (for Zukauskas correlation)
        n_baffles: Number of baffles (required for pressure drop calculation)
        include_pressure_drop: If True and n_baffles provided, calculate dP using dP_Kern

    Returns:
        JSON string with shell-side heat transfer coefficient and optional pressure drop

    Note:
        The dP_Kern pressure drop calculation represents bundle crossflow only.
        It does NOT include window or nozzle losses, so actual shell-side
        pressure drop may be 20-40% higher.
    """
    try:
        # Parameter validation
        if tube_pitch <= tube_outer_diameter:
            return json.dumps({"error": "tube_pitch must be greater than tube_outer_diameter to allow flow clearance."})
        if shell_inner_diameter <= 0 or tube_outer_diameter <= 0 or tube_pitch <= 0 or baffle_spacing <= 0:
            return json.dumps(
                {
                    "error": "Dimensions must be positive values (shell_inner_diameter, tube_outer_diameter, tube_pitch, baffle_spacing)."
                }
            )
        if not (0 < baffle_cut_percent < 50):
            warning_msg = (
                f"Baffle cut ({baffle_cut_percent}%) is outside typical range (15-45%). Kern's method may be inaccurate."
            )
            if strict:
                raise ValueError(warning_msg)
            logger.warning(warning_msg)

        if baffle_spacing > shell_inner_diameter or baffle_spacing < shell_inner_diameter / 5.0:
            warning_msg = f"Baffle spacing ({baffle_spacing:.3f}m) is outside typical range (Ds/5 to Ds). Kern's method may be inaccurate."
            if strict:
                raise ValueError(warning_msg)
            logger.warning(warning_msg)

        # 1. Get Shell Fluid Properties at Bulk Temp
        bulk_props_json = get_fluid_properties(shell_fluid_name, shell_fluid_bulk_temp, shell_fluid_pressure, strict=strict)
        bulk_props = json.loads(bulk_props_json)

        if "error" in bulk_props:
            return json.dumps({"error": f"Failed to get shell fluid properties: {bulk_props['error']}"})

        rho_b = bulk_props.get("density")
        Cp_b = bulk_props.get("specific_heat_cp")
        k_b = bulk_props.get("thermal_conductivity")
        mu_b = bulk_props.get("dynamic_viscosity")
        Pr_b = bulk_props.get("prandtl_number")

        if any(p is None for p in [rho_b, Cp_b, k_b, mu_b, Pr_b]):
            return json.dumps(
                {"error": f"Could not get all required bulk properties for {shell_fluid_name} at {shell_fluid_bulk_temp}K."}
            )

        # 2. Calculate Equivalent Diameter (De)
        if tube_layout_angle in [30, 60]:  # Triangular pitch
            De = (4.0 * ((math.sqrt(3) / 4.0) * tube_pitch**2 - (math.pi / 8.0) * tube_outer_diameter**2)) / (
                (math.pi / 2.0) * tube_outer_diameter
            )
        elif tube_layout_angle in [45, 90]:  # Square pitch (rotated or aligned)
            De = (4.0 * (tube_pitch**2 - (math.pi / 4.0) * tube_outer_diameter**2)) / (math.pi * tube_outer_diameter)
        else:
            return json.dumps({"error": "Invalid tube_layout_angle. Use 30, 45, 60, or 90."})

        # 3. Calculate Shell-Side Cross-Flow Area (As)
        clearance = tube_pitch - tube_outer_diameter
        As = shell_inner_diameter * clearance * baffle_spacing / tube_pitch

        # 4. Calculate Shell-Side Mass Velocity (Gs)
        Gs = shell_fluid_flow_rate / As if As > 0 else 0

        # 5. Calculate Shell-Side Reynolds Number (Re_s)
        Re_s = De * Gs / mu_b if mu_b > 0 else 0

        # 6. Estimate Nusselt number: prefer tube-bank crossflow correlations when available
        Nu_s = None
        correlation_used = "unknown"

        if HT_AVAILABLE:
            try:
                # Use Zukauskas/ESDU/Grimison correlations for tube banks in crossflow
                from ht.conv_tube_bank import Nu_Zukauskas_Bejan

                # Estimate superficial velocity and Re on Do
                Vs = Gs / rho_b if rho_b else 0.0
                Re_tb = rho_b * Vs * tube_outer_diameter / mu_b if mu_b and rho_b else 0.0

                # Determine pitches; if not provided, assume square/triangular with equal pitch spacing
                p_par = pitch_parallel if pitch_parallel is not None else tube_pitch
                p_norm = pitch_normal if pitch_normal is not None else tube_pitch
                n_rows = tube_rows if tube_rows is not None else 10

                # Optional wall Pr correction
                Pr_w = None
                if tube_wall_temp is not None:
                    try:
                        wall_props_json = get_fluid_properties(
                            shell_fluid_name, tube_wall_temp, shell_fluid_pressure, strict=strict
                        )
                        wall_props = json.loads(wall_props_json)
                        mu_w = wall_props.get("dynamic_viscosity")
                        Cp_w = wall_props.get("specific_heat_cp")
                        k_w = wall_props.get("thermal_conductivity")
                        if mu_w and Cp_w and k_w:
                            Pr_w = mu_w * Cp_w / k_w
                    except Exception:
                        Pr_w = None

                if Pr_w is not None:
                    Nu_s = Nu_Zukauskas_Bejan(
                        Re_tb, Pr_b, tube_rows=n_rows, pitch_parallel=p_par, pitch_normal=p_norm, Pr_wall=Pr_w
                    )
                else:
                    Nu_s = Nu_Zukauskas_Bejan(Re_tb, Pr_b, tube_rows=n_rows, pitch_parallel=p_par, pitch_normal=p_norm)
                correlation_used = "Zukauskas-Bejan tube-bank crossflow"
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"ht tube-bank correlation failed: {e}")

        # Fallback to Kern method if tube-bank correlations are not available/applicable
        use_tube_od_for_h = False  # Track if we should use Do instead of De for h
        if Nu_s is None:
            # Warn about baffle cut applicability for Kern
            if baffle_cut_percent != 25.0:
                warning_msg = (
                    "Kern constants (C=0.36, n=0.55) assume ~25% baffle cut; provided "
                    f"{baffle_cut_percent}%. Results may be approximate."
                )
                logger.warning(warning_msg)
            C_kern, n_kern = 0.36, 0.55
            Nu_s = C_kern * (Re_s**n_kern) * (Pr_b ** (1.0 / 3.0)) if Re_s > 0 else 0
            correlation_used = f"Kern Approx Nu = {C_kern:.2f}*Re^{n_kern:.2f}*Pr^(1/3)"
            # Kern method: Nu is defined on De, so use De for h
            use_tube_od_for_h = False
        else:
            # Tube-bank correlations (Zukauskas-Bejan): Nu is defined on tube OD
            use_tube_od_for_h = True

        # 7. Calculate h_o without viscosity correction
        # Use the appropriate length scale based on the correlation:
        # - Kern method: Nu is based on De (shell equivalent diameter)
        # - Tube-bank correlations: Nu is based on Do (tube outer diameter)
        if use_tube_od_for_h:
            h_o_provisional = Nu_s * k_b / tube_outer_diameter if tube_outer_diameter > 0 else 0
        else:
            h_o_provisional = Nu_s * k_b / De if De > 0 else 0

        # 8. Apply Viscosity Correction (if wall temp is known)
        viscosity_correction = 1.0
        mu_w = None

        if tube_wall_temp is not None:
            try:
                wall_props_json = get_fluid_properties(shell_fluid_name, tube_wall_temp, shell_fluid_pressure, strict=strict)
                wall_props = json.loads(wall_props_json)

                if "error" not in wall_props:
                    mu_w = wall_props.get("dynamic_viscosity")

                    if mu_w is not None and mu_w > 0 and mu_b > 0:
                        viscosity_correction = (mu_b / mu_w) ** 0.14
                    else:
                        logger.warning("Could not get wall viscosity, correction factor set to 1.0.")
                else:
                    logger.warning(f"Error getting wall properties: {wall_props.get('error')}. Correction factor set to 1.0.")
            except Exception as e_wall_prop:
                logger.warning(
                    f"Could not get wall properties for viscosity correction: {e_wall_prop}. " f"Correction factor set to 1.0."
                )
        else:
            logger.info("Tube wall temperature not provided, viscosity correction factor (mu_b/mu_w)^0.14 not applied.")

        h_o_final = h_o_provisional * viscosity_correction

        # Calculate pressure drop using dP_Kern if requested and n_baffles provided
        pressure_drop_result = None
        if include_pressure_drop and n_baffles is not None and HT_AVAILABLE:
            try:
                # Import dP_Kern from ht.conv_tube_bank (NOT ht.hx!)
                from ht.conv_tube_bank import dP_Kern

                # Calculate pressure drop
                if mu_w is not None:
                    dP = dP_Kern(
                        m=shell_fluid_flow_rate,
                        rho=rho_b,
                        mu=mu_b,
                        DShell=shell_inner_diameter,
                        LSpacing=baffle_spacing,
                        pitch=tube_pitch,
                        Do=tube_outer_diameter,
                        NBaffles=n_baffles,
                        mu_w=mu_w,
                    )
                else:
                    dP = dP_Kern(
                        m=shell_fluid_flow_rate,
                        rho=rho_b,
                        mu=mu_b,
                        DShell=shell_inner_diameter,
                        LSpacing=baffle_spacing,
                        pitch=tube_pitch,
                        Do=tube_outer_diameter,
                        NBaffles=n_baffles,
                    )

                pressure_drop_result = {
                    "pressure_drop_Pa": dP,
                    "pressure_drop_kPa": dP / 1000,
                    "n_baffles": n_baffles,
                    "correlation": "dP_Kern (ht.conv_tube_bank)",
                    "note": "Bundle crossflow only; excludes window/nozzle losses. Actual dP may be 20-40% higher.",
                }
            except ImportError as ie:
                logger.warning(f"Could not import dP_Kern: {ie}")
                pressure_drop_result = {"error": f"dP_Kern import failed: {ie}"}
            except Exception as e:
                logger.warning(f"Pressure drop calculation failed: {e}")
                pressure_drop_result = {"error": str(e)}
        elif include_pressure_drop and n_baffles is None:
            pressure_drop_result = {"note": "Pressure drop not calculated: n_baffles not provided"}

        # Create result
        result = {
            "shell_side_h_W_m2K": h_o_final,
            "shell_reynolds_number_Re_s": Re_s,
            "equivalent_diameter_De_m": De,
            "cross_flow_area_As_m2": As,
            "mass_velocity_Gs_kg_m2s": Gs,
            "provisional_h_o_W_m2K": h_o_provisional,
            "viscosity_correction_factor": viscosity_correction,
            "bulk_viscosity_mu_b_Pa_s": mu_b,
            "wall_viscosity_mu_w_Pa_s": mu_w,
            "correlation_details": correlation_used,
            "nusselt_number_Nu_s": Nu_s,
            "geometry_details": {
                "shell_inner_diameter_m": shell_inner_diameter,
                "tube_outer_diameter_m": tube_outer_diameter,
                "tube_pitch_m": tube_pitch,
                "tube_layout_angle_deg": tube_layout_angle,
                "baffle_spacing_m": baffle_spacing,
                "baffle_cut_percent": baffle_cut_percent,
            },
        }

        # Add pressure drop if calculated
        if pressure_drop_result is not None:
            result["pressure_drop"] = pressure_drop_result

        return json.dumps(result)

    except Exception as e:
        logger.error(f"Unexpected error in calculate_hx_shell_side_h_kern: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
