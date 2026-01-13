"""
Pressure drop calculation tool for various heat exchanger geometries.

This module provides functionality to calculate pressure drops using
established correlations from the fluids and ht libraries.
"""

import json
import logging
import math
from typing import Optional

from utils.import_helpers import HT_AVAILABLE, FLUIDS_AVAILABLE

# Tools might need to call other tools
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.pressure_drop")


def calculate_pressure_drop(
    geometry: str,
    mass_flow_kg_s: float,
    fluid_name: str,
    fluid_temperature_K: float,
    # Common geometry parameters
    hydraulic_diameter_m: Optional[float] = None,
    flow_length_m: Optional[float] = None,
    flow_area_m2: Optional[float] = None,
    # PHE-specific parameters
    chevron_angle: Optional[float] = None,
    plate_enlargement_factor: Optional[float] = None,
    n_channels: Optional[int] = None,
    n_passes: int = 1,
    port_diameter_m: Optional[float] = None,
    # Shell-side (Kern) specific parameters
    shell_diameter_m: Optional[float] = None,
    baffle_spacing_m: Optional[float] = None,
    tube_pitch_m: Optional[float] = None,
    tube_od_m: Optional[float] = None,
    n_baffles: Optional[int] = None,
    # Pipe/tube internal flow parameters
    pipe_roughness_m: float = 0.0,
    # Options
    include_port_losses: bool = True,
    correlation: str = "auto",
    fluid_pressure_Pa: float = 101325.0,
    strict: bool = False,
) -> str:
    """Calculate pressure drop for various heat exchanger geometries.

    Supports plate heat exchangers (PHE), shell-side (Kern method), and internal pipe flow.

    Args:
        geometry: Geometry type. Options:
            - 'plate_chevron': Chevron-style plate heat exchanger channel
            - 'pipe_internal': Internal pipe/tube flow
            - 'shell_side_kern': Shell-side using Kern method
        mass_flow_kg_s: Mass flow rate (kg/s)
        fluid_name: Name of the fluid
        fluid_temperature_K: Bulk fluid temperature (K)

        hydraulic_diameter_m: Hydraulic diameter (m). Required for plate_chevron and pipe_internal.
        flow_length_m: Flow path length per pass (m). Required for plate_chevron and pipe_internal.
        flow_area_m2: Cross-sectional flow area per channel (m²). Required for plate_chevron.

        chevron_angle: Chevron angle in degrees (required for plate_chevron, typically 30-65)
        plate_enlargement_factor: Area enhancement factor from corrugations (required for Muley_Manglik)
        n_channels: Number of channels on this fluid side (required for plate_chevron)
        n_passes: Number of passes (default 1)
        port_diameter_m: Port diameter for PHE manifold loss calculation (m)

        shell_diameter_m: Shell inner diameter (m). Required for shell_side_kern.
        baffle_spacing_m: Baffle spacing (m). Required for shell_side_kern.
        tube_pitch_m: Tube center-to-center pitch (m). Required for shell_side_kern.
        tube_od_m: Tube outer diameter (m). Required for shell_side_kern.
        n_baffles: Number of baffles. Required for shell_side_kern.

        pipe_roughness_m: Pipe internal surface roughness (m). For pipe_internal.

        include_port_losses: Whether to include port/manifold losses for PHE (default True)
        correlation: Correlation to use. Options depend on geometry:
            - plate_chevron: 'auto', 'Martin_1999', 'Martin_VDI', 'Kumar', 'Muley_Manglik'
            - pipe_internal: 'auto' (uses fluids.one_phase_dP)
            - shell_side_kern: 'Kern' (only option)
        fluid_pressure_Pa: Fluid pressure for property lookup (Pa, default 101325)
        strict: If True, require ht/fluids libraries

    Returns:
        JSON string with pressure drop results including:
        - pressure_drop_Pa: Total pressure drop (Pa)
        - pressure_drop_kPa: Total pressure drop (kPa)
        - friction_factor: Darcy friction factor
        - reynolds_number: Reynolds number
        - velocity_m_s: Flow velocity (m/s)
        - correlation_used: Name of correlation used
        - components: Breakdown of frictional vs port losses (for PHE)
    """
    try:
        # Input validation
        if not isinstance(geometry, str) or not geometry.strip():
            return json.dumps({"error": "geometry must be a non-empty string."})

        geometry_lower = geometry.lower()
        valid_geometries = ['plate_chevron', 'chevron', 'pipe_internal', 'pipe', 'shell_side_kern', 'shell_kern', 'kern']
        if not any(g in geometry_lower for g in valid_geometries):
            return json.dumps({
                "error": f"Unsupported geometry: {geometry}. Valid options: plate_chevron, pipe_internal, shell_side_kern"
            })

        if mass_flow_kg_s is None or mass_flow_kg_s <= 0:
            return json.dumps({"error": "mass_flow_kg_s must be positive."})

        if fluid_temperature_K is None or fluid_temperature_K <= 0:
            return json.dumps({"error": "fluid_temperature_K must be positive."})

        # Get fluid properties
        fluid_props_json = get_fluid_properties(
            fluid_name, fluid_temperature_K, fluid_pressure_Pa, strict=strict
        )
        fluid_props = json.loads(fluid_props_json)

        if "error" in fluid_props:
            return json.dumps({
                "error": f"Failed to get fluid properties: {fluid_props['error']}"
            })

        rho = fluid_props.get("density")
        mu = fluid_props.get("dynamic_viscosity")
        if rho is None or mu is None:
            return json.dumps({"error": "Missing density or viscosity from fluid properties."})

        # Route to appropriate calculation
        if 'plate_chevron' in geometry_lower or 'chevron' in geometry_lower:
            return _calculate_phe_pressure_drop(
                mass_flow_kg_s=mass_flow_kg_s,
                rho=rho,
                mu=mu,
                hydraulic_diameter_m=hydraulic_diameter_m,
                flow_length_m=flow_length_m,
                flow_area_m2=flow_area_m2,
                chevron_angle=chevron_angle,
                plate_enlargement_factor=plate_enlargement_factor,
                n_channels=n_channels,
                n_passes=n_passes,
                port_diameter_m=port_diameter_m,
                include_port_losses=include_port_losses,
                correlation=correlation,
                fluid_name=fluid_name,
                fluid_temperature_K=fluid_temperature_K,
                strict=strict,
            )

        elif 'pipe_internal' in geometry_lower or geometry_lower == 'pipe':
            return _calculate_pipe_pressure_drop(
                mass_flow_kg_s=mass_flow_kg_s,
                rho=rho,
                mu=mu,
                hydraulic_diameter_m=hydraulic_diameter_m,
                flow_length_m=flow_length_m,
                pipe_roughness_m=pipe_roughness_m,
                fluid_name=fluid_name,
                fluid_temperature_K=fluid_temperature_K,
                strict=strict,
            )

        elif 'shell_side_kern' in geometry_lower or 'shell_kern' in geometry_lower or 'kern' in geometry_lower:
            return _calculate_shell_side_kern_pressure_drop(
                mass_flow_kg_s=mass_flow_kg_s,
                rho=rho,
                mu=mu,
                shell_diameter_m=shell_diameter_m,
                baffle_spacing_m=baffle_spacing_m,
                tube_pitch_m=tube_pitch_m,
                tube_od_m=tube_od_m,
                n_baffles=n_baffles,
                fluid_name=fluid_name,
                fluid_temperature_K=fluid_temperature_K,
                strict=strict,
            )

        else:
            return json.dumps({"error": f"Unhandled geometry: {geometry}"})

    except Exception as e:
        logger.error(f"Error in calculate_pressure_drop: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


def _calculate_phe_pressure_drop(
    mass_flow_kg_s: float,
    rho: float,
    mu: float,
    hydraulic_diameter_m: Optional[float],
    flow_length_m: Optional[float],
    flow_area_m2: Optional[float],
    chevron_angle: Optional[float],
    plate_enlargement_factor: Optional[float],
    n_channels: Optional[int],
    n_passes: int,
    port_diameter_m: Optional[float],
    include_port_losses: bool,
    correlation: str,
    fluid_name: str,
    fluid_temperature_K: float,
    strict: bool,
) -> str:
    """Calculate pressure drop for chevron-style plate heat exchanger."""

    # Validate required PHE parameters
    if hydraulic_diameter_m is None or hydraulic_diameter_m <= 0:
        return json.dumps({"error": "hydraulic_diameter_m is required and must be positive for plate_chevron."})
    if flow_length_m is None or flow_length_m <= 0:
        return json.dumps({"error": "flow_length_m is required and must be positive for plate_chevron."})
    if flow_area_m2 is None or flow_area_m2 <= 0:
        return json.dumps({"error": "flow_area_m2 (per channel) is required and must be positive for plate_chevron."})
    if chevron_angle is None:
        return json.dumps({"error": "chevron_angle is required for plate_chevron geometry."})
    if chevron_angle < 0 or chevron_angle > 90:
        return json.dumps({"error": "chevron_angle must be between 0 and 90 degrees."})
    if n_channels is None or n_channels <= 0:
        return json.dumps({"error": "n_channels is required and must be positive for plate_chevron."})

    # Validate correlation
    valid_correlations = ['auto', 'Martin_1999', 'Martin_VDI', 'Kumar', 'Muley_Manglik']
    if correlation not in valid_correlations:
        return json.dumps({
            "error": f"Invalid correlation: {correlation}. Valid options: {valid_correlations}"
        })

    if correlation == 'Muley_Manglik' and plate_enlargement_factor is None:
        return json.dumps({"error": "plate_enlargement_factor is required for Muley_Manglik correlation."})

    # Default to Martin_VDI if auto
    if correlation == 'auto':
        correlation = 'Martin_VDI'

    # Check for fluids library
    if not FLUIDS_AVAILABLE:
        if strict:
            return json.dumps({"error": "fluids library required with strict=True"})
        return json.dumps({
            "error": "fluids library not available for PHE pressure drop calculations",
            "suggestion": "Install with: pip install fluids>=1.0.0"
        })

    try:
        # Import friction factor correlations
        from fluids.friction import (
            friction_plate_Martin_1999,
            friction_plate_Martin_VDI,
            friction_plate_Kumar,
            friction_plate_Muley_Manglik,
        )

        # Calculate per-channel mass flow
        m_per_channel = mass_flow_kg_s / n_channels

        # Calculate velocity per channel
        velocity = m_per_channel / (rho * flow_area_m2)

        # Calculate Reynolds number
        Re = rho * velocity * hydraulic_diameter_m / mu

        # Calculate friction factor using selected correlation
        correlation_notes = []
        if correlation == 'Martin_1999':
            if chevron_angle < 0 or chevron_angle > 80:
                correlation_notes.append(f"Warning: Martin_1999 valid for chevron 0-80°, got {chevron_angle}°")
            if Re < 200 or Re > 10000:
                correlation_notes.append(f"Warning: Martin_1999 valid for Re 200-10000, got Re={Re:.1f}")
            f_darcy = friction_plate_Martin_1999(Re, chevron_angle)

        elif correlation == 'Martin_VDI':
            if chevron_angle < 0 or chevron_angle > 80:
                correlation_notes.append(f"Warning: Martin_VDI valid for chevron 0-80°, got {chevron_angle}°")
            if Re < 200 or Re > 10000:
                correlation_notes.append(f"Warning: Martin_VDI valid for Re 200-10000, got Re={Re:.1f}")
            f_darcy = friction_plate_Martin_VDI(Re, chevron_angle)

        elif correlation == 'Kumar':
            if chevron_angle < 30 or chevron_angle > 65:
                correlation_notes.append(f"Warning: Kumar valid for chevron 30-65°, got {chevron_angle}°")
            if Re < 0.1 or Re > 10000:
                correlation_notes.append(f"Warning: Kumar valid for Re 0.1-10000, got Re={Re:.1f}")
            f_darcy = friction_plate_Kumar(Re, chevron_angle)

        elif correlation == 'Muley_Manglik':
            if chevron_angle < 30 or chevron_angle > 60:
                correlation_notes.append(f"Warning: Muley_Manglik valid for chevron 30-60°, got {chevron_angle}°")
            if Re < 1000:
                correlation_notes.append(f"Warning: Muley_Manglik valid for Re > 1000, got Re={Re:.1f}")
            f_darcy = friction_plate_Muley_Manglik(Re, chevron_angle, plate_enlargement_factor)

        # Calculate frictional pressure drop per pass
        # dP_channel = f_d * (L / D_h) * (rho * v^2 / 2)
        dP_per_channel = f_darcy * (flow_length_m / hydraulic_diameter_m) * (rho * velocity**2 / 2)

        # Total frictional pressure drop = per_channel * n_passes
        dP_friction = dP_per_channel * n_passes

        # Calculate port losses if requested and port diameter is provided
        dP_port = 0.0
        if include_port_losses and port_diameter_m is not None and port_diameter_m > 0:
            # Port area
            A_port = math.pi * port_diameter_m**2 / 4
            # Port velocity (total mass flow through port)
            v_port = mass_flow_kg_s / (rho * A_port)
            # Port losses: approximately 1.4 velocity heads per port pair (inlet + outlet)
            # Total = 2 ports * 0.7 = 1.4 velocity heads
            dP_port = 1.4 * rho * v_port**2

        # Total pressure drop
        dP_total = dP_friction + dP_port

        result = {
            "pressure_drop_Pa": dP_total,
            "pressure_drop_kPa": dP_total / 1000,
            "friction_factor": f_darcy,
            "reynolds_number": Re,
            "velocity_m_s": velocity,
            "correlation_used": correlation,
            "components": {
                "frictional_Pa": dP_friction,
                "frictional_kPa": dP_friction / 1000,
                "port_losses_Pa": dP_port,
                "port_losses_kPa": dP_port / 1000,
            },
            "geometry_parameters": {
                "hydraulic_diameter_m": hydraulic_diameter_m,
                "flow_length_m": flow_length_m,
                "flow_area_m2": flow_area_m2,
                "chevron_angle_deg": chevron_angle,
                "n_channels": n_channels,
                "n_passes": n_passes,
            },
            "fluid": {
                "name": fluid_name,
                "temperature_K": fluid_temperature_K,
                "density_kg_m3": rho,
                "viscosity_Pa_s": mu,
            },
        }

        if plate_enlargement_factor is not None:
            result["geometry_parameters"]["plate_enlargement_factor"] = plate_enlargement_factor

        if port_diameter_m is not None:
            result["geometry_parameters"]["port_diameter_m"] = port_diameter_m
            if include_port_losses:
                result["components"]["port_velocity_m_s"] = v_port

        if correlation_notes:
            result["warnings"] = correlation_notes

        return json.dumps(result)

    except ImportError as e:
        return json.dumps({
            "error": f"Failed to import fluids friction correlations: {e}",
            "suggestion": "Install with: pip install fluids>=1.0.0"
        })
    except Exception as e:
        logger.error(f"Error in PHE pressure drop calculation: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


def _calculate_pipe_pressure_drop(
    mass_flow_kg_s: float,
    rho: float,
    mu: float,
    hydraulic_diameter_m: Optional[float],
    flow_length_m: Optional[float],
    pipe_roughness_m: float,
    fluid_name: str,
    fluid_temperature_K: float,
    strict: bool,
) -> str:
    """Calculate pressure drop for internal pipe flow."""

    # Validate required parameters
    if hydraulic_diameter_m is None or hydraulic_diameter_m <= 0:
        return json.dumps({"error": "hydraulic_diameter_m is required and must be positive for pipe_internal."})
    if flow_length_m is None or flow_length_m <= 0:
        return json.dumps({"error": "flow_length_m is required and must be positive for pipe_internal."})

    if not FLUIDS_AVAILABLE:
        if strict:
            return json.dumps({"error": "fluids library required with strict=True"})
        return json.dumps({
            "error": "fluids library not available for pipe pressure drop calculations",
            "suggestion": "Install with: pip install fluids>=1.0.0"
        })

    try:
        from fluids.friction import one_phase_dP

        # Calculate pressure drop using fluids wrapper
        dP = one_phase_dP(
            m=mass_flow_kg_s,
            rho=rho,
            mu=mu,
            D=hydraulic_diameter_m,
            roughness=pipe_roughness_m,
            L=flow_length_m
        )

        # Calculate additional info
        A_pipe = math.pi * hydraulic_diameter_m**2 / 4
        velocity = mass_flow_kg_s / (rho * A_pipe)
        Re = rho * velocity * hydraulic_diameter_m / mu

        # Calculate friction factor from dP (back-calculation for reporting)
        # dP = f * (L/D) * (rho * v^2 / 2)
        # f = dP * 2 * D / (L * rho * v^2)
        if velocity > 0 and flow_length_m > 0:
            f_darcy = dP * 2 * hydraulic_diameter_m / (flow_length_m * rho * velocity**2)
        else:
            f_darcy = None

        result = {
            "pressure_drop_Pa": dP,
            "pressure_drop_kPa": dP / 1000,
            "friction_factor": f_darcy,
            "reynolds_number": Re,
            "velocity_m_s": velocity,
            "correlation_used": "fluids.one_phase_dP",
            "geometry_parameters": {
                "hydraulic_diameter_m": hydraulic_diameter_m,
                "flow_length_m": flow_length_m,
                "roughness_m": pipe_roughness_m,
            },
            "fluid": {
                "name": fluid_name,
                "temperature_K": fluid_temperature_K,
                "density_kg_m3": rho,
                "viscosity_Pa_s": mu,
            },
        }

        return json.dumps(result)

    except ImportError as e:
        return json.dumps({
            "error": f"Failed to import fluids: {e}",
            "suggestion": "Install with: pip install fluids>=1.0.0"
        })
    except Exception as e:
        logger.error(f"Error in pipe pressure drop calculation: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


def _calculate_shell_side_kern_pressure_drop(
    mass_flow_kg_s: float,
    rho: float,
    mu: float,
    shell_diameter_m: Optional[float],
    baffle_spacing_m: Optional[float],
    tube_pitch_m: Optional[float],
    tube_od_m: Optional[float],
    n_baffles: Optional[int],
    fluid_name: str,
    fluid_temperature_K: float,
    strict: bool,
) -> str:
    """Calculate shell-side pressure drop using Kern's method."""

    # Validate required parameters
    if shell_diameter_m is None or shell_diameter_m <= 0:
        return json.dumps({"error": "shell_diameter_m is required and must be positive for shell_side_kern."})
    if baffle_spacing_m is None or baffle_spacing_m <= 0:
        return json.dumps({"error": "baffle_spacing_m is required and must be positive for shell_side_kern."})
    if tube_pitch_m is None or tube_pitch_m <= 0:
        return json.dumps({"error": "tube_pitch_m is required and must be positive for shell_side_kern."})
    if tube_od_m is None or tube_od_m <= 0:
        return json.dumps({"error": "tube_od_m is required and must be positive for shell_side_kern."})
    if n_baffles is None or n_baffles < 0:
        return json.dumps({"error": "n_baffles is required and must be non-negative for shell_side_kern."})

    if not HT_AVAILABLE:
        if strict:
            return json.dumps({"error": "ht library required with strict=True"})
        return json.dumps({
            "error": "ht library not available for shell-side pressure drop calculations",
            "suggestion": "Install with: pip install ht>=1.2.0"
        })

    try:
        # Import dP_Kern from ht.conv_tube_bank (NOT ht.hx!)
        from ht.conv_tube_bank import dP_Kern

        # Calculate pressure drop using Kern method
        dP = dP_Kern(
            m=mass_flow_kg_s,
            rho=rho,
            mu=mu,
            DShell=shell_diameter_m,
            LSpacing=baffle_spacing_m,
            pitch=tube_pitch_m,
            Do=tube_od_m,
            NBaffles=n_baffles
        )

        # Calculate derived quantities for reporting
        # Shell-side flow area: Ss = DShell * (pitch - Do) * LSpacing / pitch
        Ss = shell_diameter_m * (tube_pitch_m - tube_od_m) * baffle_spacing_m / tube_pitch_m
        # Equivalent diameter: De = 4 * (pitch^2 - pi*Do^2/4) / (pi*Do)
        De = 4 * (tube_pitch_m**2 - math.pi * tube_od_m**2 / 4) / (math.pi * tube_od_m)
        # Shell-side velocity
        Vs = mass_flow_kg_s / (Ss * rho)
        # Reynolds number
        Re = rho * De * Vs / mu

        result = {
            "pressure_drop_Pa": dP,
            "pressure_drop_kPa": dP / 1000,
            "reynolds_number": Re,
            "velocity_m_s": Vs,
            "correlation_used": "dP_Kern (ht.conv_tube_bank)",
            "geometry_parameters": {
                "shell_diameter_m": shell_diameter_m,
                "baffle_spacing_m": baffle_spacing_m,
                "tube_pitch_m": tube_pitch_m,
                "tube_od_m": tube_od_m,
                "n_baffles": n_baffles,
            },
            "derived_parameters": {
                "shell_flow_area_m2": Ss,
                "equivalent_diameter_m": De,
            },
            "fluid": {
                "name": fluid_name,
                "temperature_K": fluid_temperature_K,
                "density_kg_m3": rho,
                "viscosity_Pa_s": mu,
            },
            "notes": [
                "Kern method calculates bundle crossflow only",
                "Does not include window or nozzle losses",
                "Real shell-side pressure drop may be 20-40% higher"
            ],
        }

        return json.dumps(result)

    except ImportError as e:
        return json.dumps({
            "error": f"Failed to import ht.conv_tube_bank.dP_Kern: {e}",
            "suggestion": "Install with: pip install ht>=1.2.0"
        })
    except Exception as e:
        logger.error(f"Error in shell-side pressure drop calculation: {e}", exc_info=True)
        return json.dumps({"error": str(e)})
