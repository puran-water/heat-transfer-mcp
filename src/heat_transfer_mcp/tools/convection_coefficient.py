"""
Convection coefficient tool to calculate heat transfer coefficients for various geometries.

This module provides functionality to calculate convective heat transfer coefficients
using established correlations from the HT library or fallback implementations.
"""

import json
import logging
import math
from typing import Optional

from heat_transfer_mcp.convection.single_phase import require_prandtl_number
from heat_transfer_mcp.convection.two_phase import calculate_two_phase_coefficient
from heat_transfer_mcp.errors import CalculationError
from heat_transfer_mcp.utils.helpers import calculate_nusselt_number_external_flow, calculate_reynolds_number
from heat_transfer_mcp.utils.import_helpers import HT_AVAILABLE

# Tools might need to call other tools
from heat_transfer_mcp.tools.fluid_properties import PropertyLookupRefused, calculation_properties, calculation_property_source, property_refusal

logger = logging.getLogger("heat-transfer-mcp.convection_coefficient")


def _get_thermal_expansion_coefficient(
    fluid_name: str,
    temperature_K: float,
    pressure_Pa: float,
    property_source: str,
) -> tuple:
    """Return the isobaric expansion coefficient from the calculation's property branch (CALCULATION_PROPERTY_RULE)."""
    properties = calculation_properties(fluid_name, temperature_K, pressure_Pa, property_source)
    beta = properties.get("isobaric_expansion_per_K")
    if beta is None or not math.isfinite(beta) or beta <= 0:
        raise ValueError("A positive, qualified isobaric expansion coefficient is required; density-anomaly convection is unsupported")
    return float(beta), False, None


def calculate_two_phase_h(
    quality: float,
    tube_diameter_m: float,
    fluid_name: str,
    saturation_temperature_K: float,
    mass_flux_kg_m2s: Optional[float] = None,
    mass_flow_kg_s: Optional[float] = None,
    method: str = "auto",
    strict: bool = False,
    length_m: Optional[float] = None,
    void_fraction: Optional[float] = None,
    excess_temperature_K: Optional[float] = None,
    property_source: str = "CoolProp",
) -> str:
    """Calculate two-phase heat-transfer coefficient with correct mass-flow semantics; the saturation properties come
    from ``property_source`` or the call refuses (CALCULATION_PROPERTY_RULE)."""
    try:
        result = calculate_two_phase_coefficient(
            quality=quality,
            tube_diameter_m=tube_diameter_m,
            fluid_name=fluid_name,
            saturation_temperature_K=saturation_temperature_K,
            mass_flow_kg_s=mass_flow_kg_s,
            mass_flux_kg_m2s=mass_flux_kg_m2s,
            method=method,
            strict=strict,
            length_m=length_m,
            void_fraction=void_fraction,
            excess_temperature_K=excess_temperature_K,
            property_source=calculation_property_source(property_source),
        )
        return json.dumps(result)

    except CalculationError as e:
        return json.dumps(e.to_dict())
    except Exception as e:
        logger.error(f"Error in calculate_two_phase_h: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


def calculate_convection_coefficient(
    geometry: str,
    characteristic_dimension: float,
    fluid_name: str,
    bulk_fluid_temperature: float,
    surface_temperature: float,
    pressure: float = 101325.0,
    flow_type: str = "forced",
    fluid_velocity: Optional[float] = None,
    roughness: Optional[float] = 0.0,
    pipe_length: Optional[float] = None,
    # PHE-specific parameters (for geometry='plate_chevron')
    chevron_angle: Optional[float] = None,
    plate_enlargement_factor: Optional[float] = None,
    phe_correlation: str = "Martin_VDI",
    strict: bool = False,
    property_source: str = "CoolProp",
) -> str:
    """Calculates convective heat transfer coefficient for various geometries and flow conditions.

    Args:
        geometry: Geometry type. Supported values:
            - 'flat_plate_external': External flow over flat plate
            - 'pipe_internal': Internal pipe flow
            - 'pipe_external' or 'cylinder': External flow over cylinder
            - 'sphere': Flow over sphere
            - 'vertical_wall': Vertical surface (natural convection)
            - 'plate_chevron': Chevron-style plate heat exchanger channel
        characteristic_dimension: Characteristic length/diameter in meters.
            For plate_chevron: hydraulic diameter D_h (m)
        fluid_name: Name of the fluid
        bulk_fluid_temperature: Bulk temperature of the fluid in Kelvin
        surface_temperature: Temperature of the surface in Kelvin
        pressure: Pressure in Pascals
        flow_type: Flow regime ('natural' or 'forced')
        fluid_velocity: Fluid velocity in m/s (required for 'forced' flow)
        roughness: Surface roughness in meters (relevant for internal pipe flow)
        pipe_length: Flow length in meters (for entry effects in pipe flow)
        chevron_angle: Chevron angle in degrees (required for plate_chevron, typically 30-65)
        plate_enlargement_factor: Area enhancement factor from corrugations (1.1-1.5 typical).
            Required for Muley_Manglik correlation. If not provided, geometry class can compute it.
        phe_correlation: PHE Nusselt correlation to use. Options:
            - 'Kumar': APV data-based, supports viscosity correction (Re 0.1-10000, chevron 30-65°)
            - 'Martin_1999': Martin 1999 theoretical (Re 200-10000, chevron 0-80°)
            - 'Martin_VDI': Martin VDI revision (default, Re 200-10000, chevron 0-80°)
            - 'Muley_Manglik': Includes enlargement factor (Re > 1000, chevron 30-60°)
            - 'Khan_Khan': For low-Re liquids
        strict: If True, require ht library and fail if correlations are not available
        property_source: The branch every property lookup reads, 'CoolProp' or 'thermo' (CALCULATION_PROPERTY_RULE)

    Returns:
        JSON string with the calculated convection coefficient and related parameters.
        For plate_chevron, also includes the paired friction correlation name.
    """
    # Validate inputs to avoid misleading errors and non-physical results
    try:
        if not isinstance(geometry, str) or not geometry.strip():
            return json.dumps({"error": "Geometry must be a non-empty string."})
        geometry_lower = geometry.lower()
        recognized_keywords = [
            "flat_plate",
            "plate",
            "pipe",
            "cylinder",
            "sphere",
            "vertical",
            "horizontal",
            "wall",
            "plate_chevron",
            "chevron",
        ]
        if not any(k in geometry_lower for k in recognized_keywords):
            return json.dumps({"error": f"Unsupported geometry: {geometry}."})

        # Validate PHE-specific parameters for plate_chevron geometry
        if "plate_chevron" in geometry_lower or "chevron" in geometry_lower:
            if chevron_angle is None:
                return json.dumps({"error": "chevron_angle is required for plate_chevron geometry."})
            if chevron_angle < 0 or chevron_angle > 90:
                return json.dumps({"error": "chevron_angle must be between 0 and 90 degrees."})
            valid_phe_correlations = ["Kumar", "Martin_1999", "Martin_VDI", "Muley_Manglik", "Khan_Khan"]
            if phe_correlation not in valid_phe_correlations:
                return json.dumps(
                    {"error": f"Invalid phe_correlation: {phe_correlation}. Valid options: {valid_phe_correlations}"}
                )
            if phe_correlation == "Muley_Manglik" and plate_enlargement_factor is None:
                return json.dumps({"error": "plate_enlargement_factor is required for Muley_Manglik correlation."})
        if flow_type is None or flow_type.lower() not in {"forced", "natural"}:
            return json.dumps({"error": "flow_type must be 'forced' or 'natural'."})
        # Characteristic dimension
        try:
            characteristic_dimension = float(characteristic_dimension)
        except (TypeError, ValueError):
            return json.dumps({"error": "Characteristic dimension must be a numeric value in meters."})
        if not math.isfinite(characteristic_dimension) or characteristic_dimension <= 0.0:
            return json.dumps({"error": "Characteristic dimension must be a positive, finite value."})
        # Forced convection requires a valid velocity
        if flow_type.lower() == "forced":
            if fluid_velocity is None:
                return json.dumps({"error": "Fluid velocity is required for forced convection calculations."})
            try:
                fluid_velocity = float(fluid_velocity)
            except (TypeError, ValueError):
                return json.dumps({"error": "Fluid velocity must be a numeric value in m/s."})
            if not math.isfinite(fluid_velocity) or fluid_velocity <= 0.0:
                return json.dumps({"error": "Fluid velocity must be a positive, finite value for forced convection."})
        # Temperatures and pressure
        try:
            bulk_fluid_temperature = float(bulk_fluid_temperature)
            surface_temperature = float(surface_temperature)
            pressure = float(pressure)
        except (TypeError, ValueError):
            return json.dumps({"error": "Temperatures (K) and pressure (Pa) must be numeric values."})
        for T_val, label in [(bulk_fluid_temperature, "bulk_fluid_temperature"), (surface_temperature, "surface_temperature")]:
            if not math.isfinite(T_val):
                return json.dumps({"error": f"{label} must be a finite real number."})
            if T_val <= 0.0:
                return json.dumps({"error": f"{label} cannot be below 0 K (absolute zero)."})
            if T_val > 1.0e4:
                return json.dumps({"error": f"{label} is unrealistically high. Please provide < 10000 K."})
        if not math.isfinite(pressure) or pressure <= 0.0:
            return json.dumps({"error": "Pressure must be a positive, finite value."})
        if roughness is not None:
            try:
                roughness = float(roughness)
            except (TypeError, ValueError):
                return json.dumps({"error": "Roughness must be a numeric value in meters."})
            if not math.isfinite(roughness) or roughness < 0.0:
                return json.dumps({"error": "Roughness must be a non-negative, finite value."})

        # Calculate film temperature (average of bulk and surface)
        film_temperature = (bulk_fluid_temperature + surface_temperature) / 2.0
        if "chevron" in geometry_lower or "pipe_internal" in geometry_lower:
            film_temperature = bulk_fluid_temperature

        # Get fluid properties at film temperature
        property_source = calculation_property_source(property_source)
        fluid_props = calculation_properties(fluid_name, film_temperature, pressure, property_source)

        # Extract required properties
        density = fluid_props.get("density")
        dynamic_viscosity = fluid_props.get("dynamic_viscosity")
        thermal_conductivity = fluid_props.get("thermal_conductivity")
        specific_heat_cp = fluid_props.get("specific_heat_cp")
        prandtl_number = fluid_props.get("prandtl_number")

        # Check for required properties
        if None in [density, dynamic_viscosity, thermal_conductivity]:
            return json.dumps({"error": "Missing critical fluid properties for convection calculation."})

        try:
            prandtl_number = require_prandtl_number(
                prandtl_number,
                specific_heat_cp,
                dynamic_viscosity,
                thermal_conductivity,
            )
        except CalculationError as e:
            return json.dumps(e.to_dict())

        # Calculate Reynolds number for forced flow
        reynolds_number = None
        if flow_type.lower() == "forced" and fluid_velocity is not None:
            reynolds_number = calculate_reynolds_number(fluid_velocity, characteristic_dimension, density, dynamic_viscosity)

        # Calculate Nusselt number and convection coefficient based on geometry and flow regime
        result = {}
        geometry_lower = geometry.lower()
        nusselt_number = None
        convection_coefficient = None

        if strict and not HT_AVAILABLE:
            raise ImportError("ht library required with strict=True")

        if HT_AVAILABLE:

            try:
                # Handle plate_chevron geometry first (PHE convection coefficient)
                if "plate_chevron" in geometry_lower or ("chevron" in geometry_lower and "plate" not in geometry_lower):
                    # Import PHE-specific correlations from ht.conv_plate
                    from ht.conv_plate import (
                        Nu_plate_Khan_Khan,
                        Nu_plate_Kumar,
                        Nu_plate_Martin,
                        Nu_plate_Muley_Manglik,
                    )

                    # PHE is always forced convection
                    if reynolds_number is None or reynolds_number <= 0:
                        return json.dumps(
                            {
                                "error": "Reynolds number must be positive for plate_chevron geometry. Ensure fluid_velocity is provided."
                            }
                        )

                    # Get wall viscosity for Kumar correlation (viscosity correction)
                    mu_wall = None
                    if phe_correlation == "Kumar":
                        mu_wall = calculation_properties(fluid_name, surface_temperature, pressure,
                                                         property_source)["dynamic_viscosity"]

                    # Calculate Nusselt number using appropriate correlation
                    # Note: chevron_angle for ht functions is in degrees
                    paired_friction_correlation = None
                    correlation_notes = []
                    from heat_transfer_mcp.convection.plate import require_plate_domain
                    require_plate_domain(phe_correlation, reynolds_number, chevron_angle,
                                         plate_enlargement_factor, prandtl_number)

                    if phe_correlation == "Kumar":
                        # Kumar correlation (APV data-based)
                        # Valid range: Re 0.1-10000, chevron 30-65°
                        if chevron_angle < 30 or chevron_angle > 65:
                            correlation_notes.append(
                                f"Warning: Kumar correlation valid for chevron 30-65°, got {chevron_angle}°"
                            )
                        if reynolds_number < 0.1 or reynolds_number > 10000:
                            correlation_notes.append(
                                f"Warning: Kumar correlation valid for Re 0.1-10000, got Re={reynolds_number:.1f}"
                            )

                        if mu_wall is not None and dynamic_viscosity is not None:
                            nusselt_number = Nu_plate_Kumar(
                                Re=reynolds_number,
                                Pr=prandtl_number,
                                chevron_angle=chevron_angle,
                                mu=dynamic_viscosity,
                                mu_wall=mu_wall,
                            )
                        else:
                            nusselt_number = Nu_plate_Kumar(Re=reynolds_number, Pr=prandtl_number, chevron_angle=chevron_angle)
                        paired_friction_correlation = "friction_plate_Kumar"

                    elif phe_correlation in ["Martin_1999", "Martin_VDI"]:
                        # Martin correlation (theoretical)
                        # Valid range: Re 200-10000, chevron 0-80°
                        if chevron_angle < 0 or chevron_angle > 80:
                            correlation_notes.append(
                                f"Warning: Martin correlation valid for chevron 0-80°, got {chevron_angle}°"
                            )
                        if reynolds_number < 200 or reynolds_number > 10000:
                            correlation_notes.append(
                                f"Warning: Martin correlation valid for Re 200-10000, got Re={reynolds_number:.1f}"
                            )

                        variant = "1999" if phe_correlation == "Martin_1999" else "VDI"
                        nusselt_number = Nu_plate_Martin(
                            Re=reynolds_number, Pr=prandtl_number, chevron_angle=chevron_angle, variant=variant
                        )
                        paired_friction_correlation = f"friction_plate_Martin_{variant}"

                    elif phe_correlation == "Muley_Manglik":
                        # Muley-Manglik correlation (includes enlargement factor)
                        # Valid range: Re > 1000, chevron 30-60°
                        if chevron_angle < 30 or chevron_angle > 60:
                            correlation_notes.append(f"Warning: Muley_Manglik valid for chevron 30-60°, got {chevron_angle}°")
                        if reynolds_number < 1000:
                            correlation_notes.append(
                                f"Warning: Muley_Manglik valid for Re > 1000, got Re={reynolds_number:.1f}"
                            )

                        nusselt_number = Nu_plate_Muley_Manglik(
                            Re=reynolds_number,
                            Pr=prandtl_number,
                            chevron_angle=chevron_angle,
                            plate_enlargement_factor=plate_enlargement_factor,
                        )
                        paired_friction_correlation = "friction_plate_Muley_Manglik"

                    elif phe_correlation == "Khan_Khan":
                        # Khan & Khan correlation (low-Re liquids)
                        nusselt_number = Nu_plate_Khan_Khan(Re=reynolds_number, Pr=prandtl_number, chevron_angle=chevron_angle)
                        paired_friction_correlation = "friction_plate_Kumar"  # Commonly paired with Kumar friction
                        correlation_notes.append("Khan_Khan correlation specific to low-Re liquids")

                    if nusselt_number is not None:
                        convection_coefficient = nusselt_number * thermal_conductivity / characteristic_dimension

                        # Build PHE-specific result
                        result = {
                            "convection_coefficient_h": convection_coefficient,
                            "convection_coefficient_h_W_m2K": convection_coefficient,
                            "nusselt_number": nusselt_number,
                            "geometry": geometry,
                            "flow_type": "forced",
                            "data_source": fluid_props.get("data_source"),
                            "fluid_properties": {
                                "name": fluid_name,
                                "density": density,
                                "dynamic_viscosity": dynamic_viscosity,
                                "thermal_conductivity": thermal_conductivity,
                                "prandtl_number": prandtl_number,
                                "film_temperature": film_temperature,
                            },
                            "calculation_details": {
                                "reynolds_number": reynolds_number,
                                "fluid_velocity": fluid_velocity,
                                "hydraulic_diameter_m": characteristic_dimension,
                                "chevron_angle_deg": chevron_angle,
                                "phe_correlation": phe_correlation,
                                "paired_friction_correlation": paired_friction_correlation,
                            },
                            "unit": "W/(m²·K)",
                        }

                        if plate_enlargement_factor is not None:
                            result["calculation_details"]["plate_enlargement_factor"] = plate_enlargement_factor

                        if mu_wall is not None:
                            result["calculation_details"]["viscosity_correction_applied"] = True
                            result["calculation_details"]["mu_wall_Pa_s"] = mu_wall

                        if correlation_notes:
                            result["warnings"] = correlation_notes

                        return json.dumps(result)

                # Use HT library correlations directly instead of manual implementations
                elif "flat_plate_external" in geometry_lower and flow_type.lower() == "forced":
                    # Use ht.conv_external meta-function for plates
                    from ht.conv_external import Nu_external_horizontal_plate

                    nusselt_number = Nu_external_horizontal_plate(reynolds_number, prandtl_number)

                elif "pipe_internal" in geometry_lower and flow_type.lower() == "forced":
                    from ht.conv_internal import Nu_conv_internal

                    eD = roughness / characteristic_dimension if roughness else 0.0
                    kwargs = {"eD": eD}
                    if pipe_length is not None and characteristic_dimension is not None:
                        kwargs.update({"Di": characteristic_dimension, "x": pipe_length})
                    nusselt_number = Nu_conv_internal(reynolds_number, prandtl_number, **kwargs)

                elif "vertical" in geometry_lower and "cylinder" in geometry_lower and flow_type.lower() == "natural":
                    # Vertical cylinder natural convection - use proper vertical cylinder correlation
                    from ht.conv_free_immersed import Nu_vertical_cylinder

                    g = 9.81  # m/s²
                    # Get thermal expansion coefficient (beta) - use proper value for liquids
                    beta, is_approx, beta_warning = _get_thermal_expansion_coefficient(
                        fluid_name, film_temperature, pressure, property_source
                    )
                    if beta_warning:
                        logger.warning(beta_warning)
                    delta_t = abs(surface_temperature - bulk_fluid_temperature)
                    kinematic_viscosity = dynamic_viscosity / density
                    grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                    nusselt_number = Nu_vertical_cylinder(prandtl_number, grashof, L=characteristic_dimension)

                elif "pipe_external" in geometry_lower or "cylinder" in geometry_lower:
                    if flow_type.lower() == "forced":
                        # Use ht.conv_external meta-function for cylinder; include wall Pr if available
                        from ht.conv_external import Nu_external_cylinder

                        wall_props = calculation_properties(fluid_name, surface_temperature, pressure, property_source)
                        prandtl_wall = (wall_props["dynamic_viscosity"] * wall_props["specific_heat_cp"]
                                        / wall_props["thermal_conductivity"])
                        nusselt_number = Nu_external_cylinder(reynolds_number, prandtl_number, Prw=prandtl_wall)
                    else:  # Natural convection for horizontal cylinders
                        # Use ht.conv_free_immersed for natural convection around horizontal cylinder
                        from ht.conv_free_immersed import Nu_horizontal_cylinder

                        # Calculate Grashof number for ht function
                        g = 9.81  # m/s²
                        # Get thermal expansion coefficient (beta) - use proper value for liquids
                        beta, is_approx, beta_warning = _get_thermal_expansion_coefficient(
                            fluid_name, film_temperature, pressure, property_source
                        )
                        if beta_warning:
                            logger.warning(beta_warning)
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density
                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        nusselt_number = Nu_horizontal_cylinder(prandtl_number, grashof)

                elif "sphere" in geometry_lower:
                    if flow_type.lower() == "forced":
                        # Whitaker correlation for forced flow over sphere (standard textbook)
                        # Nu = 2 + (0.4*Re^0.5 + 0.06*Re^(2/3)) * Pr^0.4 * (mu/mu_s)^0.25
                        # Note: No ht library equivalent exists for forced sphere convection
                        # Viscosity ratio correction omitted (approximation)
                        nusselt_number = (
                            2 + (0.4 * math.sqrt(reynolds_number) + 0.06 * reynolds_number ** (2 / 3)) * prandtl_number**0.4
                        )
                    else:  # Natural convection - use ht library
                        from ht.conv_free_immersed import Nu_sphere_Churchill

                        g = 9.81
                        # Get thermal expansion coefficient (beta) - use proper value for liquids
                        beta, is_approx, beta_warning = _get_thermal_expansion_coefficient(
                            fluid_name, film_temperature, pressure, property_source
                        )
                        if beta_warning:
                            logger.warning(beta_warning)
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density
                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        rayleigh = grashof * prandtl_number
                        if rayleigh >= 1e13:
                            raise ValueError("Churchill sphere requires Ra < 1e13")
                        nusselt_number = Nu_sphere_Churchill(Pr=prandtl_number, Gr=grashof)

                elif ("vertical" in geometry_lower or "wall" in geometry_lower) and flow_type.lower() == "natural":
                    from ht.conv_free_immersed import Nu_free_vertical_plate

                    g = 9.81
                    # Get thermal expansion coefficient (beta) - use proper value for liquids
                    beta, is_approx, beta_warning = _get_thermal_expansion_coefficient(
                        fluid_name, film_temperature, pressure, property_source
                    )
                    if beta_warning:
                        logger.warning(beta_warning)
                    delta_t = abs(surface_temperature - bulk_fluid_temperature)
                    kinematic_viscosity = dynamic_viscosity / density
                    grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                    nusselt_number = Nu_free_vertical_plate(prandtl_number, grashof)
                else:
                    # For other geometries, try a fallback approach
                    if flow_type.lower() == "forced":
                        nusselt_number = calculate_nusselt_number_external_flow(reynolds_number, prandtl_number, geometry)
                    else:  # Natural
                        # Basic natural convection for various geometries
                        g = 9.81
                        # Get thermal expansion coefficient (beta) - use proper value for liquids
                        beta, is_approx, beta_warning = _get_thermal_expansion_coefficient(
                            fluid_name, film_temperature, pressure, property_source
                        )
                        if beta_warning:
                            logger.warning(beta_warning)
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density

                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        rayleigh = grashof * prandtl_number

                        # Generic natural convection - use appropriate correlation based on geometry
                        if "vertical" in geometry_lower:
                            if rayleigh < 1e9:  # Laminar
                                nusselt_number = 0.59 * rayleigh ** (1 / 4)
                            else:  # Turbulent
                                nusselt_number = 0.1 * rayleigh ** (1 / 3)
                        elif "horizontal" in geometry_lower and ("plate_up" in geometry_lower or "plate_down" in geometry_lower):
                            from ht.conv_free_immersed import Nu_horizontal_plate_VDI
                            buoyancy = ("plate_up" in geometry_lower) == (surface_temperature > bulk_fluid_temperature)
                            nusselt_number = Nu_horizontal_plate_VDI(Pr=prandtl_number, Gr=grashof, buoyancy=buoyancy)
                        else:
                            raise ValueError("Natural convection requires a qualified geometry and, for horizontal plates, plate_up or plate_down; characteristic length is area/perimeter")

            except Exception as ht_error:
                # Fail loudly - do not silently fall back to basic correlations
                logger.error(f"Error using HT library correlations: {ht_error}")
                raise

        # Verify we calculated a Nusselt number
        if nusselt_number is None:
            raise ValueError(f"Could not calculate Nusselt number for geometry='{geometry}', flow_type='{flow_type}'")

        # Calculate convection coefficient from Nusselt number
        convection_coefficient = nusselt_number * thermal_conductivity / characteristic_dimension

        # Prepare result
        result = {
            "convection_coefficient_h": convection_coefficient,
            "convection_coefficient_h_W_m2K": convection_coefficient,
            "nusselt_number": nusselt_number,
            "geometry": geometry,
            "flow_type": flow_type,
            "data_source": fluid_props.get("data_source"),
            "fluid_properties": {
                "name": fluid_name,
                "density": density,
                "dynamic_viscosity": dynamic_viscosity,
                "thermal_conductivity": thermal_conductivity,
                "prandtl_number": prandtl_number,
                "film_temperature": film_temperature,
            },
            "calculation_details": {},
        }

        # Add flow-specific details
        if flow_type.lower() == "forced":
            result["calculation_details"]["reynolds_number"] = reynolds_number
            result["calculation_details"]["fluid_velocity"] = fluid_velocity
        else:  # Natural convection
            g = 9.81
            beta, _, _ = _get_thermal_expansion_coefficient(fluid_name, film_temperature, pressure, property_source)
            delta_t = abs(surface_temperature - bulk_fluid_temperature)
            if density and dynamic_viscosity:
                kinematic_viscosity = dynamic_viscosity / density
                grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                rayleigh = grashof * prandtl_number if prandtl_number else None

                result["calculation_details"]["isobaric_expansion_per_K"] = beta
                result["calculation_details"]["grashof_number"] = grashof
                result["calculation_details"]["rayleigh_number"] = rayleigh
                result["calculation_details"]["temperature_difference"] = delta_t

        # Add units for clarity
        result["unit"] = "W/(m²·K)"

        return json.dumps(result)

    except PropertyLookupRefused as refused:
        return property_refusal(refused, "Failed to get fluid properties")
    except Exception as e:
        logger.error(f"Unexpected error in calculate_convection_coefficient: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
