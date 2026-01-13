"""
Convection coefficient tool to calculate heat transfer coefficients for various geometries.

This module provides functionality to calculate convective heat transfer coefficients
using established correlations from the HT library or fallback implementations.
"""

import json
import logging
import math
from typing import Dict, List, Optional, Union, Any

from utils.constants import DEG_C_to_K
from utils.import_helpers import HT_AVAILABLE
from utils.helpers import (
    calculate_reynolds_number, 
    calculate_prandtl_number, 
    calculate_nusselt_number_external_flow
)

# Tools might need to call other tools 
from tools.fluid_properties import get_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.convection_coefficient")


def calculate_two_phase_h(
    mass_flux_kg_m2s: float,
    quality: float,
    tube_diameter_m: float,
    fluid_name: str,
    saturation_temperature_K: float,
    pressure_Pa: float = 101325.0,
    heat_flux_W_m2: Optional[float] = None,
    method: str = "auto",
    strict: bool = False,
) -> str:
    """Calculate two-phase heat transfer coefficient using ht.conv_two_phase correlations.

    This provides basic two-phase (boiling/condensation) heat transfer capability using
    the ht library's established correlations. Supports 9+ methods including Shah, Chen,
    Kandlikar, and more.

    Args:
        mass_flux_kg_m2s: Mass flux G (kg/m²s). G = mass_flow_rate / cross_sectional_area
        quality: Vapor quality x (0 to 1, dimensionless). 0 = all liquid, 1 = all vapor
        tube_diameter_m: Inner tube diameter (m)
        fluid_name: Fluid name (e.g., 'water', 'R134a', 'ammonia')
        saturation_temperature_K: Saturation temperature (K)
        pressure_Pa: Saturation pressure (Pa). Used for fluid property lookup
        heat_flux_W_m2: Heat flux (W/m²). Required for some correlations like Kandlikar
        method: Correlation method - 'auto' (recommended), 'Shah_1976', 'Chen_Edelstein',
                'Kandlikar', 'Liu_Winterton', 'Thome', 'Sun_Mishima', 'Lazarek_Black',
                'Li_Wu', 'Cavallini_Smith_Zivi'
        strict: If True, require ht library and fail if not available

    Returns:
        JSON string with two-phase heat transfer coefficient and calculation details

    Example:
        >>> calculate_two_phase_h(
        ...     mass_flux_kg_m2s=500,
        ...     quality=0.3,
        ...     tube_diameter_m=0.01,
        ...     fluid_name="water",
        ...     saturation_temperature_K=373.15,
        ...     pressure_Pa=101325
        ... )
    """
    try:
        # Input validation
        if quality < 0 or quality > 1:
            return json.dumps({"error": "Quality must be between 0 and 1"})
        if mass_flux_kg_m2s <= 0:
            return json.dumps({"error": "Mass flux must be positive"})
        if tube_diameter_m <= 0:
            return json.dumps({"error": "Tube diameter must be positive"})
        if saturation_temperature_K <= 0:
            return json.dumps({"error": "Saturation temperature must be positive"})

        if not HT_AVAILABLE:
            if strict:
                return json.dumps({"error": "ht library required for two-phase calculations"})
            return json.dumps({
                "error": "ht library not available for two-phase calculations",
                "suggestion": "Install with: pip install ht>=1.2.0"
            })

        # Get fluid properties at saturation
        fluid_props_json = get_fluid_properties(fluid_name, saturation_temperature_K, pressure_Pa, strict=strict)
        fluid_props = json.loads(fluid_props_json)

        if "error" in fluid_props:
            return json.dumps({
                "error": f"Failed to get fluid properties: {fluid_props['error']}",
                "note": "Two-phase calculations require accurate liquid/vapor properties"
            })

        # Extract properties
        rho_l = fluid_props.get("density")  # Liquid density
        mu_l = fluid_props.get("dynamic_viscosity")  # Liquid viscosity
        k_l = fluid_props.get("thermal_conductivity")  # Liquid thermal conductivity
        Cp_l = fluid_props.get("specific_heat_cp")  # Liquid specific heat
        Pr_l = fluid_props.get("prandtl_number")

        if None in [rho_l, mu_l, k_l]:
            return json.dumps({
                "error": "Missing critical fluid properties for two-phase calculation"
            })

        # Try to import ht two-phase module
        import ht
        from ht.conv_two_phase import h_two_phase

        # Build kwargs for h_two_phase
        kwargs = {
            "m": mass_flux_kg_m2s,
            "x": quality,
            "D": tube_diameter_m,
            "rhol": rho_l,
            "mul": mu_l,
            "kl": k_l,
        }

        # Add vapor properties if available (estimated from ideal gas for now)
        # For more accurate vapor properties, CoolProp integration is recommended
        try:
            # Estimate vapor properties (simplified)
            # In practice, these should come from proper saturation property tables
            rho_v = rho_l * 0.001  # Very rough vapor density estimate
            mu_v = mu_l * 0.01  # Rough vapor viscosity estimate

            # Try to get better vapor properties using thermo/CoolProp
            try:
                from utils.import_helpers import COOLPROP_AVAILABLE
                if COOLPROP_AVAILABLE:
                    import CoolProp.CoolProp as CP
                    cp_fluid = fluid_name.lower()
                    # Map common names
                    fluid_map = {"water": "Water", "r134a": "R134a", "ammonia": "Ammonia"}
                    cp_name = fluid_map.get(cp_fluid, cp_fluid.capitalize())

                    try:
                        rho_v = CP.PropsSI("D", "T", saturation_temperature_K, "Q", 1, cp_name)
                        mu_v = CP.PropsSI("V", "T", saturation_temperature_K, "Q", 1, cp_name)
                    except Exception:
                        pass  # Keep estimates
            except ImportError:
                pass

            kwargs["rhog"] = rho_v
            kwargs["mug"] = mu_v
        except Exception:
            pass

        # Add heat flux if provided (required for Kandlikar and some others)
        if heat_flux_W_m2 is not None:
            kwargs["q"] = heat_flux_W_m2

        # Add Cp if available
        if Cp_l is not None:
            kwargs["Cpl"] = Cp_l

        # Calculate h using ht library
        h_tp = None
        used_method = None

        # Available methods in ht.conv_two_phase
        available_methods = [
            "Shah_1976", "Chen_Edelstein", "Kandlikar", "Liu_Winterton",
            "Thome", "Sun_Mishima", "Lazarek_Black", "Li_Wu", "Cavallini_Smith_Zivi"
        ]

        if method.lower() == "auto":
            # Let ht choose automatically
            try:
                h_tp = h_two_phase(**kwargs)
                used_method = "auto (ht selection)"
            except Exception as e:
                logger.debug(f"h_two_phase auto failed: {e}")
        else:
            # Use specified method
            if method not in available_methods:
                return json.dumps({
                    "error": f"Unknown method: {method}",
                    "available_methods": available_methods
                })
            try:
                h_tp = h_two_phase(Method=method, **kwargs)
                used_method = method
            except Exception as e:
                return json.dumps({
                    "error": f"Method {method} failed: {str(e)}",
                    "suggestion": "Try method='auto' or check required parameters for this method"
                })

        if h_tp is None:
            return json.dumps({
                "error": "Could not calculate two-phase heat transfer coefficient",
                "suggestion": "Check input parameters or try a different method"
            })

        # Calculate related dimensionless numbers
        Re_l = mass_flux_kg_m2s * tube_diameter_m / mu_l
        Pr_l_calc = mu_l * Cp_l / k_l if Cp_l and k_l else Pr_l

        result = {
            "two_phase_h_W_m2K": h_tp,
            "method_used": used_method,
            "input_parameters": {
                "mass_flux_kg_m2s": mass_flux_kg_m2s,
                "quality": quality,
                "tube_diameter_m": tube_diameter_m,
                "saturation_temperature_K": saturation_temperature_K,
                "heat_flux_W_m2": heat_flux_W_m2,
            },
            "fluid_properties": {
                "name": fluid_name,
                "liquid_density_kg_m3": rho_l,
                "liquid_viscosity_Pa_s": mu_l,
                "liquid_thermal_conductivity_W_mK": k_l,
                "liquid_specific_heat_J_kgK": Cp_l,
            },
            "dimensionless_numbers": {
                "Reynolds_liquid": Re_l,
                "Prandtl_liquid": Pr_l_calc,
            },
            "available_methods": available_methods,
            "notes": [
                "Two-phase h depends strongly on vapor quality and flow regime",
                "For best accuracy, use CoolProp for fluid properties",
                "Heat flux (q) is required for nucleate boiling correlations"
            ]
        }

        return json.dumps(result)

    except ImportError as e:
        return json.dumps({
            "error": f"Required library not available: {e}",
            "suggestion": "Install ht library with: pip install ht>=1.2.0"
        })
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

    Returns:
        JSON string with the calculated convection coefficient and related parameters.
        For plate_chevron, also includes the paired friction correlation name.
    """
    # Validate inputs to avoid misleading errors and non-physical results
    try:
        if not isinstance(geometry, str) or not geometry.strip():
            return json.dumps({
                "error": "Geometry must be a non-empty string."
            })
        geometry_lower = geometry.lower()
        recognized_keywords = ['flat_plate', 'plate', 'pipe', 'cylinder', 'sphere', 'vertical', 'horizontal', 'wall', 'plate_chevron', 'chevron']
        if not any(k in geometry_lower for k in recognized_keywords):
            return json.dumps({
                "error": f"Unsupported geometry: {geometry}."
            })

        # Validate PHE-specific parameters for plate_chevron geometry
        if 'plate_chevron' in geometry_lower or 'chevron' in geometry_lower:
            if chevron_angle is None:
                return json.dumps({
                    "error": "chevron_angle is required for plate_chevron geometry."
                })
            if chevron_angle < 0 or chevron_angle > 90:
                return json.dumps({
                    "error": "chevron_angle must be between 0 and 90 degrees."
                })
            valid_phe_correlations = ['Kumar', 'Martin_1999', 'Martin_VDI', 'Muley_Manglik', 'Khan_Khan']
            if phe_correlation not in valid_phe_correlations:
                return json.dumps({
                    "error": f"Invalid phe_correlation: {phe_correlation}. Valid options: {valid_phe_correlations}"
                })
            if phe_correlation == 'Muley_Manglik' and plate_enlargement_factor is None:
                return json.dumps({
                    "error": "plate_enlargement_factor is required for Muley_Manglik correlation."
                })
        if flow_type is None or flow_type.lower() not in {"forced", "natural"}:
            return json.dumps({
                "error": "flow_type must be 'forced' or 'natural'."
            })
        # Characteristic dimension
        try:
            characteristic_dimension = float(characteristic_dimension)
        except (TypeError, ValueError):
            return json.dumps({
                "error": "Characteristic dimension must be a numeric value in meters."
            })
        if not math.isfinite(characteristic_dimension) or characteristic_dimension <= 0.0:
            return json.dumps({
                "error": "Characteristic dimension must be a positive, finite value."
            })
        # Forced convection requires a valid velocity
        if flow_type.lower() == 'forced':
            if fluid_velocity is None:
                return json.dumps({
                    "error": "Fluid velocity is required for forced convection calculations."
                })
            try:
                fluid_velocity = float(fluid_velocity)
            except (TypeError, ValueError):
                return json.dumps({
                    "error": "Fluid velocity must be a numeric value in m/s."
                })
            if not math.isfinite(fluid_velocity) or fluid_velocity <= 0.0:
                return json.dumps({
                    "error": "Fluid velocity must be a positive, finite value for forced convection."
                })
        # Temperatures and pressure
        try:
            bulk_fluid_temperature = float(bulk_fluid_temperature)
            surface_temperature = float(surface_temperature)
            pressure = float(pressure)
        except (TypeError, ValueError):
            return json.dumps({
                "error": "Temperatures (K) and pressure (Pa) must be numeric values."
            })
        for T_val, label in [(bulk_fluid_temperature, 'bulk_fluid_temperature'), (surface_temperature, 'surface_temperature')]:
            if not math.isfinite(T_val):
                return json.dumps({
                    "error": f"{label} must be a finite real number."
                })
            if T_val < 0.0:
                return json.dumps({
                    "error": f"{label} cannot be below 0 K (absolute zero)."
                })
            if T_val > 1.0e4:
                return json.dumps({
                    "error": f"{label} is unrealistically high. Please provide < 10000 K."
                })
        if not math.isfinite(pressure) or pressure <= 0.0:
            return json.dumps({
                "error": "Pressure must be a positive, finite value."
            })
        if roughness is not None:
            try:
                roughness = float(roughness)
            except (TypeError, ValueError):
                return json.dumps({
                    "error": "Roughness must be a numeric value in meters."
                })
            if not math.isfinite(roughness) or roughness < 0.0:
                return json.dumps({
                    "error": "Roughness must be a non-negative, finite value."
                })

        # Calculate film temperature (average of bulk and surface)
        film_temperature = (bulk_fluid_temperature + surface_temperature) / 2.0
        
        # Get fluid properties at film temperature
        fluid_props_json = get_fluid_properties(fluid_name, film_temperature, pressure, strict=strict)
        fluid_props = json.loads(fluid_props_json)
        
        if "error" in fluid_props:
            return json.dumps({
                "error": f"Failed to get fluid properties: {fluid_props['error']}"
            })
        
        # Extract required properties
        density = fluid_props.get("density")
        dynamic_viscosity = fluid_props.get("dynamic_viscosity")
        thermal_conductivity = fluid_props.get("thermal_conductivity")
        specific_heat_cp = fluid_props.get("specific_heat_cp")
        prandtl_number = fluid_props.get("prandtl_number")
        
        # Check for required properties
        if None in [density, dynamic_viscosity, thermal_conductivity]:
            return json.dumps({
                "error": "Missing critical fluid properties for convection calculation."
            })
        
        # If Prandtl number not provided, calculate it
        if prandtl_number is None and specific_heat_cp is not None:
            prandtl_number = calculate_prandtl_number(
                dynamic_viscosity, specific_heat_cp, thermal_conductivity
            )
        
        # Calculate Reynolds number for forced flow
        reynolds_number = None
        if flow_type.lower() == 'forced' and fluid_velocity is not None:
            reynolds_number = calculate_reynolds_number(
                fluid_velocity, characteristic_dimension, density, dynamic_viscosity
            )
        
        # Calculate Nusselt number and convection coefficient based on geometry and flow regime
        result = {}
        geometry_lower = geometry.lower()
        nusselt_number = None
        convection_coefficient = None
        
        if strict and not HT_AVAILABLE:
            raise ImportError("ht library required with strict=True")
            
        if HT_AVAILABLE:
            import ht

            try:
                # Handle plate_chevron geometry first (PHE convection coefficient)
                if 'plate_chevron' in geometry_lower or ('chevron' in geometry_lower and 'plate' not in geometry_lower):
                    # Import PHE-specific correlations from ht.conv_plate
                    from ht.conv_plate import (
                        Nu_plate_Kumar,
                        Nu_plate_Martin,
                        Nu_plate_Muley_Manglik,
                        Nu_plate_Khan_Khan
                    )

                    # PHE is always forced convection
                    if reynolds_number is None or reynolds_number <= 0:
                        return json.dumps({
                            "error": "Reynolds number must be positive for plate_chevron geometry. Ensure fluid_velocity is provided."
                        })

                    # Get wall viscosity for Kumar correlation (viscosity correction)
                    mu_wall = None
                    if phe_correlation == 'Kumar':
                        try:
                            wall_props = json.loads(get_fluid_properties(fluid_name, surface_temperature, pressure, strict=strict))
                            if "error" not in wall_props:
                                mu_wall = wall_props.get("dynamic_viscosity")
                        except Exception:
                            pass

                    # Calculate Nusselt number using appropriate correlation
                    # Note: chevron_angle for ht functions is in degrees
                    paired_friction_correlation = None
                    correlation_notes = []

                    if phe_correlation == 'Kumar':
                        # Kumar correlation (APV data-based)
                        # Valid range: Re 0.1-10000, chevron 30-65°
                        if chevron_angle < 30 or chevron_angle > 65:
                            correlation_notes.append(f"Warning: Kumar correlation valid for chevron 30-65°, got {chevron_angle}°")
                        if reynolds_number < 0.1 or reynolds_number > 10000:
                            correlation_notes.append(f"Warning: Kumar correlation valid for Re 0.1-10000, got Re={reynolds_number:.1f}")

                        if mu_wall is not None and dynamic_viscosity is not None:
                            nusselt_number = Nu_plate_Kumar(
                                Re=reynolds_number,
                                Pr=prandtl_number,
                                chevron_angle=chevron_angle,
                                mu=dynamic_viscosity,
                                mu_wall=mu_wall
                            )
                        else:
                            nusselt_number = Nu_plate_Kumar(
                                Re=reynolds_number,
                                Pr=prandtl_number,
                                chevron_angle=chevron_angle
                            )
                        paired_friction_correlation = "friction_plate_Kumar"

                    elif phe_correlation in ['Martin_1999', 'Martin_VDI']:
                        # Martin correlation (theoretical)
                        # Valid range: Re 200-10000, chevron 0-80°
                        if chevron_angle < 0 or chevron_angle > 80:
                            correlation_notes.append(f"Warning: Martin correlation valid for chevron 0-80°, got {chevron_angle}°")
                        if reynolds_number < 200 or reynolds_number > 10000:
                            correlation_notes.append(f"Warning: Martin correlation valid for Re 200-10000, got Re={reynolds_number:.1f}")

                        variant = '1999' if phe_correlation == 'Martin_1999' else 'VDI'
                        nusselt_number = Nu_plate_Martin(
                            Re=reynolds_number,
                            Pr=prandtl_number,
                            chevron_angle=chevron_angle,
                            variant=variant
                        )
                        paired_friction_correlation = f"friction_plate_Martin_{variant}"

                    elif phe_correlation == 'Muley_Manglik':
                        # Muley-Manglik correlation (includes enlargement factor)
                        # Valid range: Re > 1000, chevron 30-60°
                        if chevron_angle < 30 or chevron_angle > 60:
                            correlation_notes.append(f"Warning: Muley_Manglik valid for chevron 30-60°, got {chevron_angle}°")
                        if reynolds_number < 1000:
                            correlation_notes.append(f"Warning: Muley_Manglik valid for Re > 1000, got Re={reynolds_number:.1f}")

                        nusselt_number = Nu_plate_Muley_Manglik(
                            Re=reynolds_number,
                            Pr=prandtl_number,
                            chevron_angle=chevron_angle,
                            plate_enlargement_factor=plate_enlargement_factor
                        )
                        paired_friction_correlation = "friction_plate_Muley_Manglik"

                    elif phe_correlation == 'Khan_Khan':
                        # Khan & Khan correlation (low-Re liquids)
                        nusselt_number = Nu_plate_Khan_Khan(
                            Re=reynolds_number,
                            Pr=prandtl_number,
                            chevron_angle=chevron_angle
                        )
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
                            "fluid_properties": {
                                "name": fluid_name,
                                "density": density,
                                "dynamic_viscosity": dynamic_viscosity,
                                "thermal_conductivity": thermal_conductivity,
                                "prandtl_number": prandtl_number,
                                "film_temperature": film_temperature
                            },
                            "calculation_details": {
                                "reynolds_number": reynolds_number,
                                "fluid_velocity": fluid_velocity,
                                "hydraulic_diameter_m": characteristic_dimension,
                                "chevron_angle_deg": chevron_angle,
                                "phe_correlation": phe_correlation,
                                "paired_friction_correlation": paired_friction_correlation
                            },
                            "unit": "W/(m²·K)"
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
                elif 'flat_plate_external' in geometry_lower and flow_type.lower() == 'forced':
                    # Use ht.conv_external meta-function for plates
                    from ht.conv_external import Nu_external_horizontal_plate
                    nusselt_number = Nu_external_horizontal_plate(reynolds_number, prandtl_number)
                
                elif 'pipe_internal' in geometry_lower and flow_type.lower() == 'forced':
                    from ht.conv_internal import Nu_conv_internal
                    eD = roughness / characteristic_dimension if roughness else 0.0
                    kwargs = {"eD": eD}
                    if pipe_length is not None and characteristic_dimension is not None:
                        kwargs.update({"Di": characteristic_dimension, "x": pipe_length})
                    nusselt_number = Nu_conv_internal(reynolds_number, prandtl_number, **kwargs)
                
                elif 'pipe_external' in geometry_lower or 'cylinder' in geometry_lower:
                    if flow_type.lower() == 'forced':
                        # Use ht.conv_external meta-function for cylinder; include wall Pr if available
                        from ht.conv_external import Nu_external_cylinder
                        prandtl_wall = None
                        try:
                            wall_props = json.loads(get_fluid_properties(fluid_name, surface_temperature, pressure, strict=strict))
                            if "error" not in wall_props:
                                mu_w = wall_props.get("dynamic_viscosity")
                                Cp_w = wall_props.get("specific_heat_cp")
                                k_w = wall_props.get("thermal_conductivity")
                                if mu_w and Cp_w and k_w:
                                    prandtl_wall = mu_w * Cp_w / k_w
                        except Exception:
                            pass
                        if prandtl_wall is not None:
                            nusselt_number = Nu_external_cylinder(reynolds_number, prandtl_number, Prw=prandtl_wall)
                        else:
                            nusselt_number = Nu_external_cylinder(reynolds_number, prandtl_number)
                    else:  # Natural convection
                        # Use ht.conv_free_immersed for natural convection around cylinder
                        from ht.conv_free_immersed import Nu_horizontal_cylinder
                        # Calculate Grashof number for ht function
                        g = 9.81  # m/s²
                        beta = 1.0 / film_temperature  # Thermal expansion coefficient (approximation)
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density
                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        nusselt_number = Nu_horizontal_cylinder(prandtl_number, grashof)
                
                elif 'sphere' in geometry_lower:
                    if flow_type.lower() == 'forced':
                        # Whitaker correlation for forced flow over sphere (standard textbook)
                        # Nu = 2 + (0.4*Re^0.5 + 0.06*Re^(2/3)) * Pr^0.4 * (mu/mu_s)^0.25
                        # Note: No ht library equivalent exists for forced sphere convection
                        # Viscosity ratio correction omitted (approximation)
                        nusselt_number = 2 + (0.4 * math.sqrt(reynolds_number) + 0.06 * reynolds_number**(2/3)) * prandtl_number**0.4
                    else:  # Natural convection - use ht library
                        from ht.conv_free_immersed import Nu_sphere_Churchill
                        g = 9.81
                        beta = 1.0 / film_temperature
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density
                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        rayleigh = grashof * prandtl_number
                        nusselt_number = Nu_sphere_Churchill(prandtl_number, rayleigh)
                
                elif ('vertical' in geometry_lower or 'wall' in geometry_lower) and flow_type.lower() == 'natural':
                    from ht.conv_free_immersed import Nu_free_vertical_plate
                    g = 9.81
                    beta = 1.0 / film_temperature
                    delta_t = abs(surface_temperature - bulk_fluid_temperature)
                    kinematic_viscosity = dynamic_viscosity / density
                    grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                    nusselt_number = Nu_free_vertical_plate(prandtl_number, grashof)
                else:
                    # For other geometries, try a fallback approach
                    if flow_type.lower() == 'forced':
                        nusselt_number = calculate_nusselt_number_external_flow(
                            reynolds_number, prandtl_number, geometry)
                    else:  # Natural
                        # Basic natural convection for various geometries
                        g = 9.81
                        beta = 1.0 / film_temperature
                        delta_t = abs(surface_temperature - bulk_fluid_temperature)
                        kinematic_viscosity = dynamic_viscosity / density
                        
                        grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                        rayleigh = grashof * prandtl_number
                        
                        # Generic natural convection - use appropriate correlation based on geometry
                        if 'vertical' in geometry_lower:
                            if rayleigh < 1e9:  # Laminar
                                nusselt_number = 0.59 * rayleigh**(1/4)
                            else:  # Turbulent
                                nusselt_number = 0.1 * rayleigh**(1/3)
                        elif 'horizontal' in geometry_lower:
                            if 'plate_up' in geometry_lower:  # Upper surface of heated plate
                                if rayleigh < 1e7:  # Laminar
                                    nusselt_number = 0.54 * rayleigh**(1/4)
                                else:  # Turbulent
                                    nusselt_number = 0.15 * rayleigh**(1/3)
                            else:  # Lower surface of heated plate or upper surface of cooled plate
                                nusselt_number = 0.27 * rayleigh**(1/4)
                        else:
                            # Default approximation
                            nusselt_number = 0.5 * rayleigh**(1/4)
                
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
            "fluid_properties": {
                "name": fluid_name,
                "density": density,
                "dynamic_viscosity": dynamic_viscosity,
                "thermal_conductivity": thermal_conductivity,
                "prandtl_number": prandtl_number,
                "film_temperature": film_temperature
            },
            "calculation_details": {}
        }
        
        # Add flow-specific details
        if flow_type.lower() == 'forced':
            result["calculation_details"]["reynolds_number"] = reynolds_number
            result["calculation_details"]["fluid_velocity"] = fluid_velocity
        else:  # Natural convection
            g = 9.81
            beta = 1.0 / film_temperature
            delta_t = abs(surface_temperature - bulk_fluid_temperature)
            if density and dynamic_viscosity:
                kinematic_viscosity = dynamic_viscosity / density
                grashof = (g * beta * delta_t * characteristic_dimension**3) / (kinematic_viscosity**2)
                rayleigh = grashof * prandtl_number if prandtl_number else None
                
                result["calculation_details"]["grashof_number"] = grashof
                result["calculation_details"]["rayleigh_number"] = rayleigh
                result["calculation_details"]["temperature_difference"] = delta_t
        
        # Add units for clarity
        result["unit"] = "W/(m²·K)"
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in calculate_convection_coefficient: {e}", exc_info=True)
        return json.dumps({
            "error": f"An unexpected error occurred: {str(e)}"
        })
