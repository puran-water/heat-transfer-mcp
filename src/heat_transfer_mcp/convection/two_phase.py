"""Two-phase convection adapters around the installed ht APIs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from heat_transfer_mcp.convection.single_phase import validate_positive_finite
from heat_transfer_mcp.errors import CalculationError
from heat_transfer_mcp.property_validation import require_thermo_method, thermo_chemical
from heat_transfer_mcp.tools.fluid_properties import STRICT_PROPERTY_RULE, STRICT_PROPERTY_SOURCE

SUPPORTED_METHODS = [
    "auto",
    "Shah_1976",
    "Cavallini_Smith_Zecchin",
    "Sun_Mishima",
    "Liu_Winterton",
    "Chen_Edelstein",
]

CAVALLINI_SMITH_ZECCHIN_MASS_FLUX_RANGE_KG_M2S = (20.0, 3000.0)
CAVALLINI_SMITH_ZECCHIN_MAX_MACH = 0.8
CAVALLINI_SMITH_ZECCHIN_SCOPE_SOURCE = (
    "Cavallini, Smith & Zecchin (1974) forced-convection condensation; "
    "adapter validity envelope for conventional in-tube condensation data"
)


@dataclass(frozen=True)
class SaturationProperties:
    fluid_name: str
    saturation_temperature_K: float
    pressure_Pa: float
    source: str
    rhol: float
    rhog: float
    mul: float
    mug: float
    kl: float
    Cpl: float
    kg: Optional[float] = None
    Cpg: Optional[float] = None
    Hvap: Optional[float] = None
    sigma: Optional[float] = None
    Psat: Optional[float] = None
    Pc: Optional[float] = None
    MW: Optional[float] = None
    property_methods: Dict[str, Any] = field(default_factory=dict)


def calculate_two_phase_coefficient(
    *,
    quality: float,
    tube_diameter_m: float,
    fluid_name: str,
    saturation_temperature_K: float,
    mass_flow_kg_s: Optional[float] = None,
    mass_flux_kg_m2s: Optional[float] = None,
    method: str = "auto",
    strict: bool = False,
    length_m: Optional[float] = None,
    void_fraction: Optional[float] = None,
    excess_temperature_K: Optional[float] = None,
    property_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate two-phase h with mass-flow semantics expected by ht; saturation properties per get_saturation_properties.

    The installed ``ht.conv_two_phase.h_two_phase`` API takes mass flow rate
    ``m`` in kg/s, not mass flux. If mass flux is provided by callers, convert
    it through the tube flow area before calling ht.
    """
    quality = _validate_quality(quality)
    tube_diameter_m = validate_positive_finite(tube_diameter_m, "tube_diameter_m")
    saturation_temperature_K = validate_positive_finite(saturation_temperature_K, "saturation_temperature_K")
    mass_flow, mass_flux, mass_warning = _resolve_mass_flow(
        mass_flow_kg_s=mass_flow_kg_s,
        mass_flux_kg_m2s=mass_flux_kg_m2s,
        tube_diameter_m=tube_diameter_m,
    )
    length = validate_positive_finite(length_m, "length_m") if length_m is not None else 1.0

    method_key = _normalize_method(method)
    properties = get_saturation_properties(
        fluid_name=fluid_name,
        saturation_temperature_K=saturation_temperature_K,
        strict=strict,
        property_source=property_source,
    )
    pressure_Pa = properties.pressure_Pa
    alpha = _resolve_void_fraction(void_fraction, quality, properties)
    if method_key in {"Sun_Mishima", "Liu_Winterton", "Chen_Edelstein"} and excess_temperature_K is None:
        raise CalculationError(code="missing_boiling_superheat", message="Boiling requires supplied excess_temperature_K; no assumed 5 K wall superheat")
    excess_temperature = validate_positive_finite(excess_temperature_K, "excess_temperature_K") if excess_temperature_K is not None else None


    h_value, source_api, library_method, notes = _dispatch_method(
        method_key=method_key,
        mass_flow_kg_s=mass_flow,
        quality=quality,
        tube_diameter_m=tube_diameter_m,
        length_m=length,
        void_fraction=alpha,
        properties=properties,
        excess_temperature_K=excess_temperature,
    )
    if not math.isfinite(h_value) or h_value <= 0.0:
        raise CalculationError(
            code="invalid_two_phase_result",
            message=f"Two-phase correlation returned non-positive h: {h_value}",
        )

    reynolds_liquid = mass_flux * tube_diameter_m / properties.mul
    prandtl_liquid = properties.mul * properties.Cpl / properties.kl
    result: Dict[str, Any] = {
        "two_phase_h_W_m2K": h_value,
        "method_requested": method,
        "method_used": method_key,
        "source_api": source_api,
        "library_method_used": library_method,
        "parameters_used": {
            "mass_flow_kg_s": mass_flow,
            "mass_flux_kg_m2s": mass_flux,
            "quality": quality,
            "tube_diameter_m": tube_diameter_m,
            "saturation_temperature_K": saturation_temperature_K,
            "pressure_Pa": pressure_Pa,
            "length_m": length,
            "void_fraction": alpha,
            "excess_temperature_K": excess_temperature,
        },
        "parameter_basis": {
            "saturation_temperature_K": "supplied",
            "pressure_Pa": "computed saturation pressure",
            "mass_flow_kg_s": "supplied" if mass_flow_kg_s is not None else "computed from supplied mass flux",
            "mass_flux_kg_m2s": "computed from mass flow" if mass_flow_kg_s is not None else "supplied",
            "length_m": "supplied" if length_m is not None else "assumed 1 m (used only by length-dependent correlations)",
            "void_fraction": "supplied" if void_fraction is not None else "homogeneous no-slip assumption",
            "excess_temperature_K": "supplied" if excess_temperature_K is not None else "not used",
        },
        "supplied_mass_flux_kg_m2s": mass_flux_kg_m2s,
        "fluid_properties": asdict(properties),
        "dimensionless_numbers": {
            "Reynolds_liquid": reynolds_liquid,
            "Prandtl_liquid": prandtl_liquid,
        },
        "available_methods": SUPPORTED_METHODS,
        "required_inputs_by_method": required_inputs_by_method(),
        "notes": notes,
    }
    if mass_warning:
        result.setdefault("warnings", []).append(mass_warning)
    return result


def get_saturation_properties(*, fluid_name: str, saturation_temperature_K: float, strict: bool = False,
                              property_source: Optional[str] = None) -> SaturationProperties:
    """One saturation degree of freedom: T is supplied; P is derived by the backend. The named branch
    (``property_source``, else STRICT_PROPERTY_SOURCE, strict or not) answers or refuses (STRICT_PROPERTY_RULE)."""
    named = property_source or STRICT_PROPERTY_SOURCE
    branches = {"CoolProp": _saturation_properties_coolprop, "thermo": _saturation_properties_thermo}
    if named not in branches:
        raise CalculationError(code="invalid_property_source",
                               message=f"property_source must be one of {sorted(branches)}, not {named!r}")
    try:
        return branches[named](fluid_name, saturation_temperature_K)
    except Exception as error:
        raise CalculationError(code="property_source_refused",
            message=(f"the named branch {named} gave no saturated liquid/vapor properties; no other branch "
                     "answers in its place"),
            details={"property_source": named, "reason": str(error),
                     "rule": STRICT_PROPERTY_RULE}) from error


def required_inputs_by_method() -> Dict[str, Any]:
    """Public method-to-input map for schema/tests and caller diagnostics."""
    base = ["mass_flow_kg_s or mass_flux_kg_m2s", "quality", "tube_diameter_m", "fluid_name", "saturation_temperature_K"]
    return {method: base + (["excess_temperature_K"] if method in {"Sun_Mishima", "Liu_Winterton", "Chen_Edelstein"} else [])
            for method in SUPPORTED_METHODS}


def _dispatch_method(
    *,
    method_key: str,
    mass_flow_kg_s: float,
    quality: float,
    tube_diameter_m: float,
    length_m: float,
    void_fraction: float,
    properties: SaturationProperties,
    excess_temperature_K: float,
) -> tuple[float, str, str, list[str]]:
    if method_key == "Shah_1976":
        from ht.condensation import Shah
        _require_properties(properties, ["rhol", "mul", "kl", "Cpl", "Pc", "pressure_Pa"], method_key)
        value = Shah(m=mass_flow_kg_s, x=quality, D=tube_diameter_m,
                     rhol=properties.rhol, mul=properties.mul, kl=properties.kl,
                     Cpl=properties.Cpl, P=properties.pressure_Pa, Pc=properties.Pc)
        return float(value), "ht.condensation.Shah", "Shah", []
    if method_key == "auto":
        return _call_h_two_phase(
            method_key=method_key,
            library_method="Ravipudi_Godbold",
            mass_flow_kg_s=mass_flow_kg_s,
            quality=quality,
            tube_diameter_m=tube_diameter_m,
            length_m=length_m,
            void_fraction=void_fraction,
            properties=properties,
        )
    if method_key == "Cavallini_Smith_Zecchin":
        from ht.condensation import Cavallini_Smith_Zecchin

        _validate_cavallini_smith_zecchin_scope(
            mass_flow_kg_s=mass_flow_kg_s,
            quality=quality,
            tube_diameter_m=tube_diameter_m,
            properties=properties,
        )
        return (
            float(
                Cavallini_Smith_Zecchin(
                    m=mass_flow_kg_s,
                    x=quality,
                    D=tube_diameter_m,
                    rhol=properties.rhol,
                    rhog=properties.rhog,
                    mul=properties.mul,
                    mug=properties.mug,
                    kl=properties.kl,
                    Cpl=properties.Cpl,
                )
            ),
            "ht.condensation.Cavallini_Smith_Zecchin",
            "Cavallini_Smith_Zecchin",
            ["Mach screening uses an ideal-gas sound-speed estimate from vapor cp and molecular weight."],
        )
    if method_key == "Sun_Mishima":
        return _call_boiling_method(
            "Sun_Mishima",
            mass_flow_kg_s,
            quality,
            tube_diameter_m,
            properties,
            excess_temperature_K,
        )
    if method_key == "Liu_Winterton":
        return _call_boiling_method(
            "Liu_Winterton",
            mass_flow_kg_s,
            quality,
            tube_diameter_m,
            properties,
            excess_temperature_K,
        )
    if method_key == "Chen_Edelstein":
        return _call_boiling_method(
            "Chen_Edelstein",
            mass_flow_kg_s,
            quality,
            tube_diameter_m,
            properties,
            excess_temperature_K,
        )
    raise CalculationError(
        code="unsupported_two_phase_method",
        message=f"Unsupported two-phase method '{method_key}'. Supported methods: {SUPPORTED_METHODS}",
        details={"supported_methods": SUPPORTED_METHODS},
    )


def _validate_cavallini_smith_zecchin_scope(
    *,
    mass_flow_kg_s: float,
    quality: float,
    tube_diameter_m: float,
    properties: SaturationProperties,
) -> None:
    area_m2 = math.pi * tube_diameter_m**2 / 4.0
    mass_flux = mass_flow_kg_s / area_m2
    min_flux, max_flux = CAVALLINI_SMITH_ZECCHIN_MASS_FLUX_RANGE_KG_M2S
    vapor_superficial_velocity = mass_flux * quality / properties.rhog
    speed_of_sound = _estimate_vapor_speed_of_sound(properties)
    mach = vapor_superficial_velocity / speed_of_sound if speed_of_sound > 0.0 else math.inf

    if mass_flux < min_flux or mass_flux > max_flux or mach >= CAVALLINI_SMITH_ZECCHIN_MAX_MACH:
        raise CalculationError(
            code="OUT_OF_SCOPE",
            message=(
                "Cavallini_Smith_Zecchin is outside its configured validity envelope: "
                f"mass_flux_kg_m2s must be between {min_flux:g} and {max_flux:g} and "
                f"vapor superficial Mach must be below {CAVALLINI_SMITH_ZECCHIN_MAX_MACH:g}. "
                f"{CAVALLINI_SMITH_ZECCHIN_SCOPE_SOURCE}."
            ),
            field="mass_flux_kg_m2s",
            details={
                "mass_flux_kg_m2s": mass_flux,
                "valid_mass_flux_range_kg_m2s": [min_flux, max_flux],
                "vapor_superficial_velocity_m_s": vapor_superficial_velocity,
                "estimated_vapor_speed_of_sound_m_s": speed_of_sound,
                "vapor_superficial_mach": mach,
                "max_vapor_superficial_mach": CAVALLINI_SMITH_ZECCHIN_MAX_MACH,
                "source": CAVALLINI_SMITH_ZECCHIN_SCOPE_SOURCE,
            },
        )


def _estimate_vapor_speed_of_sound(properties: SaturationProperties) -> float:
    if properties.MW and properties.Cpg:
        mw_kg_mol = properties.MW / 1000.0
        gas_constant = 8.31446261815324 / mw_kg_mol
        cv = properties.Cpg - gas_constant
        if cv > 0.0:
            gamma = properties.Cpg / cv
            return math.sqrt(gamma * gas_constant * properties.saturation_temperature_K)
    raise CalculationError(code="missing_two_phase_properties",
        message="Vapor sound-speed screening requires molecular weight and cp with cp > R; no room-temperature air fallback")


def _call_h_two_phase(
    *,
    method_key: str,
    library_method: str,
    mass_flow_kg_s: float,
    quality: float,
    tube_diameter_m: float,
    length_m: float,
    void_fraction: float,
    properties: SaturationProperties,
) -> tuple[float, str, str, list[str]]:
    from ht.conv_two_phase import h_two_phase

    h_value = h_two_phase(
        m=mass_flow_kg_s,
        x=quality,
        D=tube_diameter_m,
        Cpl=properties.Cpl,
        kl=properties.kl,
        rhol=properties.rhol,
        rhog=properties.rhog,
        mul=properties.mul,
        mu_b=properties.mul,
        mu_w=properties.mul,
        mug=properties.mug,
        L=length_m,
        alpha=void_fraction,
        method=library_method,
    )
    notes = ["Mass flux, when supplied, is converted to ht's required mass flow m before calculation."]
    return float(h_value), "ht.conv_two_phase.h_two_phase", library_method, notes


def _call_boiling_method(
    method_key: str,
    mass_flow_kg_s: float,
    quality: float,
    tube_diameter_m: float,
    properties: SaturationProperties,
    excess_temperature_K: float,
) -> tuple[float, str, str, list[str]]:
    _require_properties(properties, ["Hvap", "sigma"], method_key)
    if method_key == "Sun_Mishima":
        from ht.boiling_flow import Sun_Mishima

        h_value = Sun_Mishima(
            m=mass_flow_kg_s,
            D=tube_diameter_m,
            rhol=properties.rhol,
            rhog=properties.rhog,
            mul=properties.mul,
            kl=properties.kl,
            Hvap=properties.Hvap,
            sigma=properties.sigma,
            Te=excess_temperature_K,
        )
    elif method_key == "Liu_Winterton":
        _require_properties(properties, ["MW", "Pc"], method_key)
        from ht.boiling_flow import Liu_Winterton

        h_value = Liu_Winterton(
            m=mass_flow_kg_s,
            x=quality,
            D=tube_diameter_m,
            rhol=properties.rhol,
            rhog=properties.rhog,
            mul=properties.mul,
            kl=properties.kl,
            Cpl=properties.Cpl,
            MW=properties.MW,
            P=properties.pressure_Pa,
            Pc=properties.Pc,
            Te=excess_temperature_K,
        )
    elif method_key == "Chen_Edelstein":
        _require_properties(properties, ["Hvap", "sigma", "Psat"], method_key)
        from ht.boiling_flow import Chen_Edelstein

        dpsat = _estimate_dpsat(properties.fluid_name, properties.saturation_temperature_K,
                                excess_temperature_K, property_source=properties.source)
        h_value = Chen_Edelstein(
            m=mass_flow_kg_s,
            x=quality,
            D=tube_diameter_m,
            rhol=properties.rhol,
            rhog=properties.rhog,
            mul=properties.mul,
            mug=properties.mug,
            kl=properties.kl,
            Cpl=properties.Cpl,
            Hvap=properties.Hvap,
            sigma=properties.sigma,
            dPsat=dpsat,
            Te=excess_temperature_K,
        )
    else:  # pragma: no cover - guarded by caller
        raise AssertionError(method_key)

    return (
        float(h_value),
        f"ht.boiling_flow.{method_key}",
        method_key,
        ["Wall superheat supplied by caller; not inferred from a design target."],
    )


def _validate_quality(value: float) -> float:
    try:
        quality = float(value)
    except (TypeError, ValueError) as exc:
        raise CalculationError(
            code="invalid_two_phase_input",
            message="quality must be a number between 0 and 1",
            field="quality",
        ) from exc
    if not math.isfinite(quality) or quality <= 0.0 or quality >= 1.0:
        raise CalculationError(
            code="invalid_two_phase_input",
            message="quality must be strictly between 0 and 1 for two-phase correlations; x=0 and x=1 are single-phase boundaries",
            field="quality",
        )
    return quality


def _resolve_mass_flow(
    *,
    mass_flow_kg_s: Optional[float],
    mass_flux_kg_m2s: Optional[float],
    tube_diameter_m: float,
) -> tuple[float, float, Optional[str]]:
    if mass_flow_kg_s is None and mass_flux_kg_m2s is None:
        raise CalculationError(
            code="missing_two_phase_mass_flow",
            message="Either mass_flow_kg_s or mass_flux_kg_m2s is required",
        )
    area_m2 = math.pi * tube_diameter_m**2 / 4.0
    warning = None
    if mass_flow_kg_s is not None:
        mass_flow = validate_positive_finite(mass_flow_kg_s, "mass_flow_kg_s")
        mass_flux = mass_flow / area_m2
        if mass_flux_kg_m2s is not None:
            provided_flux = validate_positive_finite(mass_flux_kg_m2s, "mass_flux_kg_m2s")
            rel_diff = abs(provided_flux - mass_flux) / max(provided_flux, mass_flux)
            if rel_diff > 0.01:
                raise CalculationError(code="inconsistent_mass_flow", message="Supplied mass flow and mass flux disagree by more than the existing 1% consistency threshold")
        return mass_flow, mass_flux, warning

    mass_flux = validate_positive_finite(mass_flux_kg_m2s, "mass_flux_kg_m2s")
    return mass_flux * area_m2, mass_flux, None


def _resolve_void_fraction(
    void_fraction: Optional[float],
    quality: float,
    properties: SaturationProperties,
) -> float:
    if void_fraction is not None:
        alpha = float(void_fraction)
        if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
            raise CalculationError(
                code="invalid_two_phase_input",
                message="void_fraction must be between 0 and 1",
                field="void_fraction",
            )
        return alpha
    if quality <= 0.0:
        return 1e-9
    if quality >= 1.0:
        return 1.0 - 1e-9
    return 1.0 / (1.0 + ((1.0 - quality) / quality) * (properties.rhog / properties.rhol))


def _normalize_method(method: str) -> str:
    raw = str(method or "auto").strip()
    aliases = {
        "auto": "auto",
        "shah": "Shah_1976",
        "shah_1976": "Shah_1976",
        "cavallini_smith_zecchin": "Cavallini_Smith_Zecchin",
        "sun_mishima": "Sun_Mishima",
        "liu_winterton": "Liu_Winterton",
        "chen_edelstein": "Chen_Edelstein",
    }
    normalized = aliases.get(raw.lower())
    if normalized is None:
        raise CalculationError(
            code="unsupported_two_phase_method",
            message=f"Unsupported two-phase method '{method}'. Supported methods: {SUPPORTED_METHODS}",
            details={"supported_methods": SUPPORTED_METHODS},
        )
    return normalized


def _saturation_properties_coolprop(
    fluid_name: str,
    saturation_temperature_K: float,
) -> SaturationProperties:
    import CoolProp.CoolProp as CP

    cp_name = _coolprop_name(fluid_name)
    if not CP.PropsSI("Ttriple", cp_name) <= saturation_temperature_K < CP.PropsSI("Tcrit", cp_name):
        raise ValueError("Saturation temperature must be between triple and critical points")
    pressure_Pa = float(CP.PropsSI("P", "T", saturation_temperature_K, "Q", 0, cp_name))
    props = SaturationProperties(
        fluid_name=fluid_name,
        saturation_temperature_K=saturation_temperature_K,
        pressure_Pa=pressure_Pa,
        source="CoolProp",
        rhol=float(CP.PropsSI("D", "T", saturation_temperature_K, "Q", 0, cp_name)),
        rhog=float(CP.PropsSI("D", "T", saturation_temperature_K, "Q", 1, cp_name)),
        mul=float(CP.PropsSI("V", "T", saturation_temperature_K, "Q", 0, cp_name)),
        mug=float(CP.PropsSI("V", "T", saturation_temperature_K, "Q", 1, cp_name)),
        kl=float(CP.PropsSI("L", "T", saturation_temperature_K, "Q", 0, cp_name)),
        kg=float(CP.PropsSI("L", "T", saturation_temperature_K, "Q", 1, cp_name)),
        Cpl=float(CP.PropsSI("C", "T", saturation_temperature_K, "Q", 0, cp_name)),
        Cpg=float(CP.PropsSI("C", "T", saturation_temperature_K, "Q", 1, cp_name)),
        Hvap=float(
            CP.PropsSI("H", "T", saturation_temperature_K, "Q", 1, cp_name)
            - CP.PropsSI("H", "T", saturation_temperature_K, "Q", 0, cp_name)
        ),
        sigma=float(CP.PropsSI("I", "T", saturation_temperature_K, "Q", 0, cp_name)),
        Psat=float(CP.PropsSI("P", "T", saturation_temperature_K, "Q", 0, cp_name)),
        Pc=float(CP.PropsSI("PCRIT", cp_name)),
        MW=float(CP.PropsSI("M", cp_name) * 1000.0),
    )
    _require_positive_properties(props, ["rhol", "rhog", "mul", "mug", "kl", "Cpl"])
    return props


def _saturation_properties_thermo(
    fluid_name: str,
    saturation_temperature_K: float,
) -> SaturationProperties:
    chem = thermo_chemical(fluid_name, saturation_temperature_K)
    if not chem.Tt <= saturation_temperature_K < chem.Tc:
        raise ValueError("Saturation temperature must be between triple and critical points")
    methods = {"VaporPressure": require_thermo_method(chem, "VaporPressure")}
    pressure_Pa = _require_attr(chem, "Psat")
    chem.calculate(T=saturation_temperature_K, P=pressure_Pa)
    for family in ("VolumeLiquid", "VolumeGas", "ViscosityLiquid", "ViscosityGas",
                   "ThermalConductivityLiquid", "HeatCapacityLiquid"):
        methods[family] = require_thermo_method(chem, family)
    for family, attr in (("ThermalConductivityGas", "kg"), ("HeatCapacityGas", "Cpg"),
                         ("EnthalpyVaporization", "Hvap"), ("SurfaceTension", "sigma")):
        if _optional_attr(chem, attr) is not None:
            methods[family] = require_thermo_method(chem, family)
    props = SaturationProperties(
        fluid_name=fluid_name,
        saturation_temperature_K=saturation_temperature_K,
        pressure_Pa=pressure_Pa,
        source="thermo",
        rhol=_require_attr(chem, "rhol"),
        rhog=_require_attr(chem, "rhog"),
        mul=_require_attr(chem, "mul"),
        mug=_require_attr(chem, "mug"),
        kl=_require_attr(chem, "kl"),
        kg=_optional_attr(chem, "kg"),
        Cpl=_require_attr(chem, "Cpl"),
        Cpg=_optional_attr(chem, "Cpg"),
        Hvap=_optional_attr(chem, "Hvap"),
        sigma=_optional_attr(chem, "sigma"),
        Psat=_optional_attr(chem, "Psat"),
        Pc=_optional_attr(chem, "Pc"),
        MW=_optional_attr(chem, "MW"),
        property_methods=methods,
    )
    _require_positive_properties(props, ["rhol", "rhog", "mul", "mug", "kl", "Cpl"])
    return props


def _coolprop_name(fluid_name: str) -> str:
    names = {"water": "Water", "h2o": "Water", "r134a": "R134a", "ammonia": "Ammonia"}
    return names.get(fluid_name.lower().replace(" ", ""), fluid_name)


def _require_attr(obj: Any, name: str) -> float:
    value = getattr(obj, name, None)
    return validate_positive_finite(value, name)


def _optional_attr(obj: Any, name: str) -> Optional[float]:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _require_positive_properties(properties: SaturationProperties, names: list[str]) -> None:
    for name in names:
        validate_positive_finite(getattr(properties, name), name)


def _require_properties(properties: SaturationProperties, names: list[str], method_key: str) -> None:
    missing = [name for name in names if getattr(properties, name) is None]
    if missing:
        raise CalculationError(
            code="missing_two_phase_properties",
            message=f"{method_key} requires saturation properties that are unavailable",
            details={"missing": missing, "property_source": properties.source},
        )


def _estimate_dpsat(fluid_name: str, saturation_temperature_K: float, excess_temperature_K: float,
                    *, property_source: str) -> float:
    try:
        pressures = []
        for temperature in (saturation_temperature_K, saturation_temperature_K + excess_temperature_K):
            if property_source == "thermo":
                chem = thermo_chemical(fluid_name, temperature)
                if not chem.Tt <= temperature < chem.Tc:
                    raise ValueError("vapor-pressure temperature must be between triple and critical points")
                require_thermo_method(chem, "VaporPressure")
                pressures.append(_require_attr(chem, "Psat"))
            elif property_source == "CoolProp":
                import CoolProp.CoolProp as CP
                name = _coolprop_name(fluid_name)
                if not CP.PropsSI("Ttriple", name) <= temperature < CP.PropsSI("Tcrit", name):
                    raise ValueError("vapor-pressure temperature must be between triple and critical points")
                pressures.append(CP.PropsSI("P", "T", temperature, "Q", 0, name))
            else:
                raise ValueError(f"Unsupported vapor-pressure backend: {property_source}")
        psat, psat_wall = pressures
        return validate_positive_finite(float(psat_wall - psat), "dPsat")
    except Exception as exc:
        raise CalculationError(
            code="missing_two_phase_properties",
            message=f"Chen_Edelstein requires a qualified vapor-pressure increment from the saturation backend: {exc}",
        ) from exc
