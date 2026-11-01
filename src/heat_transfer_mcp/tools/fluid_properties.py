"""
Fluid properties tool to retrieve thermophysical properties of fluids.

This module provides functionality to calculate fluid properties using the HT library.
"""

import json
import logging
import math
from functools import lru_cache
from typing import Optional

from heat_transfer_mcp.property_validation import require_thermo_method, thermo_chemical
from heat_transfer_mcp.utils.constants import DEG_C_to_K
from heat_transfer_mcp.utils.import_helpers import COOLPROP_AVAILABLE, THERMO_AVAILABLE, coolprop_fluid_properties

logger = logging.getLogger("heat-transfer-mcp.fluid_properties")

#: The property branches, by the name a caller gives (``property_source``) -> the ``data_source`` its answer carries.
PROPERTY_SOURCES = {"CoolProp": "CoolProp", "thermo": "thermo_library"}
#: The branch ``strict=True`` names when the caller names none: the reference EOS the server declares (pyproject.toml,
#: at the house mirror's version).
STRICT_PROPERTY_SOURCE = "CoolProp"
STRICT_PROPERTY_RULING_ID = 'PUBLIC-RULE-961ea8823bc5'
UNNAMED_PROPERTY_RULING_ID = 'PUBLIC-RULE-580dcf227f5b'
STRICT_PROPERTY_RULE = (
    "get_fluid_properties: the named branch (property_source, else STRICT_PROPERTY_SOURCE, strict or not) answers or "
    "the call refuses with its reason; no other branch answers in its place, and data_source names the branch "
    f"({STRICT_PROPERTY_RULING_ID} OQ-C114-1; no unnamed fallback to thermo, {UNNAMED_PROPERTY_RULING_ID} OQ-C126-2)")
CALCULATION_PROPERTY_RULING_ID = 'PUBLIC-RULE-62db1b064c93'
CALCULATION_PROPERTY_RULE = (
    "A tool that looks up fluid properties takes property_source (default STRICT_PROPERTY_SOURCE) and passes it to "
    "every lookup of its calculation: each lookup answers from that branch or the calculation refuses with the "
    "branch's reason; no lookup falls back to another branch or to another value, and the result names the branch "
    f"in data_source ({CALCULATION_PROPERTY_RULING_ID} OQ-C116-2)")


class PropertyLookupRefused(ValueError):
    """CALCULATION_PROPERTY_RULE: a lookup inside a calculation was refused; ``payload`` is the lookup's own."""

    def __init__(self, payload: dict):
        super().__init__(payload.get("error") or payload.get("message") or str(payload))
        self.payload = payload


def calculation_property_source(property_source: Optional[str]) -> str:
    """CALCULATION_PROPERTY_RULE: the one branch every lookup of a calculation reads; unnamed is the declared one."""
    branch = property_source or STRICT_PROPERTY_SOURCE
    if branch not in PROPERTY_SOURCES:
        raise ValueError(f"property_source must be one of {sorted(PROPERTY_SOURCES)}, not {property_source!r}")
    return branch


def calculation_properties(fluid_name: str, temperature: float, pressure: float, property_source: Optional[str]) -> dict:
    """CALCULATION_PROPERTY_RULE: one lookup of a calculation from its named branch, or PropertyLookupRefused."""
    props = json.loads(get_fluid_properties(fluid_name, temperature, pressure,
                                            property_source=calculation_property_source(property_source)))
    if "error" in props or ("code" in props and "message" in props):
        raise PropertyLookupRefused(props)
    return props


def property_refusal(refused: PropertyLookupRefused, context: str) -> str:
    """The calculation's refusal: the lookup's reason under ``context``, its code and details kept."""
    payload = refused.payload
    return json.dumps({"error": f"{context}: {refused}", **{k: payload[k] for k in ("code", "details") if k in payload}})


def _refused(branch: str, fluid_name: str, T: float, P: float, reason: Exception) -> str:
    """STRICT_PROPERTY_RULE: the named branch's refusal, never another branch's answer."""
    hint = " Name property_source='thermo' to answer from thermo." if branch == "CoolProp" else ""
    return json.dumps(
        {
            "error": (
                f"Property lookup refused: the named branch {branch} gave no qualified answer for {fluid_name!r} at "
                f"T={T} K, P={P} Pa ({reason}); no other branch answers in its place.{hint}"
            ),
            "code": "property_source_refused",
            "details": {"property_source": branch, "reason": str(reason), "rule": STRICT_PROPERTY_RULE},
        }
    )


@lru_cache(maxsize=128)
def _cached_get_properties_thermo(fluid_name: str, temperature: float, pressure: float) -> Optional[dict]:
    """Cached property lookup using thermo's own methods (THERMO_OWN_METHODS_RULE) for pure fluids and air.

    - Uses thermo_chemical for pure components
    - Uses DryAirLemmon for air
    """
    if not THERMO_AVAILABLE:
        return None
    try:
        # Handle common mixture aliases first
        fname = fluid_name.strip().lower()
        if fname == "air":
            from thermo.phases.air_phase import DryAirLemmon
            from chemicals.air import (lemmon2000_air_MW, lemmon2000_air_T_max, lemmon2000_air_P_max,
                                       lemmon2000_air_T_reducing, lemmon2000_air_P_dew)
            if not 60 <= temperature <= lemmon2000_air_T_max or pressure > lemmon2000_air_P_max:
                raise ValueError("DryAirLemmon outside its 60–2000 K, <= 2 GPa EOS envelope")
            if temperature < lemmon2000_air_T_reducing and pressure >= lemmon2000_air_P_dew(temperature):
                raise ValueError("Cryogenic two-phase/liquid air requires a phase-qualified mixture model")
            air = DryAirLemmon(T=temperature, P=pressure)
            mw_kg_mol = lemmon2000_air_MW / 1000
            rho = float(air.rho()) * mw_kg_mol
            cp = float(air.Cp()) / mw_kg_mol
            k, mu = float(air.k()), float(air.mu())
            return {"rho": rho, "Cp": cp, "k": k, "mu": mu, "nu": mu/rho, "Pr": mu*cp/k,
                    "phase": "g", "beta": air.isobaric_expansion(),
                    "property_methods": {"EOS": "Lemmon 2000 dry air; gas root", "molecular_weight_g_mol": lemmon2000_air_MW}}

        # Otherwise treat as pure chemical
        chem = thermo_chemical(fluid_name, temperature, pressure)
        # Only liquid or gas phases are supported for now
        if hasattr(chem, "phase") and chem.phase not in ["l", "g"]:
            raise ValueError(f"Unsupported phase '{chem.phase}' for {fluid_name} at T={temperature} K, P={pressure} Pa")
        phase_suffix = "Liquid" if chem.phase == "l" else "Gas"
        methods = {}
        for family in ("HeatCapacity", "Volume", "Viscosity", "ThermalConductivity"):
            methods[family] = require_thermo_method(chem, family + phase_suffix)
        return {
            "rho": float(chem.rho) if getattr(chem, "rho", None) is not None else None,
            "Cp": float(chem.Cp) if getattr(chem, "Cp", None) is not None else None,
            "k": float(chem.k) if getattr(chem, "k", None) is not None else None,
            "mu": float(chem.mu) if getattr(chem, "mu", None) is not None else None,
            "nu": float(chem.nu) if getattr(chem, "nu", None) is not None else None,
            "Pr": float(chem.Pr) if getattr(chem, "Pr", None) is not None else None,
            "beta": chem.isobaric_expansion,
            "phase": chem.phase,
            "property_methods": methods,
        }
    except Exception as exc:
        raise ValueError(f"Property lookup refused for {fluid_name}: {exc}") from exc


_LUBE_OIL_HINTS = ("iso vg", "lube", "lubricat", "gear oil", "compressor oil", "mineral oil")


def _looks_like_lube_oil(fluid_name: str) -> bool:
    """Whether a fluid name denotes a lubricant rather than a pure component."""
    name = (fluid_name or "").strip().lower()
    if name in {"oil", "lube oil", "lubricating oil"}:
        return True
    return any(hint in name for hint in _LUBE_OIL_HINTS)


def _lube_oil_properties(
    *,
    fluid_name: str,
    temperature_K: float,
    nu_40_cSt: Optional[float],
    nu_100_cSt: Optional[float],
    density_15_kg_m3: Optional[float],
    provenance: Optional[str],
) -> str:
    """Thin adapter onto the shared petroleum-fraction correlations."""
    from puran_mcp_support.engineering.thermophysical.lube_oil import InsufficientOilData, LubeOil

    try:
        oil = LubeOil(
            nu_40_cSt=nu_40_cSt or 0.0,
            nu_100_cSt=nu_100_cSt or 0.0,
            density_15_kg_m3=density_15_kg_m3 or 0.0,
            designation=fluid_name,
            provenance=provenance or "",
        )
    except InsufficientOilData as exc:
        return json.dumps(
            {
                "code": "insufficient_oil_data",
                "message": str(exc),
                "details": {"missing": exc.missing},
            }
        )

    T_C = temperature_K - 273.15
    props = oil.properties_at(T_C)
    return json.dumps(
        {
            "fluid_name": fluid_name,
            "temperature_k": temperature_K,
            "temperature_c": T_C,
            "density": props["density_kg_m3"],
            "specific_heat_cp": props["specific_heat_J_kgK"],
            "thermal_conductivity": props["thermal_conductivity_W_mK"],
            "dynamic_viscosity": props["dynamic_viscosity_Pa_s"],
            "kinematic_viscosity": props["kinematic_viscosity_cSt"] * 1e-6,
            "kinematic_viscosity_cSt": props["kinematic_viscosity_cSt"],
            "prandtl_number": props["prandtl"],
            "viscosity_index": props["viscosity_index"],
            "phase": "l",
            "data_source": 'puran_mcp_support.engineering.thermophysical.lube_oil',
            "accuracy": "petroleum-fraction correlations, exact at the supplied anchors",
            "correlations": props["correlations"],
            "provenance": props["provenance"],
            "warnings": props["warnings"],
        },
        default=str,
    )


def get_fluid_properties(
    fluid_name: str,
    temperature: float,
    pressure: float = 101325.0,
    strict: bool = False,
    lube_oil_nu_40_cSt: Optional[float] = None,
    lube_oil_nu_100_cSt: Optional[float] = None,
    lube_oil_density_15_kg_m3: Optional[float] = None,
    lube_oil_provenance: Optional[str] = None,
    property_source: Optional[str] = None,
) -> str:
    """Retrieves thermophysical properties of common fluids at specified conditions.

    Args:
        fluid_name: Name of the fluid (e.g., 'water', 'air', 'Ethanol')
        temperature: Temperature in Kelvin (K)
        pressure: Pressure in Pascals (Pa)
        strict: Kept for callers; every call answers from one branch (``property_source``, else
            STRICT_PROPERTY_SOURCE) or refuses, never a fallback (STRICT_PROPERTY_RULE)
        lube_oil_nu_40_cSt, lube_oil_nu_100_cSt, lube_oil_density_15_kg_m3:
            Datasheet anchors for a lubricating oil. Supplying them routes the request to
            the petroleum-fraction correlations, which is the only way to get properties
            for an oil — no pure-component database carries them. All three are required
            together: a grade name alone fixes only the 40 °C viscosity, and the
            temperature slope is what an oil cooler actually runs on.
        lube_oil_provenance: The datasheet the anchors came from, recorded with the result.
        property_source: The branch that answers, 'CoolProp' or 'thermo' (PROPERTY_SOURCES), else
            STRICT_PROPERTY_SOURCE; it answers or the call refuses. A fluid CoolProp lacks needs 'thermo'.

    Returns:
        JSON string with detailed fluid properties
    """
    oil_anchors = (lube_oil_nu_40_cSt, lube_oil_nu_100_cSt, lube_oil_density_15_kg_m3)
    if any(a is not None for a in oil_anchors):
        return _lube_oil_properties(
            fluid_name=fluid_name,
            temperature_K=temperature,
            nu_40_cSt=lube_oil_nu_40_cSt,
            nu_100_cSt=lube_oil_nu_100_cSt,
            density_15_kg_m3=lube_oil_density_15_kg_m3,
            provenance=lube_oil_provenance,
        )
    if _looks_like_lube_oil(fluid_name):
        return json.dumps(
            {
                "code": "insufficient_oil_data",
                "message": (
                    f"{fluid_name!r} is a lubricating oil, whose properties are not in any "
                    "pure-component database. Supply lube_oil_nu_40_cSt, "
                    "lube_oil_nu_100_cSt and lube_oil_density_15_kg_m3 from the lubricant "
                    "datasheet. A grade designation alone does not determine the "
                    "temperature slope."
                ),
                "details": {
                    "missing": [
                        "lube_oil_nu_40_cSt",
                        "lube_oil_nu_100_cSt",
                        "lube_oil_density_15_kg_m3",
                    ]
                },
            }
        )
    try:
        result = {}

        # Basic input validation
        try:
            T = float(temperature)
        except (TypeError, ValueError):
            return json.dumps({"error": "Temperature must be a numeric value in Kelvin."})
        try:
            P = float(pressure)
        except (TypeError, ValueError):
            return json.dumps({"error": "Pressure must be a numeric value in Pascals."})
        if T <= 0.0:
            return json.dumps({"error": "Temperature must be above 0 K (absolute zero)."})
        if T > 1.0e4:
            return json.dumps({"error": "Temperature is unrealistically high. Please provide T < 10000 K."})
        if not math.isfinite(T) or not math.isfinite(P):
            return json.dumps({"error": "Temperature and pressure must be finite real numbers."})
        if P <= 0.0:
            return json.dumps({"error": "Pressure must be positive."})

        if property_source is not None and property_source not in PROPERTY_SOURCES:
            return json.dumps({"error": f"property_source must be one of {sorted(PROPERTY_SOURCES)}, not {property_source!r}"})
        branch = property_source or STRICT_PROPERTY_SOURCE   # STRICT_PROPERTY_RULE

        if branch == "CoolProp":
            try:
                if not COOLPROP_AVAILABLE:
                    raise ImportError("CoolProp is not installed in this server's environment (pyproject.toml declares it)")
                coolprop_props = coolprop_fluid_properties(fluid_name, T, P)
                result = {
                    "fluid_name": fluid_name,
                    "temperature_k": T,
                    "pressure_pa": P,
                    "density": coolprop_props.get("density"),
                    "specific_heat_cp": coolprop_props.get("specific_heat_cp"),
                    "thermal_conductivity": coolprop_props.get("thermal_conductivity"),
                    "dynamic_viscosity": coolprop_props.get("dynamic_viscosity"),
                    "kinematic_viscosity": coolprop_props.get("kinematic_viscosity"),
                    "prandtl_number": coolprop_props.get("prandtl_number"),
                    "isobaric_expansion_per_K": coolprop_props.get("isobaric_expansion_per_K"),
                    "phase": coolprop_props.get("phase", "unknown"),
                    "data_source": "CoolProp",
                    "accuracy": "high (reference EOS)",
                }
                for name in ("density", "specific_heat_cp", "thermal_conductivity", "dynamic_viscosity"):
                    value = result.get(name)
                    if value is None or not math.isfinite(value) or value <= 0:
                        raise ValueError(f"CoolProp did not supply positive finite {name}")
                # Add temperature in Celsius for convenience
                result["temperature_c"] = round(T - DEG_C_to_K, 2)
                return json.dumps(result)
            except Exception as e:
                return _refused(branch, fluid_name, T, P, e)

        if not THERMO_AVAILABLE:
            return _refused(branch, fluid_name, T, P, ImportError("thermo is not installed in this server's environment"))
        try:
            logger.info(f"Attempting to get properties for {fluid_name} using thermo")
            props = _cached_get_properties_thermo(fluid_name, T, P)
            if props is not None:
                # Ensure all derived properties are present
                rho = props.get("rho")
                Cp = props.get("Cp")
                k = props.get("k")
                mu = props.get("mu")
                nu = (
                    props.get("nu")
                    if props.get("nu") is not None and props.get("nu") == props.get("nu")
                    else (mu / rho if (mu and rho) else None)
                )
                Pr = (
                    props.get("Pr")
                    if props.get("Pr") is not None and props.get("Pr") == props.get("Pr")
                    else (mu * Cp / k if (mu and Cp and k) else None)
                )

                result = {
                    "fluid_name": fluid_name,
                    "temperature_k": T,
                    "pressure_pa": P,
                    "density": rho,
                    "specific_heat_cp": Cp,
                    "thermal_conductivity": k,
                    "dynamic_viscosity": mu,
                    "kinematic_viscosity": nu,
                    "prandtl_number": Pr,
                    "isobaric_expansion_per_K": props.get("beta"),
                    "phase": props.get("phase", "unknown"),
                    "data_source": "thermo_library",
                    "accuracy": "correlation-dependent; no blanket accuracy claim",
                    "property_methods": props.get("property_methods", {}),
                }
            else:
                raise ValueError(f"No qualified single-phase properties for '{fluid_name}' at T={T} K, P={P} Pa")
        except Exception as e:
            return _refused(branch, fluid_name, T, P, e)

        for name in ("density", "specific_heat_cp", "thermal_conductivity", "dynamic_viscosity"):
            value = result.get(name)
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Property backend did not supply positive finite {name}")
        # Clean up result - remove None values
        clean_result = {k: v for k, v in result.items() if v is not None}

        # Add a note about the temperature in Celsius for convenience
        if "temperature_k" in clean_result:
            clean_result["temperature_c"] = round(clean_result["temperature_k"] - DEG_C_to_K, 2)

        return json.dumps(clean_result)

    except Exception as e:
        logger.error(f"Unexpected error in get_fluid_properties: {e}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
