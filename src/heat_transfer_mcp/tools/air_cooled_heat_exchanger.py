"""Rate a vendor's air-cooled heat exchanger (ACHE / fin-fan) against site conditions.

Thin adapter. Every correlation, threshold and balance lives in ``engineering_utils.thermal``
so that a number quoted back to a vendor can be regenerated later by the same tool call
rather than reconstructed from a reviewer's working. This module only marshals payloads,
resolves air properties, and shapes the response envelope.

The tool answers one question: given the bundle geometry the vendor disclosed, what duty
does it actually deliver at our air-on temperature, and how does that compare with the duty
the service requires? It does not size a bundle from scratch.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from puran_mcp_support.engineering.thermal.air_cooled_exchanger import AirState, BundleGeometry, InsufficientVendorData, achievable_duty, fan_power, mtd_correction, overall_U_extended_basis, rate_air_side
from puran_mcp_support.engineering.thermal.api661_checks import run_checks
from puran_mcp_support.engineering.thermal.wet_gas_cooling import DEFAULT_ADP_C, TrainStage, cool_wet_gas, cool_wet_gas_train
from heat_transfer_mcp.tools.fluid_properties import PropertyLookupRefused, calculation_properties, property_refusal

logger = logging.getLogger("heat-transfer-mcp.air_cooled_heat_exchanger")

#: Aluminium 1100, near-universal for ACHE fins. Assumed when the vendor does not state it,
#: and reported as an assumption in the response rather than folded in silently.
DEFAULT_FIN_CONDUCTIVITY = 205.0


def _air_properties(T_C: float, pressure_Pa: float, property_source: str) -> Dict[str, float]:
    """Air properties at the design air-on temperature, from the named branch (CALCULATION_PROPERTY_RULE)."""
    return calculation_properties("air", T_C + 273.15, pressure_Pa, property_source)


def rate_air_cooled_heat_exchanger(
    bundle: Optional[Dict[str, Any]] = None,
    air_inlet_temp_C: Optional[float] = None,
    air_outlet_temp_C: Optional[float] = None,
    air_mass_flow_kg_s: Optional[float] = None,
    air_face_velocity_m_s: Optional[float] = None,
    site_altitude_m: float = 0.0,
    process: Optional[Dict[str, Any]] = None,
    fan: Optional[Dict[str, Any]] = None,
    tube_side_h_W_m2K: Optional[float] = None,
    tube_inside_area_m2: Optional[float] = None,
    fouling_air_m2K_W: Optional[float] = None,
    fouling_tube_m2K_W: Optional[float] = None,
    api661_invoked: bool = True,
    purchaser_agreed_single_fan: bool = False,
    property_source: str = "CoolProp",
) -> str:
    """Rate an ACHE bundle at site air-on and compare with the required duty; air properties from
    ``property_source`` or the call refuses (CALCULATION_PROPERTY_RULE)."""
    args: Dict[str, Any] = {
        "bundle": bundle,
        "air_inlet_temp_C": air_inlet_temp_C,
        "air_outlet_temp_C": air_outlet_temp_C,
        "air_mass_flow_kg_s": air_mass_flow_kg_s,
        "air_face_velocity_m_s": air_face_velocity_m_s,
        "site_altitude_m": site_altitude_m,
        "process": process,
        "fan": fan,
        "tube_side_h_W_m2K": tube_side_h_W_m2K,
        "tube_inside_area_m2": tube_inside_area_m2,
        "fouling_air_m2K_W": fouling_air_m2K_W,
        "fouling_tube_m2K_W": fouling_tube_m2K_W,
        "api661_invoked": api661_invoked,
        "purchaser_agreed_single_fan": purchaser_agreed_single_fan,
    }
    try:
        geom_in = args.get("bundle") or {}
        geometry = BundleGeometry(
            tube_rows=geom_in.get("tube_rows"),
            tube_passes=geom_in.get("tube_passes"),
            tubes_per_row=geom_in.get("tubes_per_row"),
            tube_length_m=geom_in.get("tube_length_m"),
            tube_od_m=geom_in.get("tube_od_m"),
            fin_thickness_m=geom_in.get("fin_thickness_m"),
            fin_density_per_m=geom_in.get("fin_density_per_m"),
            fin_height_m=geom_in.get("fin_height_m"),
            pitch_normal_m=geom_in.get("pitch_normal_m"),
            pitch_parallel_m=geom_in.get("pitch_parallel_m"),
            tube_thickness_m=geom_in.get("tube_thickness_m"),
            parallel_bays=geom_in.get("parallel_bays") or 1,
            bundles_per_bay=geom_in.get("bundles_per_bay") or 1,
            fans_per_bay=geom_in.get("fans_per_bay") or 1,
            fan_diameter_m=geom_in.get("fan_diameter_m"),
            fin_conductivity_W_mK=geom_in.get("fin_conductivity_W_mK") or DEFAULT_FIN_CONDUCTIVITY,
        )

        air_on_C = args.get("air_inlet_temp_C")
        if air_on_C is None:
            return json.dumps(
                {
                    "code": "insufficient_process_data",
                    "message": (
                        "air_inlet_temp_C is required — the site design air-on "
                        "temperature governs the entire rating."
                    ),
                    "details": {"missing": ["air_inlet_temp_C"]},
                }
            )

        altitude_m = float(args.get("site_altitude_m", 0.0))
        from fluids.atmosphere import ATMOSPHERE_1976
        site_pressure_Pa = ATMOSPHERE_1976(Z=altitude_m).P
        props = _air_properties(air_on_C, site_pressure_Pa, property_source)
        density = float(props["density"])

        air = AirState(
            T_in_C=float(air_on_C),
            density_kg_m3=density,
            cp_J_kgK=float(props["specific_heat_cp"]),
            viscosity_Pa_s=float(props["dynamic_viscosity"]),
            conductivity_W_mK=float(props["thermal_conductivity"]),
            mass_flow_kg_s=args.get("air_mass_flow_kg_s"),
            face_velocity_m_s=args.get("air_face_velocity_m_s"),
        )

        air_side = rate_air_side(geometry, air)
        warnings: List[str] = list(air_side.warnings)

        # ---- required process duty -------------------------------------------------
        process = args.get("process") or {}
        duty: Optional[Dict[str, Any]] = None
        if process:
            stages_in = process.get("train_stages")
            if stages_in:
                stages = [
                    TrainStage(
                        label=s.get("label", f"stage {i + 1}"),
                        gas_nm3h_wet=s["gas_nm3h_wet"],
                        P_bara=s["P_bara"],
                        T_in_C=s["T_in_C"],
                        T_out_C=s["T_out_C"],
                        knockout=s.get("knockout", True),
                    )
                    for i, s in enumerate(stages_in)
                ]
                duty = cool_wet_gas_train(
                    stages,
                    adp_C=process.get("inlet_adp_C", DEFAULT_ADP_C),
                    water_mole_frac_in=process.get("water_mole_frac_in"),
                    dry_composition=process.get("dry_composition"),
                )
                warnings.extend(duty.get("warnings", []))
            elif process.get("gas_nm3h_wet"):
                res = cool_wet_gas(
                    process["gas_nm3h_wet"],
                    T_in_C=process["T_in_C"],
                    T_out_C=process["T_out_C"],
                    P_bara=process["P_bara"],
                    adp_C=process.get("inlet_adp_C", DEFAULT_ADP_C),
                    water_mole_frac_in=process.get("water_mole_frac_in"),
                    dry_composition=process.get("dry_composition"),
                    label=process.get("label", "process cooler"),
                )
                duty = res.as_dict()
                warnings.extend(res.warnings)

        # ---- MTD and achievable duty ------------------------------------------------
        # A train describes several coolers but the bundle being rated is one of them, so
        # the MTD needs that stage's own terminal temperatures. `rated_stage_index` names
        # it; the last stage is the default because the high-pressure aftercooler is
        # usually the one under review.
        rated_T_in = process.get("T_in_C")
        rated_T_out = process.get("T_out_C")
        stages_in = process.get("train_stages") if process else None
        if stages_in:
            idx = process.get("rated_stage_index", len(stages_in) - 1)
            try:
                rated_stage = stages_in[idx]
                rated_T_in = rated_stage["T_in_C"]
                rated_T_out = rated_stage["T_out_C"]
            except (IndexError, KeyError, TypeError):
                warnings.append(
                    f"rated_stage_index={idx!r} does not identify a stage in the supplied "
                    "train, so the MTD correction was skipped."
                )
                rated_T_in = rated_T_out = None

        thermal: Optional[Dict[str, Any]] = None
        air_out_C = args.get("air_outlet_temp_C")
        h_tube = args.get("tube_side_h_W_m2K")
        if air_out_C is not None and rated_T_in is not None and rated_T_out is not None:
            ft = mtd_correction(
                T_hot_in_C=rated_T_in,
                T_hot_out_C=rated_T_out,
                T_air_in_C=air_on_C,
                T_air_out_C=air_out_C,
                tube_passes=geometry.tube_passes,
                tube_rows=geometry.tube_rows,
            )
            warnings.extend(ft["warnings"])
            thermal = {"Ft": ft["Ft"], "Ft_support_tier": ft["support_tier"]}

            a_inside = args.get("tube_inside_area_m2")
            if h_tube and a_inside:
                u_ext = overall_U_extended_basis(
                    h_air_extended_basis=air_side.h_air_effective_extended_basis,
                    h_tube_W_m2K=float(h_tube),
                    area_extended_m2=air_side.area_extended_m2,
                    area_tube_inside_m2=float(a_inside),
                    fouling_air_m2K_W=args.get("fouling_air_m2K_W") or 0.0,
                    fouling_tube_m2K_W=args.get("fouling_tube_m2K_W") or 0.0,
                )
                # Required duty for the rated stage, so the margin is against the right one.
                q_required = None
                if duty:
                    if "stages" in duty:
                        stage_idx = process.get("rated_stage_index", len(duty["stages"]) - 1)
                        try:
                            q_required = duty["stages"][stage_idx]["total_kW"]
                        except (IndexError, KeyError, TypeError):
                            q_required = None
                    else:
                        q_required = duty.get("total_kW")

                delivered = achievable_duty(
                    U_extended_basis_W_m2K=u_ext,
                    area_extended_m2=air_side.area_extended_m2,
                    T_hot_in_C=rated_T_in,
                    T_hot_out_C=rated_T_out,
                    T_air_in_C=air_on_C,
                    T_air_out_C=air_out_C,
                    tube_passes=geometry.tube_passes,
                    tube_rows=geometry.tube_rows,
                    h_air_extended_basis=air_side.h_air_effective_extended_basis,
                    Q_required_kW=q_required,
                )
                thermal.update(delivered.as_dict())
                warnings.extend(w for w in delivered.warnings if w not in warnings)
            else:
                missing = [
                    name
                    for name, val in (
                        ("tube_side_h_W_m2K", h_tube),
                        ("tube_inside_area_m2", a_inside),
                    )
                    if not val
                ]
                warnings.append(
                    f"{' and '.join(missing)} not supplied, so overall U and the achievable "
                    "duty were not computed — this rating cannot say whether the bundle "
                    "makes its duty. The air-side result and geometry checks stand on "
                    "their own."
                )

        # ---- fan power ---------------------------------------------------------------
        fan: Optional[Dict[str, Any]] = None
        fan_in = args.get("fan") or {}
        # Supplying a fan block is a request for fan power. If it cannot be answered, that
        # is an error rather than a warning — a caller who only wants the air-side rating
        # simply omits the block.
        if fan_in and fan_in.get("fan_static_efficiency") is None:
            raise InsufficientVendorData(
                [
                    name
                    for name in ("fan_static_efficiency", "drive_efficiency", "motor_efficiency")
                    if fan_in.get(name) is None
                ],
                "a fan block was supplied, so fan power was requested",
            )
        if fan_in.get("fan_static_efficiency") is not None:
            fp = fan_power(
                air_volumetric_flow_m3_s=air_side.air_volumetric_flow_m3_s,
                bundle_dP_Pa=air_side.bundle_dP_Pa,
                fan_static_efficiency=fan_in.get("fan_static_efficiency"),
                drive_efficiency=fan_in.get("drive_efficiency"),
                motor_efficiency=fan_in.get("motor_efficiency"),
                external_dP_Pa=fan_in.get("external_dP_Pa", 0.0),
                vendor_motor_kW=fan_in.get("vendor_motor_kW"),
            )
            fan = fp.as_dict()
            warnings.extend(fp.warnings)
        else:
            warnings.append(
                "Fan efficiencies not supplied, so shaft and absorbed electrical power "
                "were not computed. A vendor motor rating cannot be checked without "
                "fan static, drive and motor efficiencies."
            )

        # ---- standards checks ---------------------------------------------------------
        conformance = run_checks(
            fan_diameter_m=geometry.fan_diameter_m,
            bundle_face_area_m2=air_side.area_face_m2,
            fans_per_bay=geometry.fans_per_bay,
            parallel_bays=geometry.parallel_bays,
            face_velocity_m_s=air_side.face_velocity_m_s,
            air_side_fouling_m2K_W=args.get("fouling_air_m2K_W"),
            tube_side_fouling_m2K_W=args.get("fouling_tube_m2K_W"),
            purchaser_agreed_single_fan=bool(args.get("purchaser_agreed_single_fan", False)),
            api661_invoked=bool(args.get("api661_invoked", True)),
            fin_density_per_m=geometry.fin_density_per_m,
            ft_support_tier=(thermal or {}).get("Ft_support_tier"),
            correlation_spread_pct=air_side.correlation_spread_pct,
        ).as_dict()

        # The runtime wraps whatever is returned into the `result` envelope, so the
        # payload is returned unwrapped here.
        return json.dumps(
            {
                "air_side": air_side.as_dict(),
                "required_duty": duty,
                "thermal": thermal,
                "fan": fan,
                "conformance": conformance,
                "site": {
                    "air_inlet_temp_C": air_on_C,
                    "altitude_m": altitude_m,
                    "air_density_kg_m3": density,
                    "air_data_source": props.get("data_source"),
                    "air_pressure_Pa": site_pressure_Pa,
                    "pressure_basis": "Assumed standard atmosphere at site altitude; supplied air temperature is retained",
                },
                # What this rating did and did not establish, so an incomplete answer
                # cannot be mistaken for a verdict.
                "rating_completeness": {
                    "air_side_rated": True,
                    "required_duty_computed": duty is not None,
                    "achievable_duty_computed": bool(
                        thermal and thermal.get("Q_achievable_kW") is not None
                    ),
                    "fan_power_computed": fan is not None,
                    "conformance_checked": True,
                    "verdict_available": bool(thermal and thermal.get("verdict")),
                },
                # Values the caller did not supply and the tool assumed.
                "assumptions": {
                    "fin_conductivity_W_mK": geometry.fin_conductivity_W_mK,
                    "fin_conductivity_supplied": geom_in.get("fin_conductivity_W_mK") is not None,
                    "parallel_bays": geometry.parallel_bays,
                    "bundles_per_bay": geometry.bundles_per_bay,
                    "bays_supplied": geom_in.get("parallel_bays") is not None,
                    "fans_per_bay_supplied": geom_in.get("fans_per_bay") is not None,
                    "fouling_air_m2K_W": args.get("fouling_air_m2K_W"),
                    "inlet_adp_C": (process or {}).get("inlet_adp_C") or DEFAULT_ADP_C,
                    "inlet_adp_supplied": (process or {}).get("inlet_adp_C") is not None,
                },
                "warnings": warnings,
            },
            default=str,
        )

    except InsufficientVendorData as exc:
        return json.dumps(
            {
                "code": "insufficient_vendor_data",
                "message": str(exc),
                "details": {"missing": exc.missing},
            }
        )
    except PropertyLookupRefused as refused:
        return property_refusal(refused, "Air properties refused")
    except Exception as exc:  # pragma: no cover - defensive envelope
        logger.exception("ACHE rating failed")
        return json.dumps({"code": "calculation_error", "message": str(exc)})
