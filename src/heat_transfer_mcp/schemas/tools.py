"""Input/output models for the registered heat-transfer tools."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Type

from puran_mcp_support.engineering.pydantic_base import StrictFiniteBaseModel
from pydantic import BaseModel, Field, create_model, model_validator

from heat_transfer_mcp.schemas.common import ToolOutput, ToolSchema
from heat_transfer_mcp.units import QuantityField

FieldDef = Tuple[Any, Any]


class _InputBase(StrictFiniteBaseModel):
    @model_validator(mode="before")
    @classmethod
    def convert_quantities(cls, value):
        # FastMCP validates models before invoking the runtime handler.
        from heat_transfer_mcp.runtime import _convert_quantities

        return _convert_quantities(cls, value) if isinstance(value, dict) else value

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        for name, field in cls.model_fields.items():
            if (field.json_schema_extra or {}).get("quantity_kind"):
                prop = schema.get("properties", {}).get(name)
                if prop is not None:
                    # Clients may send a number in the declared unit or a unit string.
                    schema["properties"][name] = {
                        "anyOf": [prop, {"type": "string"}],
                        **(field.json_schema_extra or {}),
                    }
        return schema


def _model(name: str, fields: Dict[str, FieldDef]) -> Type[BaseModel]:
    return create_model(name, __base__=_InputBase, **fields)  # type: ignore[arg-type]


def _output(name: str) -> Type[ToolOutput]:
    return create_model(name, __base__=ToolOutput)  # type: ignore[return-value]


def _q(default: Any, kind: str, unit: Optional[str] = None, **kwargs: Any) -> Any:
    return QuantityField(default, kind=kind, unit=unit, **kwargs)


def _plain(default: Any, **kwargs: Any) -> Any:
    return Field(default, **kwargs)


class LayerInput(_InputBase):
    thickness: float = QuantityField(..., kind="length", unit="m", gt=0.0)
    thermal_conductivity_k: Optional[float] = QuantityField(None, kind="thermal_conductivity", unit="W/m/K", gt=0.0)
    thermal_conductivity: Optional[float] = QuantityField(None, kind="thermal_conductivity", unit="W/m/K", gt=0.0)
    k: Optional[float] = QuantityField(None, kind="thermal_conductivity", unit="W/m/K", gt=0.0)
    material_name: Optional[str] = Field(None)

    @model_validator(mode="after")
    def normalize_thermal_conductivity_aliases(self) -> "LayerInput":
        if self.thermal_conductivity_k is None:
            self.thermal_conductivity_k = self.thermal_conductivity if self.thermal_conductivity is not None else self.k
        return self


# tools.fluid_properties.CALCULATION_PROPERTY_RULE: every lookup of a calculation reads this one branch or refuses
_CALCULATION_PROPERTY_SOURCE: FieldDef = (Literal["CoolProp", "thermo"], _plain("CoolProp"))

GetAmbientConditionsInput = _model(
    "GetAmbientConditionsInput",
    {
        "latitude": (Optional[float], _plain(None)),
        "longitude": (Optional[float], _plain(None)),
        "elevation": (Optional[float], _q(None, "length", "m")),
        "location_name": (Optional[str], _plain(None)),
        "start_date": (Optional[str], _plain(None)),
        "end_date": (Optional[str], _plain(None)),
        "time_resolution": (str, _plain("daily")),
        "parameters": (List[str], Field(default_factory=lambda: ["tavg", "wspd", "pres", "rhum", "tsun"])),
    },
)

GetFluidPropertiesInput = _model(
    "GetFluidPropertiesInput",
    {
        "fluid_name": (str, _plain(...)),
        "temperature": (float, _q(..., "temperature_abs", "K")),
        "pressure": (float, _q(101325.0, "pressure", "Pa")),
        "strict": (bool, _plain(False)),
        # tools.fluid_properties.STRICT_PROPERTY_RULE: the named branch (unnamed: CoolProp) answers or the call refuses
        "property_source": (Optional[Literal["CoolProp", "thermo"]], _plain(None)),
        # Lubricating-oil datasheet anchors. Oils are absent from every pure-component
        # database, so these are the only route to their properties; all three are
        # required together because a grade name fixes only the 40 C viscosity.
        "lube_oil_nu_40_cSt": (Optional[float], _plain(None, gt=0.0)),
        "lube_oil_nu_100_cSt": (Optional[float], _plain(None, gt=0.0)),
        "lube_oil_density_15_kg_m3": (Optional[float], _plain(None, gt=0.0)),
        "lube_oil_provenance": (Optional[str], _plain(None)),
    },
)

GetMaterialPropertiesInput = _model(
    "GetMaterialPropertiesInput",
    {
        "material_name": (str, _plain(...)),
        "temperature": (Optional[float], _q(None, "temperature_abs", "K")),
    },
)

CalculateConvectionCoefficientInput = _model(
    "CalculateConvectionCoefficientInput",
    {
        "geometry": (str, _plain(...)),
        "characteristic_dimension": (float, _q(..., "length", "m")),
        "fluid_name": (str, _plain(...)),
        "bulk_fluid_temperature": (float, _q(..., "temperature_abs", "K")),
        "surface_temperature": (float, _q(..., "temperature_abs", "K")),
        "pressure": (float, _q(101325.0, "pressure", "Pa")),
        "flow_type": (str, _plain("forced")),
        "fluid_velocity": (Optional[float], _q(None, "velocity", "m/s")),
        "roughness": (Optional[float], _q(0.0, "roughness", "m")),
        "pipe_length": (Optional[float], _q(None, "length", "m")),
        "chevron_angle": (Optional[float], _q(None, "angle", "degree")),
        "plate_enlargement_factor": (Optional[float], _plain(None)),
        "phe_correlation": (str, _plain("Martin_VDI")),
        "strict": (bool, _plain(False)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

CalculateTwoPhaseHInput = _model(
    "CalculateTwoPhaseHInput",
    {
        "quality": (float, _plain(..., gt=0.0, lt=1.0)),
        "tube_diameter_m": (float, _q(..., "length", "m")),
        "fluid_name": (str, _plain(...)),
        "saturation_temperature_K": (float, _q(..., "temperature_abs", "K")),
        "mass_flux_kg_m2s": (Optional[float], _q(None, "mass_flux", "kg/m^2/s")),
        "mass_flow_kg_s": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "method": (str, _plain("auto")),
        "strict": (bool, _plain(False)),
        "length_m": (Optional[float], _q(None, "length", "m")),
        "void_fraction": (Optional[float], _plain(None, gt=0.0, lt=1.0)),
        "excess_temperature_K": (Optional[float], _q(None, "temperature_delta", "K")),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

CalculateOverallHeatTransferCoefficientInput = _model(
    "CalculateOverallHeatTransferCoefficientInput",
    {
        "geometry": (str, _plain(...)),
        "layers": (List[LayerInput], _plain(...)),
        "inner_convection_coefficient_h": (float, _q(..., "htc", "W/m^2/K")),
        "outer_convection_coefficient_h": (float, _q(..., "htc", "W/m^2/K")),
        "inner_diameter": (Optional[float], _q(None, "length", "m")),
        "outer_diameter": (Optional[float], _q(None, "length", "m")),
    },
)

CalculateSurfaceHeatTransferInput = _model(
    "CalculateSurfaceHeatTransferInput",
    {
        "geometry": (str, _plain(...)),
        "dimensions": (Dict[str, Any], _plain(...)),
        "internal_temperature": (float, _q(..., "temperature_abs", "K")),
        "ambient_air_temperature": (float, _q(..., "temperature_abs", "K")),
        "wind_speed": (float, _q(..., "velocity", "m/s")),
        "surface_emissivity": (float, _plain(..., ge=0.0, le=1.0)),
        "internal_to_surface_conductance": (Optional[float], _q(None, "htc", "W/m^2/K")),
        "wall_layers": (Optional[List[LayerInput]], _plain(None)),
        "fluid_name_internal": (str, _plain("water")),
        "fluid_name_external": (str, _plain("air")),
        "include_solar_gain": (bool, _plain(False)),
        "incident_solar_radiation": (Optional[float], _q(None, "irradiance", "W/m^2", ge=0.0)),
        "surface_absorptivity": (float, _plain(0.8, ge=0.0, le=1.0)),
        "sky_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
        "internal_convection_coefficient_h_override": (Optional[float], _q(None, "htc", "W/m^2/K")),
        "view_factor_sky_vertical": (float, _plain(0.5, ge=0.0, le=1.0)),
        "view_factor_sky_horizontal": (float, _plain(1.0, ge=0.0, le=1.0)),
        "ground_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

CalculateHeatExchangerPerformanceInput = _model(
    "CalculateHeatExchangerPerformanceInput",
    {
        "hx_type": (str, _plain("shell_tube")),
        "flow_arrangement": (Optional[str], _plain(None)),
        "hot_fluid_name": (Optional[str], _plain(None)),
        "hot_fluid_flow_rate": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "hot_fluid_inlet_temp": (Optional[float], _q(None, "temperature_abs", "K")),
        "cold_fluid_name": (Optional[str], _plain(None)),
        "cold_fluid_flow_rate": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "cold_fluid_inlet_temp": (Optional[float], _q(None, "temperature_abs", "K")),
        "overall_heat_transfer_coefficient_U": (Optional[float], _q(None, "htc", "W/m^2/K")),
        "heat_transfer_area": (Optional[float], _q(None, "area", "m^2")),
        "hot_fluid_pressure": (float, _q(101325.0, "pressure", "Pa")),
        "cold_fluid_pressure": (float, _q(101325.0, "pressure", "Pa")),
        "n_shell_passes": (int, _plain(1)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

CalculateHeatDutyInput = _model(
    "CalculateHeatDutyInput",
    {
        "calculation_method": (str, _plain(...)),
        "fluid_name": (Optional[str], _plain(None)),
        "flow_rate": (Optional[float], _q(None, "flow_rate_mass", "kg/s", gt=0.0)),
        "inlet_temp": (Optional[float], _q(None, "temperature_abs", "K")),
        "outlet_temp": (Optional[float], _q(None, "temperature_abs", "K")),
        "specific_heat_cp": (Optional[float], _q(None, "specific_heat", "J/kg/K", gt=0.0)),
        "fluid_pressure": (float, _q(101325.0, "pressure", "Pa", gt=0.0)),
        "overall_heat_transfer_coefficient_U": (Optional[float], _q(None, "htc", "W/m^2/K", gt=0.0)),
        "heat_transfer_area": (Optional[float], _q(None, "area", "m^2", gt=0.0)),
        "mean_temperature_difference": (Optional[float], _q(None, "temperature_delta", "K", gt=0.0)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

CalculateSolarRadiationOnSurfaceInput = _model(
    "CalculateSolarRadiationOnSurfaceInput",
    {
        "latitude": (float, _plain(...)),
        "longitude": (float, _plain(...)),
        "datetime_utc": (str, _plain(...)),
        "surface_tilt": (float, _q(..., "angle", "degree")),
        "surface_azimuth": (float, _q(..., "angle", "degree")),
        "altitude": (float, _q(0.0, "length", "m")),
        "atmospheric_pressure": (Optional[float], _q(None, "pressure", "Pa")),
        "direct_normal_irradiance_dni": (Optional[float], _q(None, "irradiance", "W/m^2", ge=0.0)),
        "diffuse_horizontal_irradiance_dhi": (Optional[float], _q(None, "irradiance", "W/m^2", ge=0.0)),
        "allow_measured_night_irradiance": (bool, _plain(False)),
    },
)

CalculateGroundHeatLossInput = _model(
    "CalculateGroundHeatLossInput",
    {
        "structure_type": (str, _plain(...)),
        "dimensions": (Dict[str, Any], _plain(...)),
        "depth": (float, _q(0.0, "length", "m")),
        "insulation_R_value_si": (float, _q(0.0, "fouling", "m^2*K/W")),
        "wall_thickness": (float, _q(0.2, "length", "m")),
        "wall_conductivity": (float, _q(1.7, "thermal_conductivity", "W/m/K")),
        "soil_conductivity": (Optional[float], _q(None, "thermal_conductivity", "W/m/K")),
        "internal_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
        "average_external_air_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
        "internal_convection_coefficient_h": (float, _q(8.0, "htc", "W/m^2/K")),
    },
)

CalculateBuriedObjectHeatLossInput = _model(
    "CalculateBuriedObjectHeatLossInput",
    {
        "object_type": (str, _plain(...)),
        "diameter": (float, _q(..., "length", "m")),
        "length": (Optional[float], _q(None, "length", "m")),
        "burial_depth": (Optional[float], _q(None, "length", "m")),
        "soil_conductivity": (Optional[float], _q(None, "thermal_conductivity", "W/m/K")),
        "object_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
        "ground_surface_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
    },
)

CalculateHxShellSideHKernInput = _model(
    "CalculateHxShellSideHKernInput",
    {
        "shell_inner_diameter": (float, _q(..., "length", "m")),
        "tube_outer_diameter": (float, _q(..., "length", "m")),
        "tube_pitch": (float, _q(..., "length", "m")),
        "tube_layout_angle": (int, _q(..., "angle", "degree")),
        "baffle_spacing": (float, _q(..., "length", "m")),
        "baffle_cut_percent": (float, _plain(...)),
        "shell_fluid_name": (str, _plain(...)),
        "shell_fluid_flow_rate": (float, _q(..., "flow_rate_mass", "kg/s")),
        "shell_fluid_bulk_temp": (float, _q(..., "temperature_abs", "K")),
        "shell_fluid_pressure": (float, _q(101325.0, "pressure", "Pa")),
        "tube_wall_temp": (Optional[float], _q(None, "temperature_abs", "K")),
        "tube_rows": (Optional[int], _plain(None)),
        "pitch_parallel": (Optional[float], _q(None, "length", "m")),
        "pitch_normal": (Optional[float], _q(None, "length", "m")),
        "n_baffles": (Optional[int], _plain(None)),
        "include_pressure_drop": (bool, _plain(True)),
        "strict": (bool, _plain(False)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

EstimateHxPhysicalDimsInput = _model(
    "EstimateHxPhysicalDimsInput",
    {
        "required_area": (float, _q(..., "area", "m^2")),
        "tube_outer_diameter": (float, _q(..., "length", "m")),
        "tube_length_effective": (float, _q(..., "length", "m")),
        "tube_pitch": (float, _q(..., "length", "m")),
        "tube_layout_angle": (int, _q(..., "angle", "degree")),
        "number_of_tube_passes": (int, _plain(1)),
        "method": (str, _plain("HEDH")),
    },
)

TankHeatLossInput = _model(
    "TankHeatLossInput",
    {
        "geometry": (str, _plain(...)),
        "dimensions": (Dict[str, float], _plain(...)),
        "contents_temperature": (float, _q(..., "temperature_abs", "K")),
        "fluid_name_internal": (str, _plain("water")),
        "fluid_name_external": (str, _plain("air")),
        "headspace_height_m": (float, _q(0.0, "length", "m")),
        "headspace_fluid": (str, _plain("air")),
        "headspace_h_inner_override_w_m2k": (Optional[float], _q(None, "htc", "W/m^2/K")),
        "wall_layers": (Optional[List[LayerInput]], _plain(None)),
        "insulation_R_value_si": (Optional[float], _q(None, "fouling", "m^2*K/W")),
        "assumed_insulation_k_w_mk": (float, _q(0.035, "thermal_conductivity", "W/m/K")),
        "ambient_air_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
        "wind_speed": (Optional[float], _q(None, "velocity", "m/s")),
        "sky_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
        "include_solar_gain": (bool, _plain(False)),
        "incident_solar_radiation": (Optional[float], _q(None, "irradiance", "W/m^2", ge=0.0)),
        "latitude": (Optional[float], _plain(None)),
        "longitude": (Optional[float], _plain(None)),
        "start_date": (Optional[str], _plain(None)),
        "end_date": (Optional[str], _plain(None)),
        "design_percentile": (Optional[float], _plain(None)),
        "time_resolution": (str, _plain("daily")),
        "percentiles": (Optional[List[float]], _plain(None)),
        "surface_emissivity": (float, _plain(0.85, ge=0.0, le=1.0)),
        "include_ground_contact": (bool, _plain(False)),
        "ground_config": (Optional[Dict[str, Any]], _plain(None)),
        "average_external_air_temperature": (Optional[float], _q(None, "temperature_abs", "K")),
        "solve_for": (Optional[str], _plain(None)),
        "target_heat_loss_w": (Optional[float], _q(None, "power", "W")),
        "sweep": (Optional[Dict[str, List[Any]]], _plain(None)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

ExtremeConditionsInput = _model(
    "ExtremeConditionsInput",
    {
        "latitude": (Optional[float], _plain(None)),
        "longitude": (Optional[float], _plain(None)),
        "location_name": (Optional[str], _plain(None)),
        "start_date": (Optional[str], _plain(None)),
        "end_date": (Optional[str], _plain(None)),
        "time_resolution": (str, _plain("daily")),
        "percentiles": (Optional[List[float]], _plain(None)),
        "include_wind": (bool, _plain(True)),
        "include_solar": (bool, _plain(False)),
        "design_basis": (str, _plain("cold")),
    },
)

PipeHeatManagementInput = _model(
    "PipeHeatManagementInput",
    {
        "outer_diameter_m": (float, _q(..., "length", "m")),
        "length_m": (float, _q(..., "length", "m")),
        "internal_temperature_K": (float, _q(..., "temperature_abs", "K")),
        "ambient_air_temperature_K": (float, _q(..., "temperature_abs", "K")),
        "wind_speed_m_s": (float, _q(2.0, "velocity", "m/s")),
        "fluid_name_internal": (str, _plain("water")),
        "fluid_name_external": (str, _plain("air")),
        "wall_layers": (Optional[List[LayerInput]], _plain(None)),
        "surface_emissivity": (float, _plain(0.85, ge=0.0, le=1.0)),
        "installation": (str, _plain("above_ground")),
        "burial_depth_m": (Optional[float], _q(None, "length", "m")),
        "soil_conductivity_w_mk": (Optional[float], _q(None, "thermal_conductivity", "W/m/K")),
        "ground_surface_temperature_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "solve_for": (Optional[str], _plain(None)),
        "target_temperature_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "target_heat_loss_w_per_m": (Optional[float], _q(None, "linear_power", "W/m")),
        "pipe_inner_diameter_m": (Optional[float], _q(None, "length", "m")),
        "fluid_density_kg_m3": (Optional[float], _q(1000.0, "density", "kg/m^3")),
        "fluid_cp_j_kgk": (Optional[float], _q(4180.0, "specific_heat", "J/kg/K")),
        "freeze_temperature_K": (float, _q(273.15, "temperature_abs", "K")),
        "include_latent_heat": (bool, _plain(True)),
        "latent_heat_j_kg": (float, _q(334000.0, "specific_energy", "J/kg")),
        "heat_trace_w_per_m": (float, _q(0.0, "linear_power", "W/m")),
        "heat_trace_safety_factor": (float, _plain(1.25)),
        "freeze_protection_margin_K": (float, _q(5.0, "temperature_delta", "K")),
        "round_recommendation_to_catalog": (bool, _plain(True)),
        "available_heat_trace_ratings_w_per_m": (Optional[List[float]], _q(None, "linear_power", "W/m")),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

ParameterOptimizationInput = _model(
    "ParameterOptimizationInput",
    {
        "tool_name": (str, _plain(...)),
        "base_params": (Dict[str, Any], _plain(...)),
        "sweep": (Dict[str, List[Any]], _plain(...)),
        "objective_key": (str, _plain(...)),
        "direction": (str, _plain("minimize")),
        "constraints": (Optional[List[Dict[str, Any]]], _plain(None)),
        "top_n": (int, _plain(5)),
    },
)

CalculatePressureDropInput = _model(
    "CalculatePressureDropInput",
    {
        "geometry": (str, _plain(...)),
        "mass_flow_kg_s": (float, _q(..., "flow_rate_mass", "kg/s")),
        "fluid_name": (str, _plain(...)),
        "fluid_temperature_K": (float, _q(..., "temperature_abs", "K")),
        "hydraulic_diameter_m": (Optional[float], _q(None, "length", "m")),
        "flow_length_m": (Optional[float], _q(None, "length", "m")),
        "flow_area_m2": (Optional[float], _q(None, "area", "m^2")),
        "chevron_angle": (Optional[float], _q(None, "angle", "degree")),
        "plate_enlargement_factor": (Optional[float], _plain(None)),
        "n_channels": (Optional[int], _plain(None)),
        "n_passes": (int, _plain(1)),
        "port_diameter_m": (Optional[float], _q(None, "length", "m")),
        "shell_diameter_m": (Optional[float], _q(None, "length", "m")),
        "baffle_spacing_m": (Optional[float], _q(None, "length", "m")),
        "tube_pitch_m": (Optional[float], _q(None, "length", "m")),
        "tube_od_m": (Optional[float], _q(None, "length", "m")),
        "n_baffles": (Optional[int], _plain(None)),
        "pipe_roughness_m": (float, _q(0.0, "roughness", "m")),
        "include_port_losses": (bool, _plain(True)),
        "correlation": (str, _plain("auto")),
        "fluid_pressure_Pa": (float, _q(101325.0, "pressure", "Pa")),
        "strict": (bool, _plain(False)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

SizePlateHeatExchangerInput = _model(
    "SizePlateHeatExchangerInput",
    {
        "heat_duty_W": (Optional[float], _q(None, "power", "W")),
        "hot_inlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "hot_outlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "cold_inlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "cold_outlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "hot_mass_flow_kg_s": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "cold_mass_flow_kg_s": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "hot_fluid": (str, _plain("water")),
        "cold_fluid": (str, _plain("water")),
        "hot_fluid_pressure_Pa": (float, _q(101325.0, "pressure", "Pa")),
        "cold_fluid_pressure_Pa": (float, _q(101325.0, "pressure", "Pa")),
        "plate_amplitude_m": (Optional[float], _q(None, "length", "m")),
        "plate_wavelength_m": (Optional[float], _q(None, "length", "m")),
        "plate_width_m": (Optional[float], _q(None, "length", "m")),
        "plate_length_m": (Optional[float], _q(None, "length", "m")),
        "port_diameter_m": (Optional[float], _q(None, "length", "m")),
        "plate_thickness_m": (float, _q(0.0005, "length", "m")),
        "plate_conductivity_W_mK": (float, _q(16.0, "thermal_conductivity", "W/m/K")),
        "chevron_angle_deg": (float, _q(45.0, "angle", "degree")),
        "n_plates": (Optional[int], _plain(None)),
        "passes_hot": (int, _plain(1)),
        "passes_cold": (int, _plain(1)),
        "flow_arrangement": (str, _plain("counterflow")),
        "max_pressure_drop_kPa": (Optional[float], _q(None, "pressure", "kPa")),
        "min_plates": (int, _plain(5)),
        "max_plates": (int, _plain(201)),
        "fouling_factor_hot_m2K_W": (float, _q(0.0, "fouling", "m^2*K/W")),
        "fouling_factor_cold_m2K_W": (float, _q(0.0, "fouling", "m^2*K/W")),
        "correlation": (str, _plain("Martin_VDI")),
        "sweep_max_pressure_drop_kPa": (Optional[List[float]], _plain(None)),
        "auto_optimize": (bool, _plain(False)),
        "optimize_for": (str, _plain("area")),
        "strict": (bool, _plain(False)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

SizeShellTubeHeatExchangerInput = _model(
    "SizeShellTubeHeatExchangerInput",
    {
        "heat_duty_W": (Optional[float], _q(None, "power", "W")),
        "hot_inlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "hot_outlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "cold_inlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "cold_outlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "hot_mass_flow_kg_s": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "cold_mass_flow_kg_s": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "hot_fluid": (str, _plain("water")),
        "cold_fluid": (str, _plain("water")),
        "hot_fluid_pressure_Pa": (float, _q(101325.0, "pressure", "Pa")),
        "cold_fluid_pressure_Pa": (float, _q(101325.0, "pressure", "Pa")),
        "shell_inner_diameter_m": (Optional[float], _q(None, "length", "m")),
        "baffle_spacing_m": (Optional[float], _q(None, "length", "m")),
        "baffle_cut_fraction": (float, _plain(0.25)),
        "n_baffles": (Optional[int], _plain(None)),
        "tube_outer_diameter_m": (float, _q(0.019, "length", "m")),
        "tube_inner_diameter_m": (float, _q(0.016, "length", "m")),
        "tube_pitch_m": (float, _q(0.025, "length", "m")),
        "tube_layout_angle": (int, _q(30, "angle", "degree")),
        "tube_length_m": (float, _q(3.0, "length", "m")),
        "tube_roughness_m": (float, _q(4e-5, "roughness", "m")),
        "tube_material_conductivity_W_mK": (float, _q(45.0, "thermal_conductivity", "W/m/K")),
        "n_tube_passes": (int, _plain(2)),
        "n_shell_passes": (int, _plain(1)),
        "tube_side_fluid": (str, _plain("cold")),
        "max_pressure_drop_tube_kPa": (Optional[float], _q(None, "pressure", "kPa")),
        "max_pressure_drop_shell_kPa": (Optional[float], _q(None, "pressure", "kPa")),
        "max_velocity_tube_m_s": (float, _q(3.0, "velocity", "m/s")),
        "max_velocity_shell_m_s": (float, _q(1.5, "velocity", "m/s")),
        "min_velocity_tube_m_s": (float, _q(0.3, "velocity", "m/s")),
        "min_Re_tube": (float, _plain(2500.0)),
        "fouling_factor_tube_m2K_W": (float, _q(0.0001, "fouling", "m^2*K/W")),
        "fouling_factor_shell_m2K_W": (float, _q(0.0002, "fouling", "m^2*K/W")),
        "shell_side_method": (str, _plain("Kern")),
        "tube_side_method": (str, _plain("auto")),
        "solve_for": (str, _plain("n_tubes")),
        "min_tubes": (int, _plain(10)),
        "max_tubes": (int, _plain(2000)),
        "sweep_n_tube_passes": (Optional[List[int]], _plain(None)),
        "auto_optimize": (bool, _plain(False)),
        "optimize_for": (str, _plain("area")),
        "strict": (bool, _plain(False)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)

SizeDoublePipeHeatExchangerInput = _model(
    "SizeDoublePipeHeatExchangerInput",
    {
        "heat_duty_W": (Optional[float], _q(None, "power", "W")),
        "hot_inlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "hot_outlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "cold_inlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "cold_outlet_temp_K": (Optional[float], _q(None, "temperature_abs", "K")),
        "hot_mass_flow_kg_s": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "cold_mass_flow_kg_s": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "hot_fluid": (str, _plain("water")),
        "cold_fluid": (str, _plain("water")),
        "hot_fluid_pressure_Pa": (float, _q(101325.0, "pressure", "Pa")),
        "cold_fluid_pressure_Pa": (float, _q(101325.0, "pressure", "Pa")),
        "inner_pipe_outer_diameter_m": (float, _q(0.0254, "length", "m")),
        "inner_pipe_inner_diameter_m": (float, _q(0.0229, "length", "m")),
        "inner_pipe_roughness_m": (float, _q(4e-5, "roughness", "m")),
        "inner_pipe_material_conductivity_W_mK": (float, _q(45.0, "thermal_conductivity", "W/m/K")),
        "outer_pipe_inner_diameter_m": (float, _q(0.0525, "length", "m")),
        "flow_arrangement": (str, _plain("counterflow")),
        "inner_pipe_fluid": (str, _plain("hot")),
        "n_hairpins": (int, _plain(1)),
        "max_pressure_drop_inner_kPa": (Optional[float], _q(None, "pressure", "kPa")),
        "max_pressure_drop_annulus_kPa": (Optional[float], _q(None, "pressure", "kPa")),
        "max_velocity_inner_m_s": (float, _q(3.0, "velocity", "m/s")),
        "max_velocity_annulus_m_s": (float, _q(2.0, "velocity", "m/s")),
        "min_velocity_inner_m_s": (float, _q(0.3, "velocity", "m/s")),
        "min_velocity_annulus_m_s": (float, _q(0.2, "velocity", "m/s")),
        "min_Re_inner": (float, _plain(2500.0)),
        "fouling_factor_inner_m2K_W": (float, _q(0.0001, "fouling", "m^2*K/W")),
        "fouling_factor_annulus_m2K_W": (float, _q(0.0002, "fouling", "m^2*K/W")),
        "min_length_m": (float, _q(1.0, "length", "m")),
        "max_length_m": (float, _q(20.0, "length", "m")),
        "length_step_m": (float, _q(0.1, "length", "m")),
        "solve_for": (str, _plain("length")),
        "pipe_length_m": (Optional[float], _q(None, "length", "m")),
        "strict": (bool, _plain(False)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)


class AcheBundleInput(_InputBase):
    """Finned-tube bundle geometry, as printed on a vendor datasheet."""

    tube_rows: int = Field(..., gt=0)
    tube_passes: int = Field(..., gt=0)
    tubes_per_row: int = Field(..., gt=0)
    tube_length_m: float = QuantityField(..., kind="length", unit="m", gt=0.0)
    tube_od_m: float = QuantityField(..., kind="length", unit="m", gt=0.0)
    fin_thickness_m: float = QuantityField(..., kind="length", unit="m", gt=0.0)
    fin_density_per_m: float = Field(..., gt=0.0, description="Fins per metre of tube")
    fin_height_m: float = QuantityField(..., kind="length", unit="m", gt=0.0)
    pitch_normal_m: Optional[float] = QuantityField(None, kind="length", unit="m", gt=0.0)
    pitch_parallel_m: Optional[float] = QuantityField(None, kind="length", unit="m", gt=0.0)
    tube_thickness_m: Optional[float] = QuantityField(None, kind="length", unit="m", gt=0.0)
    parallel_bays: Optional[int] = Field(None, gt=0)
    bundles_per_bay: Optional[int] = Field(None, gt=0)
    fans_per_bay: Optional[int] = Field(None, gt=0)
    fan_diameter_m: Optional[float] = QuantityField(None, kind="length", unit="m", gt=0.0)
    # Defaulted in the handler rather than here, so the response can distinguish a value
    # the vendor stated from one the tool assumed.
    fin_conductivity_W_mK: Optional[float] = QuantityField(
        None, kind="thermal_conductivity", unit="W/m/K", gt=0.0
    )


class AcheTrainStageInput(_InputBase):
    """One cooler in a series train; water content threads between stages."""

    label: Optional[str] = Field(None)
    gas_nm3h_wet: float = Field(..., gt=0.0)
    P_bara: float = Field(..., gt=0.0)
    T_in_C: float = Field(...)
    T_out_C: float = Field(...)
    knockout: bool = Field(True)


class AcheProcessInput(_InputBase):
    """Process (tube) side. Supply either a single cooler or a series train."""

    gas_nm3h_wet: Optional[float] = Field(None, gt=0.0)
    P_bara: Optional[float] = Field(None, gt=0.0)
    T_in_C: Optional[float] = Field(None)
    T_out_C: Optional[float] = Field(None)
    label: Optional[str] = Field(None)
    train_stages: Optional[List[AcheTrainStageInput]] = Field(None)
    rated_stage_index: Optional[int] = Field(
        None,
        description="Which train stage this bundle serves, for the MTD correction. "
                    "Defaults to the last stage.",
    )
    inlet_adp_C: Optional[float] = Field(
        None,
        description="Atmospheric dew point of the gas entering the first cooler. "
                    "Defaults to the house basis when omitted.",
    )
    water_mole_frac_in: Optional[float] = Field(None, ge=0.0, lt=1.0)
    dry_composition: Optional[Dict[str, float]] = Field(
        None, description="Dry-basis mole fractions, e.g. {'methane': 0.6, 'carbon dioxide': 0.4}"
    )


class AcheFanInput(_InputBase):
    """Fan efficiencies. Required to report shaft or absorbed electrical power."""

    fan_static_efficiency: Optional[float] = Field(None, gt=0.0, le=1.0)
    drive_efficiency: Optional[float] = Field(None, gt=0.0, le=1.0)
    motor_efficiency: Optional[float] = Field(None, gt=0.0, le=1.0)
    external_dP_Pa: float = QuantityField(0.0, kind="pressure", unit="Pa", ge=0.0)
    vendor_motor_kW: Optional[float] = QuantityField(None, kind="power", unit="kW", gt=0.0)


RateAirCooledHeatExchangerInput = _model(
    "RateAirCooledHeatExchangerInput",
    {
        "bundle": (AcheBundleInput, _plain(...)),
        "air_inlet_temp_C": (float, _plain(...)),
        "air_outlet_temp_C": (Optional[float], _plain(None)),
        "air_mass_flow_kg_s": (Optional[float], _q(None, "flow_rate_mass", "kg/s")),
        "air_face_velocity_m_s": (Optional[float], _q(None, "velocity", "m/s")),
        "site_altitude_m": (float, _q(0.0, "length", "m")),
        "process": (Optional[AcheProcessInput], _plain(None)),
        "fan": (Optional[AcheFanInput], _plain(None)),
        "tube_side_h_W_m2K": (Optional[float], _q(None, "htc", "W/m^2/K")),
        "tube_inside_area_m2": (Optional[float], _q(None, "area", "m^2")),
        "fouling_air_m2K_W": (Optional[float], _q(None, "fouling", "m^2*K/W")),
        "fouling_tube_m2K_W": (Optional[float], _q(None, "fouling", "m^2*K/W")),
        "api661_invoked": (bool, _plain(True)),
        "purchaser_agreed_single_fan": (bool, _plain(False)),
        "property_source": _CALCULATION_PROPERTY_SOURCE,
    },
)


_PAIRS = {
    "tank_heat_loss": TankHeatLossInput,
    "extreme_conditions": ExtremeConditionsInput,
    "pipe_heat_management": PipeHeatManagementInput,
    "parameter_optimization": ParameterOptimizationInput,
    "get_ambient_conditions": GetAmbientConditionsInput,
    "get_fluid_properties": GetFluidPropertiesInput,
    "get_material_properties": GetMaterialPropertiesInput,
    "calculate_convection_coefficient": CalculateConvectionCoefficientInput,
    "calculate_two_phase_h": CalculateTwoPhaseHInput,
    "calculate_overall_heat_transfer_coefficient": CalculateOverallHeatTransferCoefficientInput,
    "calculate_surface_heat_transfer": CalculateSurfaceHeatTransferInput,
    "calculate_heat_exchanger_performance": CalculateHeatExchangerPerformanceInput,
    "calculate_heat_duty": CalculateHeatDutyInput,
    "calculate_solar_radiation_on_surface": CalculateSolarRadiationOnSurfaceInput,
    "calculate_ground_heat_loss": CalculateGroundHeatLossInput,
    "calculate_buried_object_heat_loss": CalculateBuriedObjectHeatLossInput,
    "calculate_hx_shell_side_h_kern": CalculateHxShellSideHKernInput,
    "estimate_hx_physical_dims": EstimateHxPhysicalDimsInput,
    "calculate_pressure_drop": CalculatePressureDropInput,
    "size_plate_heat_exchanger": SizePlateHeatExchangerInput,
    "size_shell_tube_heat_exchanger": SizeShellTubeHeatExchangerInput,
    "size_double_pipe_heat_exchanger": SizeDoublePipeHeatExchangerInput,
    "rate_air_cooled_heat_exchanger": RateAirCooledHeatExchangerInput,
}


TOOL_SCHEMAS: Dict[str, ToolSchema] = {
    name: ToolSchema(name=name, input_model=input_model, output_model=_output(input_model.__name__.replace("Input", "Output")))
    for name, input_model in _PAIRS.items()
}


def get_tool_schema(name: str) -> ToolSchema:
    return TOOL_SCHEMAS[name]


__all__ = [
    "TOOL_SCHEMAS",
    "get_tool_schema",
    *[model.__name__ for model in _PAIRS.values()],
]
