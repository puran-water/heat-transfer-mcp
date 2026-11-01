"""ToolSpec definitions for the registered heat-transfer tools."""

from __future__ import annotations

from typing import Dict, List

from heat_transfer_mcp.tools.air_cooled_heat_exchanger import rate_air_cooled_heat_exchanger
from heat_transfer_mcp.tools.ambient_conditions import get_ambient_conditions
from heat_transfer_mcp.tools.buried_object_heat_loss import calculate_buried_object_heat_loss
from heat_transfer_mcp.tools.calculate_pressure_drop import calculate_pressure_drop
from heat_transfer_mcp.tools.convection_coefficient import calculate_convection_coefficient, calculate_two_phase_h
from heat_transfer_mcp.tools.estimate_hx_physical_dims import estimate_hx_physical_dims
from heat_transfer_mcp.tools.extreme_conditions import extreme_conditions
from heat_transfer_mcp.tools.fluid_properties import get_fluid_properties
from heat_transfer_mcp.tools.ground_heat_loss import calculate_ground_heat_loss
from heat_transfer_mcp.tools.heat_duty import calculate_heat_duty
from heat_transfer_mcp.tools.heat_exchanger import calculate_heat_exchanger_performance
from heat_transfer_mcp.tools.hx_shell_side_h_kern import calculate_hx_shell_side_h_kern
from heat_transfer_mcp.tools.material_properties import get_material_properties
from heat_transfer_mcp.tools.overall_heat_transfer import calculate_overall_heat_transfer_coefficient
from heat_transfer_mcp.tools.parameter_optimization import parameter_optimization
from heat_transfer_mcp.tools.pipe_heat_management import pipe_heat_management
from heat_transfer_mcp.tools.plate_heat_exchanger_sizing import size_plate_heat_exchanger
from heat_transfer_mcp.tools.size_double_pipe_heat_exchanger import size_double_pipe_heat_exchanger
from heat_transfer_mcp.tools.size_shell_tube_heat_exchanger import size_shell_tube_heat_exchanger
from heat_transfer_mcp.tools.solar_radiation import calculate_solar_radiation_on_surface
from heat_transfer_mcp.tools.surface_heat_transfer import calculate_surface_heat_transfer
from heat_transfer_mcp.tools.tank_heat_loss import tank_heat_loss

from heat_transfer_mcp.registry import ToolSpec
from heat_transfer_mcp.runtime import run_tool_handler
from heat_transfer_mcp.schemas import TOOL_SCHEMAS

TOOL_FUNCTIONS = {
    "tank_heat_loss": tank_heat_loss,
    "extreme_conditions": extreme_conditions,
    "pipe_heat_management": pipe_heat_management,
    "parameter_optimization": parameter_optimization,
    "get_ambient_conditions": get_ambient_conditions,
    "get_fluid_properties": get_fluid_properties,
    "get_material_properties": get_material_properties,
    "calculate_convection_coefficient": calculate_convection_coefficient,
    "calculate_two_phase_h": calculate_two_phase_h,
    "calculate_overall_heat_transfer_coefficient": calculate_overall_heat_transfer_coefficient,
    "calculate_surface_heat_transfer": calculate_surface_heat_transfer,
    "calculate_heat_exchanger_performance": calculate_heat_exchanger_performance,
    "calculate_heat_duty": calculate_heat_duty,
    "calculate_solar_radiation_on_surface": calculate_solar_radiation_on_surface,
    "calculate_ground_heat_loss": calculate_ground_heat_loss,
    "calculate_buried_object_heat_loss": calculate_buried_object_heat_loss,
    "calculate_hx_shell_side_h_kern": calculate_hx_shell_side_h_kern,
    "estimate_hx_physical_dims": estimate_hx_physical_dims,
    "calculate_pressure_drop": calculate_pressure_drop,
    "size_plate_heat_exchanger": size_plate_heat_exchanger,
    "size_shell_tube_heat_exchanger": size_shell_tube_heat_exchanger,
    "size_double_pipe_heat_exchanger": size_double_pipe_heat_exchanger,
    "rate_air_cooled_heat_exchanger": rate_air_cooled_heat_exchanger,
}


def _build_spec(name: str) -> ToolSpec:
    schema = TOOL_SCHEMAS[name]

    def handler(payload):
        return run_tool_handler(
            name=name,
            input_model=schema.input_model,
            output_model=schema.output_model,
            tool_func=TOOL_FUNCTIONS[name],
            payload=payload,
        )

    handler.__name__ = f"{name}_handler"
    return ToolSpec(name=name, input_model=schema.input_model, output_model=schema.output_model, handler=handler)


ALL_TOOL_SPECS: List[ToolSpec] = [_build_spec(name) for name in TOOL_SCHEMAS]
TOOL_SPECS_BY_NAME: Dict[str, ToolSpec] = {spec.name: spec for spec in ALL_TOOL_SPECS}


def get_tool_spec(name: str) -> ToolSpec:
    return TOOL_SPECS_BY_NAME[name]
