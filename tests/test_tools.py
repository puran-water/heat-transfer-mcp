"""Test tool functionality."""

import json
import pytest
from tools.fluid_properties import get_fluid_properties
from tools.material_properties import get_material_properties


class TestFluidProperties:
    def test_water_properties(self):
        """Test water properties at room temperature."""
        props_json = get_fluid_properties("water", 298.15)
        props = json.loads(props_json)

        assert "density" in props
        assert "dynamic_viscosity" in props
        assert "thermal_conductivity" in props
        assert "specific_heat_cp" in props  # Changed from specific_heat_capacity

        # Check reasonable values for water at 25°C
        assert 990 < props["density"] < 1010  # kg/m³
        assert 0.0008 < props["dynamic_viscosity"] < 0.001  # Pa·s
        assert 0.6 < props["thermal_conductivity"] < 0.7  # W/m-K
        assert 4000 < props["specific_heat_cp"] < 4300  # J/kg-K

    def test_air_properties(self):
        """Test air properties at room temperature."""
        props_json = get_fluid_properties("air", 298.15)
        props = json.loads(props_json)

        assert "density" in props
        assert "dynamic_viscosity" in props

        # Check reasonable values for air at 25°C
        assert 1.0 < props["density"] < 1.3  # kg/m³
        assert 1.7e-5 < props["dynamic_viscosity"] < 2.0e-5  # Pa·s

    def test_invalid_fluid(self):
        """Test handling of invalid fluid names."""
        props_json = get_fluid_properties("nonexistent_fluid", 298.15)
        props = json.loads(props_json)
        assert "error" in props  # Should return error in JSON


class TestMaterialProperties:
    def test_steel_properties(self):
        """Test steel material properties."""
        props_json = get_material_properties("steel")
        props = json.loads(props_json)

        assert "thermal_conductivity_k" in props or "thermal_conductivity" in props
        assert "density_rho" in props or "density" in props
        assert "specific_heat_cp" in props or "specific_heat_capacity" in props

        # Check reasonable values for steel
        k_value = props.get("thermal_conductivity_k", props.get("thermal_conductivity", 0))
        assert 40 < k_value < 60  # W/m-K
        density_value = props.get("density_rho", props.get("density", 0))
        cp_value = props.get("specific_heat_cp", props.get("specific_heat_capacity", 0))
        assert 7500 < density_value < 8000  # kg/m³
        assert 400 < cp_value < 600  # J/kg-K

    def test_concrete_properties(self):
        """Test concrete material properties."""
        props_json = get_material_properties("concrete")
        props = json.loads(props_json)

        assert "thermal_conductivity_k" in props or "thermal_conductivity" in props

        # Check reasonable values for concrete
        k_value = props.get("thermal_conductivity_k", props.get("thermal_conductivity", 0))
        assert 1.0 < k_value < 3.0  # W/m-K

    def test_fuzzy_matching(self):
        """Test fuzzy material name matching."""
        # Should find polyurethane foam even with slight misspelling
        props1_json = get_material_properties("polyurethane foam")
        props2_json = get_material_properties("polyurethane")
        props1 = json.loads(props1_json)
        props2 = json.loads(props2_json)

        assert "thermal_conductivity_k" in props1 or "thermal_conductivity" in props1
        assert "thermal_conductivity_k" in props2 or "thermal_conductivity" in props2

    def test_temperature_dependent(self):
        """Test temperature-dependent material properties."""
        props_cold_json = get_material_properties("steel", temperature=273.15)
        props_hot_json = get_material_properties("steel", temperature=373.15)
        props_cold = json.loads(props_cold_json)
        props_hot = json.loads(props_hot_json)

        assert "thermal_conductivity_k" in props_cold or "thermal_conductivity" in props_cold
        assert "thermal_conductivity_k" in props_hot or "thermal_conductivity" in props_hot

        # Thermal conductivity may vary with temperature
        k_cold = props_cold.get("thermal_conductivity_k", props_cold.get("thermal_conductivity", 0))
        k_hot = props_hot.get("thermal_conductivity_k", props_hot.get("thermal_conductivity", 0))
        assert isinstance(k_cold, (int, float))
        assert isinstance(k_hot, (int, float))
