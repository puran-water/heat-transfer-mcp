"""Test tool functionality."""
import pytest
from tools.fluid_properties import get_fluid_properties
from tools.material_properties import get_material_properties


class TestFluidProperties:
    def test_water_properties(self):
        """Test water properties at room temperature."""
        props = get_fluid_properties("water", 298.15)
        
        assert "density" in props
        assert "dynamic_viscosity" in props
        assert "thermal_conductivity" in props
        assert "specific_heat_capacity" in props
        
        # Check reasonable values for water at 25°C
        assert 990 < props["density"] < 1010  # kg/m³
        assert 0.0008 < props["dynamic_viscosity"] < 0.001  # Pa·s
        assert 0.6 < props["thermal_conductivity"] < 0.7  # W/m-K
        assert 4000 < props["specific_heat_capacity"] < 4300  # J/kg-K
    
    def test_air_properties(self):
        """Test air properties at room temperature."""
        props = get_fluid_properties("air", 298.15)
        
        assert "density" in props
        assert "dynamic_viscosity" in props
        
        # Check reasonable values for air at 25°C
        assert 1.0 < props["density"] < 1.3  # kg/m³
        assert 1.8e-5 < props["dynamic_viscosity"] < 2.0e-5  # Pa·s
    
    def test_invalid_fluid(self):
        """Test handling of invalid fluid names."""
        with pytest.raises(Exception):
            get_fluid_properties("nonexistent_fluid", 298.15)


class TestMaterialProperties:
    def test_steel_properties(self):
        """Test steel material properties."""
        props = get_material_properties("steel")
        
        assert "thermal_conductivity" in props
        assert "density" in props
        assert "specific_heat_capacity" in props
        
        # Check reasonable values for steel
        assert 40 < props["thermal_conductivity"] < 60  # W/m-K
        assert 7500 < props["density"] < 8000  # kg/m³
        assert 400 < props["specific_heat_capacity"] < 600  # J/kg-K
    
    def test_concrete_properties(self):
        """Test concrete material properties."""
        props = get_material_properties("concrete")
        
        assert "thermal_conductivity" in props
        
        # Check reasonable values for concrete
        assert 1.0 < props["thermal_conductivity"] < 3.0  # W/m-K
    
    def test_fuzzy_matching(self):
        """Test fuzzy material name matching."""
        # Should find polyurethane foam even with slight misspelling
        props1 = get_material_properties("polyurethane foam")
        props2 = get_material_properties("polyurethane")
        
        assert "thermal_conductivity" in props1
        assert "thermal_conductivity" in props2
    
    def test_temperature_dependent(self):
        """Test temperature-dependent material properties."""
        props_cold = get_material_properties("steel", temperature=273.15)
        props_hot = get_material_properties("steel", temperature=373.15)
        
        assert "thermal_conductivity" in props_cold
        assert "thermal_conductivity" in props_hot
        
        # Thermal conductivity may vary with temperature
        assert isinstance(props_cold["thermal_conductivity"], (int, float))
        assert isinstance(props_hot["thermal_conductivity"], (int, float))