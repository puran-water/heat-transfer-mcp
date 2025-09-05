"""
Minimal test suite for core heat transfer calculations.
Tests critical physics and validates against known solutions.
"""

import json
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import estimate_sky_temperature
from tools.surface_heat_transfer import calculate_surface_heat_transfer
from tools.tank_heat_loss import tank_heat_loss


class TestSkyTemperature:
    """Test sky temperature estimation formula."""
    
    def test_sky_temp_without_dew_point(self):
        """Test Swinbank correlation for clear sky."""
        T_ambient = 273.15  # 0°C
        T_sky = estimate_sky_temperature(T_ambient)
        
        # Sky should be colder than ambient
        assert T_sky < T_ambient
        # But not unreasonably cold (should be > -50°C)
        assert T_sky > 223.15
        
    def test_sky_temp_with_dew_point(self):
        """Test improved formula with dew point."""
        T_ambient = 273.15  # 0°C
        T_dew = 268.15  # -5°C
        
        T_sky = estimate_sky_temperature(T_ambient, T_dew)
        
        # Sky temp should be between dew point and ambient
        assert T_dew < T_sky < T_ambient
        # Typical depression is 15-25K
        assert 248 < T_sky < 258
        
    def test_sky_temp_formula_correction(self):
        """Verify the critical formula fix (emissivity^0.25)."""
        T_ambient = 273.15
        T_dew = 268.15  # -5°C dew point
        
        # Calculate emissivity using the formula
        Td_C = T_dew - 273.15
        eps_clear = 0.711 + 0.0056*Td_C + 0.000073*(Td_C**2)
        
        # Sky temperature should use fourth root
        expected = T_ambient * (eps_clear ** 0.25)
        actual = estimate_sky_temperature(T_ambient, T_dew)
        
        assert abs(actual - expected) < 0.1


class TestSurfaceHeatTransfer:
    """Test surface heat transfer with view factors."""
    
    def test_vertical_cylinder_area(self):
        """Test that vertical cylinder excludes bottom from air-exposed area."""
        result_json = calculate_surface_heat_transfer(
            geometry="vertical_cylinder_tank",
            dimensions={"diameter": 10.0, "height": 5.0},
            internal_temperature=300.0,
            ambient_air_temperature=273.15,
            wind_speed=5.0,
            surface_emissivity=0.9,
            wall_layers=[{"thickness": 0.1, "thermal_conductivity_k": 50}]
        )
        
        result = json.loads(result_json)
        assert "error" not in result
        
        # Area should be lateral + top only (not bottom)
        diameter = 10.0
        height = 5.0
        lateral_area = 3.14159 * diameter * height
        top_area = 3.14159 * (diameter/2)**2
        expected_area = lateral_area + top_area
        
        actual_area = result["outer_surface_area_m2"]
        assert abs(actual_area - expected_area) / expected_area < 0.01
        
    def test_view_factors_applied(self):
        """Test that view factors affect radiation calculation."""
        # Test with default view factors
        result1_json = calculate_surface_heat_transfer(
            geometry="vertical_cylinder_tank",
            dimensions={"diameter": 10.0, "height": 5.0},
            internal_temperature=300.0,
            ambient_air_temperature=273.15,
            wind_speed=5.0,
            surface_emissivity=0.9,
            view_factor_sky_vertical=0.5,
            view_factor_sky_horizontal=1.0
        )
        
        # Test with different view factors
        result2_json = calculate_surface_heat_transfer(
            geometry="vertical_cylinder_tank",
            dimensions={"diameter": 10.0, "height": 5.0},
            internal_temperature=300.0,
            ambient_air_temperature=273.15,
            wind_speed=5.0,
            surface_emissivity=0.9,
            view_factor_sky_vertical=1.0,  # All sky
            view_factor_sky_horizontal=1.0
        )
        
        result1 = json.loads(result1_json)
        result2 = json.loads(result2_json)
        
        # More sky view should mean more radiation loss
        assert result2["radiative_heat_rate_w"] > result1["radiative_heat_rate_w"]
        
    def test_convergence(self):
        """Test that iterative solver converges."""
        result_json = calculate_surface_heat_transfer(
            geometry="flat_surface",
            dimensions={"length": 1.0, "width": 1.0},
            internal_temperature=350.0,  # Hot surface
            ambient_air_temperature=273.15,
            wind_speed=10.0,
            surface_emissivity=0.8
        )
        
        result = json.loads(result_json)
        assert "error" not in result
        assert result["converged"] == True
        assert result["iterations_required"] < 50


class TestTankHeatLoss:
    """Test the main tank heat loss omnibus tool."""
    
    def test_basic_tank_calculation(self):
        """Test basic vertical tank heat loss calculation."""
        result_json = tank_heat_loss(
            geometry="vertical_cylinder_tank",
            dimensions={"diameter": 10.0, "height": 10.0},
            contents_temperature=308.15,  # 35°C
            ambient_air_temperature=273.15,  # 0°C
            wind_speed=5.0,
            insulation_R_value_si=2.0
        )
        
        result = json.loads(result_json)
        assert "error" not in result
        
        # Basic sanity checks
        assert result["total_heat_loss_w"] > 0  # Heat flows out
        assert result["estimated_outer_surface_temp_k"] < 308.15  # Surface colder than contents
        assert result["estimated_outer_surface_temp_k"] > 273.15  # But warmer than ambient (insulated)
        
    def test_ground_contact_included(self):
        """Test that vertical tanks automatically include ground heat loss."""
        result_json = tank_heat_loss(
            geometry="vertical_cylinder_tank",
            dimensions={"diameter": 10.0, "height": 10.0},
            contents_temperature=308.15,
            ambient_air_temperature=273.15,
            wind_speed=5.0
        )
        
        result = json.loads(result_json)
        assert "error" not in result
        
        # Vertical tank should have ground heat loss
        assert result["ground_heat_loss_w"] > 0
        assert "ground_details" in result
        
    def test_headspace_modeling(self):
        """Test two-zone headspace model."""
        result_json = tank_heat_loss(
            geometry="vertical_cylinder_tank", 
            dimensions={"diameter": 10.0, "height": 10.0},
            contents_temperature=308.15,
            ambient_air_temperature=273.15,
            wind_speed=5.0,
            headspace_height_m=2.0,  # 2m of gas space
            headspace_fluid="air"
        )
        
        result = json.loads(result_json)
        assert "error" not in result
        
        # Should have headspace info
        assert "headspace_info" in result
        assert result["headspace_info"]["headspace_height_m"] == 2.0
        assert result["headspace_info"]["wetted_heat_loss_w"] > 0
        assert result["headspace_info"]["dry_heat_loss_w"] > 0
        
    def test_insulation_effectiveness(self):
        """Test that insulation reduces heat loss."""
        # No insulation
        result1_json = tank_heat_loss(
            geometry="vertical_cylinder_tank",
            dimensions={"diameter": 10.0, "height": 10.0},
            contents_temperature=308.15,
            ambient_air_temperature=273.15,
            wind_speed=5.0,
            insulation_R_value_si=0.0
        )
        
        # With insulation
        result2_json = tank_heat_loss(
            geometry="vertical_cylinder_tank",
            dimensions={"diameter": 10.0, "height": 10.0},
            contents_temperature=308.15,
            ambient_air_temperature=273.15,
            wind_speed=5.0,
            insulation_R_value_si=3.0
        )
        
        result1 = json.loads(result1_json)
        result2 = json.loads(result2_json)
        
        # Insulation should significantly reduce heat loss
        assert result2["total_heat_loss_w"] < result1["total_heat_loss_w"] * 0.5
        

class TestPhysicsValidation:
    """Validate physics against known solutions."""
    
    def test_conduction_only_flat_wall(self):
        """Test pure conduction through flat wall against Q = k*A*dT/L."""
        # Use very high h values to minimize convection resistance
        result_json = calculate_surface_heat_transfer(
            geometry="flat_surface",
            dimensions={"length": 1.0, "width": 1.0},
            internal_temperature=373.15,  # 100°C
            ambient_air_temperature=273.15,  # 0°C
            wind_speed=0.1,  # Low wind
            surface_emissivity=0.0,  # No radiation
            wall_layers=[{"thickness": 0.1, "thermal_conductivity_k": 1.0}],
            internal_convection_coefficient_h_override=10000  # Very high
        )
        
        result = json.loads(result_json)
        
        # Expected: Q = k*A*dT/L = 1.0 * 1.0 * 100 / 0.1 = 1000 W
        # With convection resistances, should be slightly less
        expected = 1000
        actual = result["total_heat_rate_loss_w"]
        
        # Should be within 20% (accounting for convection resistance)
        assert 0.8 * expected < actual < expected
        
    def test_radiation_to_space(self):
        """Test radiation to very cold sky."""
        result_json = calculate_surface_heat_transfer(
            geometry="flat_surface",
            dimensions={"length": 1.0, "width": 1.0},
            internal_temperature=300.0,  # 27°C
            ambient_air_temperature=300.0,  # Same as internal (no convection)
            wind_speed=0.0,  # No wind
            surface_emissivity=1.0,  # Black body
            sky_temperature=0.0  # Deep space
        )
        
        result = json.loads(result_json)
        
        # Stefan-Boltzmann: Q = σ * ε * A * T^4
        # σ = 5.67e-8, ε = 1.0, A = 1.0, T = 300K
        expected = 5.67e-8 * 1.0 * 1.0 * (300**4)
        actual = result["radiative_heat_rate_w"]
        
        # Should be close to theoretical
        assert abs(actual - expected) / expected < 0.1


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise run directly
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        # Run tests manually
        test_classes = [
            TestSkyTemperature(),
            TestSurfaceHeatTransfer(),
            TestTankHeatLoss(),
            TestPhysicsValidation()
        ]
        
        for test_class in test_classes:
            print(f"\nTesting {test_class.__class__.__name__}...")
            for method_name in dir(test_class):
                if method_name.startswith("test_"):
                    print(f"  {method_name}...", end=" ")
                    try:
                        getattr(test_class, method_name)()
                        print("✓")
                    except AssertionError as e:
                        print(f"✗ {e}")
                    except Exception as e:
                        print(f"ERROR: {e}")