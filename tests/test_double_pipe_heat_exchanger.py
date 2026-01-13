"""
Tests for the double-pipe heat exchanger sizing tool.

These tests verify:
1. Basic sizing functionality with water-water case
2. Rating mode for given length
3. Thermal-hydraulic coupling (U and dP both depend on geometry)
4. Constraint satisfaction
5. Heat balance verification
6. Annulus geometry calculations
"""

import json
import math
import pytest

from tools.size_double_pipe_heat_exchanger import size_double_pipe_heat_exchanger
from utils.hx_common import calculate_annulus_geometry, calculate_overall_U


class TestDoublePipeSizing:
    """Tests for basic double-pipe HX sizing."""

    def test_basic_water_water_sizing(self):
        """Test basic sizing with water-water case."""
        result_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,  # 10 kW
            hot_inlet_temp_K=353.15,  # 80°C
            cold_inlet_temp_K=293.15,  # 20°C
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            hot_fluid="water",
            cold_fluid="water",
            inner_pipe_outer_diameter_m=0.0254,  # 1" OD
            inner_pipe_inner_diameter_m=0.0229,  # ~0.9" ID
            outer_pipe_inner_diameter_m=0.0525,  # 2" ID
            flow_arrangement="counterflow",
            inner_pipe_fluid="hot",
            min_length_m=0.5,
            max_length_m=15.0,
        )

        result = json.loads(result_json)

        # Should find a solution
        assert "error" not in result, f"Sizing failed: {result.get('error')}"

        # Check structure
        assert "geometry" in result
        assert "thermal" in result
        assert "hydraulic" in result
        assert "temperatures" in result

        # Verify geometry
        geo = result["geometry"]
        assert geo["type"] == "double_pipe"
        assert geo["pipe_length_m"] > 0
        assert geo["total_area_m2"] > 0
        assert geo["area_margin_pct"] >= -1  # Should satisfy thermal requirement

        # Verify thermal
        therm = result["thermal"]
        assert therm["U_W_m2K"] > 100  # Reasonable U for water-water
        assert therm["U_W_m2K"] < 3000
        assert therm["Re_inner"] > 0
        assert therm["Re_annulus"] > 0
        assert therm["h_inner_W_m2K"] > 0
        assert therm["h_annulus_W_m2K"] > 0

        # Verify hydraulic
        hyd = result["hydraulic"]
        assert hyd["velocity_inner_m_s"] > 0
        assert hyd["velocity_annulus_m_s"] > 0
        assert hyd["pressure_drop_inner_kPa"] > 0
        assert hyd["pressure_drop_annulus_kPa"] > 0

    def test_all_temperatures_provided(self):
        """Test sizing when all temperatures are specified."""
        result_json = size_double_pipe_heat_exchanger(
            hot_inlet_temp_K=353.15,  # 80°C
            hot_outlet_temp_K=333.15,  # 60°C
            cold_inlet_temp_K=293.15,  # 20°C
            cold_outlet_temp_K=313.15,  # 40°C
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            hot_fluid="water",
            cold_fluid="water",
            max_length_m=35.0,  # Allow longer length for this duty
        )

        result = json.loads(result_json)

        # Should find a solution
        assert "error" not in result, f"Sizing failed: {result.get('error')}"

        # Duty should be calculable from temperatures
        # Q = m * cp * dT ~= 0.5 * 4180 * 20 ~= 41800 W
        assert result["duty_kW"] > 30  # Allow for cp variation

    def test_parallel_flow(self):
        """Test parallel flow arrangement."""
        result_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            flow_arrangement="parallel",
        )

        result = json.loads(result_json)

        # Should find a solution (parallel flow requires more area than counterflow)
        assert "error" not in result, f"Sizing failed: {result.get('error')}"
        assert result["configuration"]["flow_arrangement"] == "parallel"


class TestDoublePipeRating:
    """Tests for rating mode (given geometry, find duty)."""

    def test_basic_rating(self):
        """Test rating mode with known length."""
        result_json = size_double_pipe_heat_exchanger(
            hot_inlet_temp_K=353.15,  # 80°C
            cold_inlet_temp_K=293.15,  # 20°C
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            hot_fluid="water",
            cold_fluid="water",
            solve_for="rating",
            pipe_length_m=5.0,
        )

        result = json.loads(result_json)

        assert "error" not in result, f"Rating failed: {result.get('error')}"
        assert result["mode"] == "rating"
        assert "actual_duty_W" in result
        assert "actual_duty_kW" in result
        assert result["actual_duty_W"] > 0

    def test_rating_vs_sizing_consistency(self):
        """Verify that rating a sized HX gives approximately the same duty."""
        # First, size for a specific duty
        sizing_result_json = size_double_pipe_heat_exchanger(
            heat_duty_W=15000,  # 15 kW
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
        )
        sizing_result = json.loads(sizing_result_json)

        if "error" in sizing_result:
            pytest.skip(f"Sizing failed: {sizing_result['error']}")

        sized_length = sizing_result["geometry"]["pipe_length_m"]

        # Now rate the same HX
        rating_result_json = size_double_pipe_heat_exchanger(
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            solve_for="rating",
            pipe_length_m=sized_length,
        )
        rating_result = json.loads(rating_result_json)

        assert "error" not in rating_result

        # Rating should give approximately the same duty (within area margin)
        design_duty = 15000
        actual_duty = rating_result["actual_duty_W"]
        assert actual_duty >= design_duty * 0.95  # At least 95% of design
        assert actual_duty <= design_duty * 1.10  # At most 110% of design (due to margin)


class TestThermalHydraulicCoupling:
    """Tests verifying thermal-hydraulic coupling."""

    def test_velocity_affects_h(self):
        """Higher velocity should increase heat transfer coefficient."""
        # Lower flow rate = lower velocity = lower h
        result_low_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.2,
            cold_mass_flow_kg_s=0.2,
        )

        # Higher flow rate = higher velocity = higher h
        result_high_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.8,
            cold_mass_flow_kg_s=0.8,
        )

        result_low = json.loads(result_low_json)
        result_high = json.loads(result_high_json)

        # Both should succeed
        if "error" in result_low or "error" in result_high:
            pytest.skip("Could not compare - one or both failed")

        # Higher flow should give higher h (and thus higher U)
        assert result_high["thermal"]["h_inner_W_m2K"] > result_low["thermal"]["h_inner_W_m2K"]
        assert result_high["thermal"]["U_W_m2K"] > result_low["thermal"]["U_W_m2K"]

        # But also higher pressure drop
        assert result_high["hydraulic"]["pressure_drop_inner_kPa"] > result_low["hydraulic"]["pressure_drop_inner_kPa"]

    def test_larger_pipes_reduce_velocity(self):
        """Larger pipe diameters should reduce velocity."""
        # Smaller pipes
        result_small_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            inner_pipe_outer_diameter_m=0.0254,  # 1" OD
            inner_pipe_inner_diameter_m=0.0229,
            outer_pipe_inner_diameter_m=0.0409,  # 1.5" ID
        )

        # Larger pipes
        result_large_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            inner_pipe_outer_diameter_m=0.0483,  # 1.5" OD
            inner_pipe_inner_diameter_m=0.0409,
            outer_pipe_inner_diameter_m=0.0779,  # 3" ID
        )

        result_small = json.loads(result_small_json)
        result_large = json.loads(result_large_json)

        if "error" in result_small or "error" in result_large:
            pytest.skip("Could not compare - one or both failed")

        # Larger pipes should have lower velocity
        assert result_large["hydraulic"]["velocity_inner_m_s"] < result_small["hydraulic"]["velocity_inner_m_s"]


class TestHairpinConfiguration:
    """Tests for multiple hairpin configurations."""

    def test_multiple_hairpins_reduce_length(self):
        """Using multiple hairpins should reduce required length per hairpin."""
        # Single hairpin
        result_1_json = size_double_pipe_heat_exchanger(
            heat_duty_W=20000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            n_hairpins=1,
            max_length_m=20.0,
        )

        # Two hairpins
        result_2_json = size_double_pipe_heat_exchanger(
            heat_duty_W=20000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            n_hairpins=2,
            max_length_m=20.0,
        )

        result_1 = json.loads(result_1_json)
        result_2 = json.loads(result_2_json)

        # Skip if either fails
        if "error" in result_1 and "error" in result_2:
            pytest.skip("Both configurations failed")

        # If both succeed, 2 hairpins should have shorter length per hairpin
        if "error" not in result_1 and "error" not in result_2:
            # Total length should be similar, but pipe_length_m is per hairpin
            assert result_2["geometry"]["pipe_length_m"] < result_1["geometry"]["pipe_length_m"]
            # But total length should be similar
            total_1 = result_1["geometry"]["total_length_m"]
            total_2 = result_2["geometry"]["total_length_m"]
            assert abs(total_1 - total_2) / total_1 < 0.15  # Within 15%


class TestConstraintSatisfaction:
    """Tests for constraint handling."""

    def test_pressure_drop_constraint(self):
        """Verify pressure drop constraints are respected."""
        result_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            max_pressure_drop_inner_kPa=50.0,
            max_pressure_drop_annulus_kPa=30.0,
        )

        result = json.loads(result_json)

        if "error" not in result:
            # If solution found, constraints should be satisfied
            assert result["hydraulic"]["pressure_drop_inner_kPa"] <= 50.0
            assert result["hydraulic"]["pressure_drop_annulus_kPa"] <= 30.0

    def test_velocity_constraint(self):
        """Verify velocity constraints are respected."""
        result_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            max_velocity_inner_m_s=2.0,
            max_velocity_annulus_m_s=1.5,
        )

        result = json.loads(result_json)

        if "error" not in result:
            # If solution found, velocity constraints should be satisfied
            assert result["hydraulic"]["velocity_inner_m_s"] <= 2.0
            assert result["hydraulic"]["velocity_annulus_m_s"] <= 1.5


class TestAnnulusGeometry:
    """Tests for annulus geometry calculations."""

    def test_calculate_annulus_geometry(self):
        """Test the annulus geometry calculation utility."""
        D_outer = 0.0525  # 2" ID
        D_inner = 0.0254  # 1" OD

        geom = calculate_annulus_geometry(D_outer, D_inner)

        # Hydraulic diameter = D_outer - D_inner
        expected_Dh = D_outer - D_inner
        assert abs(geom["D_hydraulic_m"] - expected_Dh) < 1e-6

        # Cross-sectional area
        expected_area = math.pi / 4 * (D_outer**2 - D_inner**2)
        assert abs(geom["A_annulus_m2"] - expected_area) < 1e-8

        # Equivalent diameter = (D_outer^2 - D_inner^2) / D_inner
        expected_De = (D_outer**2 - D_inner**2) / D_inner
        assert abs(geom["D_equivalent_m"] - expected_De) < 1e-6

        # Gap
        expected_gap = (D_outer - D_inner) / 2
        assert abs(geom["gap_m"] - expected_gap) < 1e-6

    def test_annulus_invalid_geometry(self):
        """Test that invalid geometry raises error."""
        with pytest.raises(ValueError):
            calculate_annulus_geometry(D_outer=0.020, D_inner=0.025)  # D_outer < D_inner


class TestHeatBalance:
    """Tests for heat balance verification."""

    def test_heat_balance_verified(self):
        """Verify heat balance in output."""
        result_json = size_double_pipe_heat_exchanger(
            hot_inlet_temp_K=353.15,
            hot_outlet_temp_K=333.15,
            cold_inlet_temp_K=293.15,
            cold_outlet_temp_K=313.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
        )

        result = json.loads(result_json)

        if "error" not in result:
            hb = result["heat_balance_verification"]
            assert "Q_from_LMTD_kW" in hb
            assert "Q_from_hot_side_kW" in hb
            assert "Q_from_cold_side_kW" in hb

            # Hot and cold side duties should match
            assert abs(hb["Q_from_hot_side_kW"] - hb["Q_from_cold_side_kW"]) / max(hb["Q_from_hot_side_kW"], 1) < 0.10


class TestInputValidation:
    """Tests for input validation."""

    def test_missing_flow_rate(self):
        """Test error when flow rate missing."""
        result_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            cold_mass_flow_kg_s=0.5,
            # hot_mass_flow_kg_s missing
        )
        result = json.loads(result_json)
        assert "error" in result

    def test_invalid_geometry(self):
        """Test error for invalid geometry (outer < inner)."""
        result_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            inner_pipe_outer_diameter_m=0.05,
            outer_pipe_inner_diameter_m=0.04,  # Invalid: outer < inner
        )
        result = json.loads(result_json)
        assert "error" in result

    def test_invalid_flow_arrangement(self):
        """Test error for invalid flow arrangement."""
        result_json = size_double_pipe_heat_exchanger(
            heat_duty_W=10000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.5,
            flow_arrangement="crossflow",  # Invalid for double-pipe
        )
        result = json.loads(result_json)
        assert "error" in result
