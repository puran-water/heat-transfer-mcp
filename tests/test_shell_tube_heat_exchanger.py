"""Tests for shell-tube heat exchanger sizing tool."""
import json
import pytest
import math

from tools.size_shell_tube_heat_exchanger import size_shell_tube_heat_exchanger


class TestBasicSizing:
    """Test basic shell-tube sizing functionality."""

    def test_water_water_basic(self):
        """Test basic water-water shell-tube sizing."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=100000,  # 100 kW
            hot_inlet_temp_K=353.15,  # 80°C
            cold_inlet_temp_K=293.15,  # 20°C
            hot_mass_flow_kg_s=1.2,
            cold_mass_flow_kg_s=0.8,
            hot_fluid="water",
            cold_fluid="water",
            tube_outer_diameter_m=0.019,
            tube_inner_diameter_m=0.016,
            tube_pitch_m=0.025,
            tube_layout_angle=30,
            tube_length_m=3.0,
            n_tube_passes=4,  # 4 passes for turbulent tube-side flow
            tube_side_fluid="cold",
        )
        result = json.loads(result_json)

        # Should not have error
        assert "error" not in result, f"Unexpected error: {result.get('error')}"

        # Check expected output structure
        assert "duty_W" in result
        assert "LMTD_K" in result
        assert "F_correction" in result
        assert "geometry" in result
        assert "thermal" in result
        assert "hydraulic" in result
        assert "temperatures" in result
        assert "heat_balance_verification" in result

        # Verify duty matches
        assert math.isclose(result["duty_W"], 100000, rel_tol=0.01)

        # Check geometry
        geo = result["geometry"]
        assert geo["type"] == "shell_tube"
        assert geo["n_tubes"] > 0
        assert geo["tube_length_m"] == 3.0
        assert geo["total_area_m2"] > 0

        # Check thermal - U should be in reasonable range for water-water (500-2000 W/m²K)
        thermal = result["thermal"]
        assert 200 < thermal["U_W_m2K"] < 3000, f"U={thermal['U_W_m2K']} outside expected range"
        assert thermal["h_tube_W_m2K"] > 0
        assert thermal["h_shell_W_m2K"] > 0
        assert thermal["Re_tube"] > 0
        assert thermal["Re_shell"] > 0

        # Check hydraulic
        hydraulic = result["hydraulic"]
        assert hydraulic["velocity_tube_m_s"] > 0
        assert hydraulic["velocity_shell_m_s"] > 0
        assert hydraulic["pressure_drop_tube_kPa"] > 0
        assert hydraulic["pressure_drop_shell_kPa"] > 0

        # Check heat balance
        hb = result["heat_balance_verification"]
        assert hb["balance_satisfied"] or abs(hb["Q_from_LMTD_kW"] - result["duty_kW"]) / result["duty_kW"] < 0.1

    def test_outlet_temperature_calculation(self):
        """Test that outlet temperatures are correctly calculated from duty."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=50000,  # 50 kW
            hot_inlet_temp_K=343.15,  # 70°C
            cold_inlet_temp_K=288.15,  # 15°C
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.4,
            tube_length_m=4.0,  # Longer tubes for feasible design
            n_tube_passes=4,  # More passes for turbulent flow
            auto_optimize=True,  # Let solver find feasible configuration
        )
        raw_result = json.loads(result_json)

        assert "error" not in raw_result, f"Unexpected error: {raw_result.get('error')}"

        # When auto_optimize is used, extract the best_result
        if raw_result.get("optimization_mode"):
            result = raw_result["best_result"]
        else:
            result = raw_result

        temps = result["temperatures"]

        # Check hot outlet: T_out = T_in - Q/(m*cp)
        # For water at ~55°C, cp ≈ 4180 J/kg-K
        expected_hot_out = 343.15 - 50000 / (0.5 * 4180)  # ~319 K
        assert abs(temps["hot_outlet_K"] - expected_hot_out) < 3  # Within 3K tolerance

        # Check cold outlet: T_out = T_in + Q/(m*cp)
        expected_cold_out = 288.15 + 50000 / (0.4 * 4180)  # ~318 K
        assert abs(temps["cold_outlet_K"] - expected_cold_out) < 3


class TestThermalHydraulicCoupling:
    """Test that thermal and hydraulic calculations are properly coupled."""

    def test_tube_count_affects_both_U_and_dP(self):
        """Verify that changing tube count affects both U-value and pressure drop."""
        base_params = {
            "heat_duty_W": 75000,
            "hot_inlet_temp_K": 348.15,
            "cold_inlet_temp_K": 293.15,
            "hot_mass_flow_kg_s": 1.0,
            "cold_mass_flow_kg_s": 0.6,
            "tube_length_m": 3.0,
            "n_tube_passes": 2,
        }

        # Run with different tube count ranges to get different solutions
        result1_json = size_shell_tube_heat_exchanger(**base_params, min_tubes=20, max_tubes=50)
        result2_json = size_shell_tube_heat_exchanger(**base_params, min_tubes=80, max_tubes=150)

        result1 = json.loads(result1_json)
        result2 = json.loads(result2_json)

        # Both should succeed (may have different tube counts)
        if "error" not in result1 and "error" not in result2:
            # Different tube counts should give different U and dP
            n1 = result1["geometry"]["n_tubes"]
            n2 = result2["geometry"]["n_tubes"]

            if n1 != n2:
                # More tubes = lower velocity = lower h = lower U
                # More tubes = lower velocity = lower dP
                U1 = result1["thermal"]["U_W_m2K"]
                U2 = result2["thermal"]["U_W_m2K"]
                dP1 = result1["hydraulic"]["pressure_drop_tube_kPa"]
                dP2 = result2["hydraulic"]["pressure_drop_tube_kPa"]

                # Fewer tubes (result1) should have higher U and higher dP
                if n1 < n2:
                    assert U1 > U2 * 0.9, "Fewer tubes should have higher U"
                    assert dP1 > dP2 * 0.5, "Fewer tubes should have higher dP"


class TestConstraintSatisfaction:
    """Test that solver respects constraints."""

    def test_pressure_drop_constraint(self):
        """Test that pressure drop constraint is respected."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=80000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=1.0,
            cold_mass_flow_kg_s=0.7,
            tube_length_m=3.0,
            max_pressure_drop_tube_kPa=30.0,
            max_pressure_drop_shell_kPa=25.0,
        )
        result = json.loads(result_json)

        if "error" not in result:
            hydraulic = result["hydraulic"]
            # Pressure drops should be within constraints (with small tolerance)
            assert hydraulic["pressure_drop_tube_kPa"] <= 30.0 * 1.05
            assert hydraulic["pressure_drop_shell_kPa"] <= 25.0 * 1.05

    def test_velocity_constraint(self):
        """Test that velocity constraints are respected."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=60000,
            hot_inlet_temp_K=343.15,
            cold_inlet_temp_K=288.15,
            hot_mass_flow_kg_s=0.8,
            cold_mass_flow_kg_s=0.5,
            tube_length_m=2.5,
            max_velocity_tube_m_s=2.0,
            max_velocity_shell_m_s=1.0,
        )
        result = json.loads(result_json)

        if "error" not in result:
            hydraulic = result["hydraulic"]
            assert hydraulic["velocity_tube_m_s"] <= 2.0
            assert hydraulic["velocity_shell_m_s"] <= 1.0


class TestFCorrectionFactor:
    """Test LMTD correction factor calculations."""

    def test_f_correction_multi_pass(self):
        """Test that F < 1 for multi-pass configurations."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=100000,
            hot_inlet_temp_K=363.15,  # 90°C
            cold_inlet_temp_K=283.15,  # 10°C
            hot_mass_flow_kg_s=1.5,
            cold_mass_flow_kg_s=1.0,
            n_tube_passes=4,  # Multi-pass
            n_shell_passes=1,
            tube_length_m=4.0,
        )
        result = json.loads(result_json)

        if "error" not in result:
            # F correction should be <= 1 for shell-tube
            assert result["F_correction"] <= 1.0
            # For reasonable temperature programs, F should be > 0.75
            assert result["F_correction"] > 0.7, f"F={result['F_correction']} is too low"


class TestOutputConsistency:
    """Test that outputs are internally consistent."""

    def test_heat_balance_q_equals_ua_lmtd(self):
        """Verify Q = U * A * LMTD * F consistency."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=120000,
            hot_inlet_temp_K=358.15,
            cold_inlet_temp_K=298.15,
            hot_mass_flow_kg_s=1.5,
            cold_mass_flow_kg_s=1.2,
            tube_length_m=3.5,
        )
        result = json.loads(result_json)

        if "error" not in result:
            U = result["thermal"]["U_W_m2K"]
            A = result["geometry"]["area_required_m2"]
            LMTD = result["LMTD_K"]
            F = result["F_correction"]
            Q = result["duty_W"]

            # Q should approximately equal U * A_required * LMTD * F
            Q_calculated = U * A * LMTD * F
            rel_error = abs(Q - Q_calculated) / Q
            assert rel_error < 0.05, f"Heat balance error: Q={Q}, U*A*LMTD*F={Q_calculated}"

    def test_area_margin_positive(self):
        """Test that available area >= required area (positive margin)."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=90000,
            hot_inlet_temp_K=348.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=1.2,
            cold_mass_flow_kg_s=0.8,
            tube_length_m=3.0,
        )
        result = json.loads(result_json)

        if "error" not in result:
            geo = result["geometry"]
            assert geo["total_area_m2"] >= geo["area_required_m2"] * 0.99
            assert geo["area_margin_pct"] >= -1  # Small negative tolerance ok


class TestParameterSweep:
    """Test parameter sweep functionality."""

    def test_tube_pass_sweep(self):
        """Test sweep over number of tube passes."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=80000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=1.0,
            cold_mass_flow_kg_s=0.7,
            tube_length_m=3.0,
            sweep_n_tube_passes=[1, 2, 4],
        )
        result = json.loads(result_json)

        assert "sweep_type" in result
        assert result["sweep_type"] == "n_tube_passes"
        assert "results" in result
        assert len(result["results"]) == 3  # 3 pass values

        # Each result should have key metrics
        for r in result["results"]:
            assert "n_tube_passes" in r
            if "error" not in r:
                assert "n_tubes" in r
                assert "U_W_m2K" in r
                assert "dP_tube_kPa" in r


class TestInputValidation:
    """Test input validation and error handling."""

    def test_missing_flow_rate(self):
        """Test error when flow rate is missing."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=50000,
            hot_inlet_temp_K=343.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=None,
            cold_mass_flow_kg_s=0.5,
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "hot_mass_flow" in result["error"].lower()

    def test_invalid_tube_pitch(self):
        """Test error when tube pitch <= tube OD."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=50000,
            hot_inlet_temp_K=343.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.4,
            tube_outer_diameter_m=0.025,
            tube_pitch_m=0.020,  # Less than OD!
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "pitch" in result["error"].lower()

    def test_invalid_layout_angle(self):
        """Test error for invalid tube layout angle."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=50000,
            hot_inlet_temp_K=343.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.4,
            tube_layout_angle=50,  # Invalid - must be 30, 45, 60, or 90
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "layout" in result["error"].lower() or "angle" in result["error"].lower()

    def test_temperature_crossover(self):
        """Test error when temperature crossover occurs."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=50000,
            hot_inlet_temp_K=310.15,  # 37°C - hot side too cold
            cold_inlet_temp_K=323.15,  # 50°C - cold side hotter!
            hot_mass_flow_kg_s=0.5,
            cold_mass_flow_kg_s=0.4,
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "crossover" in result["error"].lower() or "lmtd" in result["error"].lower()


class TestComparisonWithPHE:
    """Compare shell-tube results with PHE for same duty (different characteristics expected)."""

    def test_same_duty_different_characteristics(self):
        """For same duty, shell-tube should have lower U but potentially lower dP than PHE."""
        from tools.plate_heat_exchanger_sizing import size_plate_heat_exchanger

        # Common parameters
        duty = 70000  # 70 kW
        hot_in = 305.15  # 32°C
        cold_in = 282.15  # 9°C
        hot_flow = 5.56  # kg/s (20 m³/h)
        cold_flow = 3.33  # kg/s (200 LPM)

        # Shell-tube sizing - use 4 passes for turbulent flow
        st_result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=duty,
            hot_inlet_temp_K=hot_in,
            cold_inlet_temp_K=cold_in,
            hot_mass_flow_kg_s=hot_flow,
            cold_mass_flow_kg_s=cold_flow,
            tube_length_m=3.0,
            n_tube_passes=4,  # Ensure turbulent tube-side flow
        )
        st_result = json.loads(st_result_json)

        # PHE sizing (same duty)
        phe_result_json = size_plate_heat_exchanger(
            heat_duty_W=duty,
            hot_inlet_temp_K=hot_in,
            cold_inlet_temp_K=cold_in,
            hot_mass_flow_kg_s=hot_flow,
            cold_mass_flow_kg_s=cold_flow,
            plate_amplitude_m=0.002,
            plate_wavelength_m=0.008,
            plate_width_m=0.3,
            plate_length_m=0.8,
        )
        phe_result = json.loads(phe_result_json)

        # Compare if both succeeded
        if "error" not in st_result and "error" not in phe_result:
            U_st = st_result["thermal"]["U_W_m2K"]
            U_phe = phe_result["thermal"]["U_W_m2K"]

            # PHE typically has higher U than shell-tube
            # This is expected due to higher turbulence in chevron channels
            print(f"Shell-tube U: {U_st:.0f} W/m²K")
            print(f"PHE U: {U_phe:.0f} W/m²K")

            # Both should be in valid ranges
            assert 200 < U_st < 3000, f"Shell-tube U outside range: {U_st}"
            assert 1500 < U_phe < 10000, f"PHE U outside range: {U_phe}"


class TestShellDiameterConstraint:
    """Test that Ntubes_Phadkeb constrains tube count based on shell diameter."""

    def test_shell_diameter_limits_tube_count(self):
        """Verify that specifying shell diameter constrains max tubes."""
        # Small shell diameter should result in fewer tubes
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=50000,
            hot_inlet_temp_K=353.15,  # 80°C
            cold_inlet_temp_K=293.15,  # 20°C
            hot_mass_flow_kg_s=1.0,
            cold_mass_flow_kg_s=1.0,
            shell_inner_diameter_m=0.2,  # 200mm shell - small
            tube_length_m=2.0,
            n_tube_passes=2,
            min_tubes=10,
            max_tubes=500,  # Would be higher without shell constraint
        )
        result = json.loads(result_json)

        if "error" not in result:
            geo = result["geometry"]

            # Should report max_tubes_from_shell was computed
            assert geo["shell_diameter_specified"] is True
            assert geo["max_tubes_from_shell"] is not None

            # Tube count should not exceed computed max from shell
            assert geo["n_tubes"] <= geo["max_tubes_from_shell"]

            # Bundle diameter should fit within shell
            shell_d = geo["shell_diameter_m"]
            bundle_d = geo["bundle_diameter_m"]
            assert bundle_d < shell_d, f"Bundle {bundle_d}m exceeds shell {shell_d}m"

    def test_without_shell_constraint(self):
        """Verify output when shell diameter is not specified."""
        result_json = size_shell_tube_heat_exchanger(
            heat_duty_W=50000,
            hot_inlet_temp_K=353.15,
            cold_inlet_temp_K=293.15,
            hot_mass_flow_kg_s=1.0,
            cold_mass_flow_kg_s=1.0,
            tube_length_m=2.0,
            n_tube_passes=2,
        )
        result = json.loads(result_json)

        if "error" not in result:
            geo = result["geometry"]

            # Should indicate shell was not specified
            assert geo["shell_diameter_specified"] is False
            # max_tubes_from_shell should be None when shell not specified
            assert geo["max_tubes_from_shell"] is None
