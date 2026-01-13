"""Tests for plate heat exchanger sizing and pressure drop tools."""
import json
import pytest
import math

from tools.calculate_pressure_drop import calculate_pressure_drop
from tools.plate_heat_exchanger_sizing import size_plate_heat_exchanger
from tools.convection_coefficient import calculate_convection_coefficient


class TestPHEConvectionCoefficient:
    """Test plate_chevron geometry in convection coefficient calculator."""

    def test_plate_chevron_martin_vdi(self):
        """Test PHE convection coefficient with Martin VDI correlation."""
        result_json = calculate_convection_coefficient(
            geometry="plate_chevron",
            characteristic_dimension=0.004,  # 4mm hydraulic diameter
            fluid_name="water",
            bulk_fluid_temperature=298.15,  # 25°C
            surface_temperature=310.15,  # 37°C
            flow_type="forced",
            fluid_velocity=0.5,  # m/s
            chevron_angle=45.0,
            phe_correlation="Martin_VDI",
        )
        result = json.loads(result_json)

        # Should not have error
        assert "error" not in result, f"Unexpected error: {result.get('error')}"

        # Check expected outputs
        assert "convection_coefficient_h" in result
        assert "nusselt_number" in result
        assert "calculation_details" in result

        # Check PHE-specific details
        details = result["calculation_details"]
        assert details["phe_correlation"] == "Martin_VDI"
        assert details["paired_friction_correlation"] == "friction_plate_Martin_VDI"
        assert details["chevron_angle_deg"] == 45.0

        # h should be reasonable for water in PHE (typically 3000-15000 W/m²K)
        h = result["convection_coefficient_h"]
        assert 1000 < h < 20000, f"h={h} outside expected range"

    def test_plate_chevron_kumar(self):
        """Test PHE convection coefficient with Kumar correlation."""
        result_json = calculate_convection_coefficient(
            geometry="plate_chevron",
            characteristic_dimension=0.004,
            fluid_name="water",
            bulk_fluid_temperature=298.15,
            surface_temperature=310.15,
            flow_type="forced",
            fluid_velocity=0.5,
            chevron_angle=45.0,
            phe_correlation="Kumar",
        )
        result = json.loads(result_json)

        assert "error" not in result
        assert result["calculation_details"]["phe_correlation"] == "Kumar"
        assert result["calculation_details"]["paired_friction_correlation"] == "friction_plate_Kumar"

    def test_plate_chevron_muley_manglik_requires_phi(self):
        """Test that Muley_Manglik requires plate_enlargement_factor."""
        result_json = calculate_convection_coefficient(
            geometry="plate_chevron",
            characteristic_dimension=0.004,
            fluid_name="water",
            bulk_fluid_temperature=298.15,
            surface_temperature=310.15,
            flow_type="forced",
            fluid_velocity=0.5,
            chevron_angle=45.0,
            phe_correlation="Muley_Manglik",
            # plate_enlargement_factor not provided
        )
        result = json.loads(result_json)

        # Should return error about missing plate_enlargement_factor
        assert "error" in result
        assert "plate_enlargement_factor" in result["error"]

    def test_plate_chevron_requires_chevron_angle(self):
        """Test that plate_chevron requires chevron_angle."""
        result_json = calculate_convection_coefficient(
            geometry="plate_chevron",
            characteristic_dimension=0.004,
            fluid_name="water",
            bulk_fluid_temperature=298.15,
            surface_temperature=310.15,
            flow_type="forced",
            fluid_velocity=0.5,
            # chevron_angle not provided
        )
        result = json.loads(result_json)

        assert "error" in result
        assert "chevron_angle" in result["error"]


class TestPressureDropCalculation:
    """Test pressure drop calculation tool."""

    def test_phe_pressure_drop_martin_vdi(self):
        """Test PHE pressure drop with Martin VDI correlation."""
        result_json = calculate_pressure_drop(
            geometry="plate_chevron",
            mass_flow_kg_s=0.5,
            fluid_name="water",
            fluid_temperature_K=298.15,
            hydraulic_diameter_m=0.004,
            flow_length_m=0.5,
            flow_area_m2=0.001,  # Per channel
            chevron_angle=45.0,
            n_channels=10,
            n_passes=1,
            port_diameter_m=0.05,
            correlation="Martin_VDI",
        )
        result = json.loads(result_json)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"

        # Check expected outputs
        assert "pressure_drop_Pa" in result
        assert "pressure_drop_kPa" in result
        assert "friction_factor" in result
        assert "reynolds_number" in result
        assert "velocity_m_s" in result
        assert "components" in result

        # Check components breakdown
        components = result["components"]
        assert "frictional_Pa" in components
        assert "port_losses_Pa" in components

        # Pressure drop should be positive and reasonable
        dP = result["pressure_drop_kPa"]
        assert dP > 0, "Pressure drop should be positive"
        assert dP < 500, f"Pressure drop {dP} kPa seems too high"

    def test_pipe_internal_pressure_drop(self):
        """Test internal pipe flow pressure drop."""
        result_json = calculate_pressure_drop(
            geometry="pipe_internal",
            mass_flow_kg_s=1.0,
            fluid_name="water",
            fluid_temperature_K=298.15,
            hydraulic_diameter_m=0.025,  # 25mm ID
            flow_length_m=10.0,
            pipe_roughness_m=0.00005,
        )
        result = json.loads(result_json)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "pressure_drop_Pa" in result
        assert result["correlation_used"] == "fluids.one_phase_dP"

    def test_shell_side_kern_pressure_drop(self):
        """Test shell-side pressure drop with Kern method."""
        result_json = calculate_pressure_drop(
            geometry="shell_side_kern",
            mass_flow_kg_s=5.0,
            fluid_name="water",
            fluid_temperature_K=333.15,  # 60°C
            shell_diameter_m=0.5,
            baffle_spacing_m=0.1,
            tube_pitch_m=0.025,
            tube_od_m=0.019,
            n_baffles=10,
        )
        result = json.loads(result_json)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "pressure_drop_Pa" in result
        assert "dP_Kern" in result["correlation_used"]
        assert "notes" in result  # Should have note about bundle crossflow only


class TestPlateSizing:
    """Test plate heat exchanger sizing tool."""

    def test_20tr_chiller_example(self):
        """Test the 20 TR chiller example from user's original request.

        Inputs:
        - 200 LPM chilled water @ 9°C inlet
        - 20 m³/h hot water @ 32°C inlet
        - 20 TR (~70.34 kW) duty
        """
        # Convert flows to kg/s (assuming water density ~1000 kg/m³)
        cold_flow = 200 / 60000 * 1000  # LPM to kg/s: ~3.33 kg/s
        hot_flow = 20 / 3600 * 1000  # m³/h to kg/s: ~5.56 kg/s

        # Temperatures
        T_cold_in = 273.15 + 9  # 9°C
        T_hot_in = 273.15 + 32  # 32°C

        # Heat duty (20 TR = 20 * 3.517 kW = 70.34 kW)
        Q = 20 * 3517  # W

        result_json = size_plate_heat_exchanger(
            heat_duty_W=Q,
            hot_inlet_temp_K=T_hot_in,
            cold_inlet_temp_K=T_cold_in,
            hot_mass_flow_kg_s=hot_flow,
            cold_mass_flow_kg_s=cold_flow,
            hot_fluid="water",
            cold_fluid="water",
            # Typical PHE geometry
            plate_amplitude_m=0.002,  # 2mm
            plate_wavelength_m=0.008,  # 8mm
            plate_width_m=0.3,
            plate_length_m=0.8,
            port_diameter_m=0.05,
            chevron_angle_deg=45.0,
            correlation="Martin_VDI",
            max_pressure_drop_kPa=50.0,  # 50 kPa limit
        )
        result = json.loads(result_json)

        # Should not have error
        assert "error" not in result, f"Unexpected error: {result.get('error')}"

        # Check that we got a valid sizing
        assert "geometry" in result
        assert "thermal" in result
        assert "hydraulic" in result
        assert "temperatures" in result

        # Verify key outputs
        geometry = result["geometry"]
        thermal = result["thermal"]
        hydraulic = result["hydraulic"]
        temps = result["temperatures"]

        # Plates should be odd (symmetric allocation)
        assert geometry["plates"] % 2 == 1, "Plate count should be odd"

        # Area should be internally consistent
        # A_heat_transfer = (plates - 2) * A_plate_surface
        expected_area = (geometry["plates"] - 2) * geometry["area_per_plate_m2"]
        assert math.isclose(geometry["total_area_m2"], expected_area, rel_tol=0.01), \
            f"Area inconsistency: total={geometry['total_area_m2']}, expected={expected_area}"

        # LMTD should be ~18-19 K for this case
        lmtd = result["LMTD_K"]
        assert 15 < lmtd < 25, f"LMTD {lmtd} K outside expected range"

        # Terminal temperature difference (correct terminology)
        ttd_min = temps["terminal_temp_diff_min_K"]
        assert ttd_min > 0, "Terminal temp diff should be positive (no crossover)"

        # U-value should be reasonable for water-water PHE (2000-6000 W/m²K typical)
        U = thermal["U_W_m2K"]
        assert 1500 < U < 8000, f"U-value {U} outside expected range"

        # Pressure drops should be under constraint
        assert hydraulic["pressure_drop_hot_kPa"] <= 50.0 * 1.1  # 10% tolerance
        assert hydraulic["pressure_drop_cold_kPa"] <= 50.0 * 1.1

        # Correlations should be paired
        assert thermal["correlation"] == "Martin_VDI"
        assert hydraulic["correlation"] == "friction_plate_Martin_VDI"

    def test_missing_required_params(self):
        """Test that missing required parameters return helpful errors."""
        # Missing flow rates
        result_json = size_plate_heat_exchanger(
            heat_duty_W=70000,
            hot_inlet_temp_K=305,
            cold_inlet_temp_K=282,
            # hot_mass_flow_kg_s missing
            cold_mass_flow_kg_s=3.33,
            plate_amplitude_m=0.002,
            plate_wavelength_m=0.008,
            plate_width_m=0.3,
            plate_length_m=0.8,
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "hot_mass_flow_kg_s" in result["error"]

    def test_temperature_crossover_detected(self):
        """Test that temperature crossover is detected and reported."""
        result_json = size_plate_heat_exchanger(
            heat_duty_W=70000,
            hot_inlet_temp_K=300,  # Hot side too cold
            hot_outlet_temp_K=290,
            cold_inlet_temp_K=295,  # Cold side hotter than hot outlet
            cold_outlet_temp_K=305,  # Would cross hot inlet
            hot_mass_flow_kg_s=5.0,
            cold_mass_flow_kg_s=3.33,
            plate_amplitude_m=0.002,
            plate_wavelength_m=0.008,
            plate_width_m=0.3,
            plate_length_m=0.8,
        )
        result = json.loads(result_json)
        assert "error" in result
        assert "crossover" in result["error"].lower() or "undefined" in result["error"].lower()


class TestCorrelationPairing:
    """Test that Nu and friction correlations are properly paired."""

    @pytest.mark.parametrize("correlation,expected_friction", [
        ("Kumar", "friction_plate_Kumar"),
        ("Martin_1999", "friction_plate_Martin_1999"),
        ("Martin_VDI", "friction_plate_Martin_VDI"),
    ])
    def test_correlation_pairing(self, correlation, expected_friction):
        """Test that each Nu correlation is paired with correct friction correlation."""
        result_json = calculate_convection_coefficient(
            geometry="plate_chevron",
            characteristic_dimension=0.004,
            fluid_name="water",
            bulk_fluid_temperature=298.15,
            surface_temperature=310.15,
            flow_type="forced",
            fluid_velocity=0.5,
            chevron_angle=45.0,
            phe_correlation=correlation,
        )
        result = json.loads(result_json)

        assert "error" not in result
        assert result["calculation_details"]["paired_friction_correlation"] == expected_friction


class TestOddPlateEnforcement:
    """Test that plate counts are enforced to be odd for symmetric allocation."""

    def test_even_plate_count_adjusted(self):
        """Test that even plate count is adjusted to odd."""
        result_json = size_plate_heat_exchanger(
            heat_duty_W=50000,
            hot_inlet_temp_K=305,
            cold_inlet_temp_K=282,
            hot_mass_flow_kg_s=5.0,
            cold_mass_flow_kg_s=3.33,
            plate_amplitude_m=0.002,
            plate_wavelength_m=0.008,
            plate_width_m=0.3,
            plate_length_m=0.8,
            n_plates=20,  # Even - should be adjusted to 21
        )
        result = json.loads(result_json)

        if "error" not in result:
            assert result["geometry"]["plates"] % 2 == 1, "Plate count should be odd"


class TestAutoOptimize:
    """Test auto-optimization functionality."""

    def test_auto_optimize_finds_feasible_design(self):
        """Test that auto_optimize searches across parameters to find optimal design."""
        result_json = size_plate_heat_exchanger(
            heat_duty_W=70000,
            hot_inlet_temp_K=305.15,
            cold_inlet_temp_K=282.15,
            hot_mass_flow_kg_s=5.56,
            cold_mass_flow_kg_s=3.33,
            plate_amplitude_m=0.002,
            plate_wavelength_m=0.008,
            plate_width_m=0.3,
            plate_length_m=0.8,
            auto_optimize=True,
            optimize_for="area",
        )
        result = json.loads(result_json)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result.get("optimization_mode") is True
        assert "best_configuration" in result
        assert "best_result" in result
        assert "top_5_candidates" in result
        assert len(result["top_5_candidates"]) > 0

        # Best result should have valid structure
        best = result["best_result"]
        assert "duty_W" in best
        assert "thermal" in best
        assert "hydraulic" in best
        assert "geometry" in best

    def test_auto_optimize_respects_pressure_constraint(self):
        """Test that auto_optimize respects max_pressure_drop_kPa."""
        result_json = size_plate_heat_exchanger(
            heat_duty_W=70000,
            hot_inlet_temp_K=305.15,
            cold_inlet_temp_K=282.15,
            hot_mass_flow_kg_s=5.56,
            cold_mass_flow_kg_s=3.33,
            plate_amplitude_m=0.002,
            plate_wavelength_m=0.008,
            plate_width_m=0.3,
            plate_length_m=0.8,
            max_pressure_drop_kPa=50.0,
            auto_optimize=True,
        )
        result = json.loads(result_json)

        if "error" not in result and result.get("optimization_mode"):
            best = result["best_result"]
            assert best["hydraulic"]["pressure_drop_hot_kPa"] <= 50.0 * 1.05
            assert best["hydraulic"]["pressure_drop_cold_kPa"] <= 50.0 * 1.05
