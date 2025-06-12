"""Test unit conversion functionality."""
import pytest
from utils import unit_converter


class TestUnitConverter:
    def setup_method(self):
        # No need for a converter instance, using module functions directly
        pass

    def test_temperature_conversion(self):
        """Test temperature conversions."""
        # Fahrenheit to Kelvin
        assert abs(unit_converter.fahrenheit_to_kelvin(95) - 308.15) < 0.1
        assert abs(unit_converter.fahrenheit_to_kelvin(32) - 273.15) < 0.1

        # Celsius to Kelvin
        assert abs(unit_converter.celsius_to_kelvin(25) - 298.15) < 0.1
        assert abs(unit_converter.celsius_to_kelvin(0) - 273.15) < 0.1

    def test_flow_rate_conversion(self):
        """Test flow rate conversions."""
        # GPM to m3/s then to kg/s (approximation for water)
        gpm_result = unit_converter.gpm_to_m3_per_s(200) * 1000  # Convert to kg/s assuming water density ~1000 kg/m3
        assert 12.0 < gpm_result < 13.0

        # SCFM to kg/s
        scfm_result = unit_converter.scfm_to_kg_per_s(5000)
        assert 2.5 < scfm_result < 3.5

    def test_length_conversion(self):
        """Test length conversions."""
        # Feet to meters
        assert abs(unit_converter.feet_to_meters(10) - 3.048) < 0.001

        # Inches to meters
        assert abs(unit_converter.inches_to_meters(12) - 0.3048) < 0.001

    def test_pressure_conversion(self):
        """Test pressure conversions."""
        # PSI to Pa
        psi_result = unit_converter.psi_to_pascal(14.7)
        assert abs(psi_result - 101325) < 1000  # Within 1 kPa

        # Bar to Pa
        bar_result = unit_converter.bar_to_pascal(1)
        assert abs(bar_result - 100000) < 100

    def test_power_conversion(self):
        """Test power conversions."""
        # BTU/hr to Watts
        btu_result = unit_converter.btu_per_hr_to_watts(1000)
        assert 290 < btu_result < 295

        # HP to Watts
        hp_result = unit_converter.hp_to_watts(10)
        assert 7400 < hp_result < 7500

    def test_parse_and_convert(self):
        """Test the parse_and_convert function."""
        # Test temperature conversion with string input
        result = unit_converter.parse_and_convert("95 degF", "K", "temperature")
        assert abs(result - 308.15) < 0.1
        
        # Test invalid unit handling
        with pytest.raises(Exception):
            unit_converter.parse_and_convert("invalid unit", "K", "temperature")

    def test_numeric_passthrough(self):
        """Test that numeric values pass through when already in target units."""
        # parse_and_convert should handle numeric values
        result = unit_converter.parse_and_convert(298.15, "K", "temperature")
        assert result == 298.15