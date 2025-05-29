"""Test unit conversion functionality."""
import pytest
from utils.unit_converter import UnitConverter


class TestUnitConverter:
    def setup_method(self):
        self.converter = UnitConverter()
    
    def test_temperature_conversion(self):
        """Test temperature conversions."""
        # Fahrenheit to Kelvin
        assert abs(self.converter.convert_to_si("95 degF", "temperature") - 308.15) < 0.1
        assert abs(self.converter.convert_to_si("32 degF", "temperature") - 273.15) < 0.1
        
        # Celsius to Kelvin
        assert abs(self.converter.convert_to_si("25 degC", "temperature") - 298.15) < 0.1
        assert abs(self.converter.convert_to_si("0 degC", "temperature") - 273.15) < 0.1
    
    def test_flow_rate_conversion(self):
        """Test flow rate conversions."""
        # GPM to kg/s (approximation for water)
        gpm_result = self.converter.convert_to_si("200 GPM", "flow_rate")
        assert 12.0 < gpm_result < 13.0
        
        # SCFM to kg/s
        scfm_result = self.converter.convert_to_si("5000 SCFM", "flow_rate")
        assert 2.5 < scfm_result < 3.5
    
    def test_length_conversion(self):
        """Test length conversions."""
        # Feet to meters
        assert abs(self.converter.convert_to_si("10 feet", "length") - 3.048) < 0.001
        
        # Inches to meters
        assert abs(self.converter.convert_to_si("12 inches", "length") - 0.3048) < 0.001
    
    def test_pressure_conversion(self):
        """Test pressure conversions."""
        # PSI to Pa
        psi_result = self.converter.convert_to_si("14.7 psi", "pressure")
        assert abs(psi_result - 101325) < 1000  # Within 1 kPa
        
        # Bar to Pa
        bar_result = self.converter.convert_to_si("1 bar", "pressure")
        assert abs(bar_result - 100000) < 100
    
    def test_power_conversion(self):
        """Test power conversions."""
        # BTU/hr to Watts
        btu_result = self.converter.convert_to_si("1000 BTU/hr", "power")
        assert 290 < btu_result < 295
        
        # HP to Watts
        hp_result = self.converter.convert_to_si("10 HP", "power")
        assert 7400 < hp_result < 7500
    
    def test_invalid_unit(self):
        """Test handling of invalid units."""
        with pytest.raises(Exception):
            self.converter.convert_to_si("invalid unit", "temperature")
    
    def test_numeric_passthrough(self):
        """Test that numeric values pass through unchanged."""
        result = self.converter.convert_to_si(298.15, "temperature")
        assert result == 298.15