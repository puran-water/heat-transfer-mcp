"""Wastewater treatment plant heat transfer examples."""

import sys
import os

# Add the parent directory to the path so we can import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.surface_heat_transfer import calculate_surface_heat_transfer
from tools.buried_object_heat_loss import calculate_buried_object_heat_loss
from tools.heat_duty import calculate_heat_duty
from tools.size_heat_exchanger_area import size_heat_exchanger_area
from tools.ground_heat_loss import calculate_ground_heat_loss


def main():
    """Run wastewater treatment plant examples."""

    # Example 1: Anaerobic Digester Heat Loss
    print("Example 1: Anaerobic Digester Heat Loss")
    print("Calculating heat loss from a mesophilic digester in winter conditions...")
    digester_heat_loss = calculate_surface_heat_transfer(
        geometry="vertical_cylinder_tank",
        dimensions={
            "diameter": "50 ft",  # 50-foot diameter digester
            "height": "30 ft",  # 30-foot sidewall height
        },
        internal_temperature="95 degF",  # Mesophilic temperature
        ambient_air_temperature="32 degF",  # Winter design temperature
        wind_speed="15 mph",  # Design wind speed
        surface_emissivity=0.9,
        wall_layers=[
            {"material_name": "concrete", "thickness": "12 in"},
            {"material_name": "polyurethane foam", "thickness": "4 in"},
        ],
    )
    print(f"Digester heat loss: {digester_heat_loss['heat_loss_w']:.0f} W")
    print(f"Digester heat loss: {digester_heat_loss['heat_loss_btu_hr']:.0f} BTU/hr")
    print()

    # Example 2: Sludge Pipeline Heat Loss
    print("Example 2: Sludge Pipeline Heat Loss")
    print("Calculating heat loss from buried hot sludge pipeline...")
    sludge_pipe_loss = calculate_buried_object_heat_loss(
        object_type="pipe",
        diameter="12 in",  # 12-inch sludge line
        length="2000 ft",  # Pipeline length
        burial_depth="5 ft",  # Cover depth
        soil_conductivity=1.3,  # Moist clay (W/m·K)
        object_temperature="140 degF",  # Hot sludge temperature
        ground_surface_temperature="55 degF",  # Annual average ground temp
    )
    print(f"Pipeline heat loss: {sludge_pipe_loss['heat_loss_w']:.0f} W")
    print(f"Pipeline heat loss: {sludge_pipe_loss['heat_loss_btu_hr']:.0f} BTU/hr")
    print()

    # Example 3: Aeration Blower Cooling
    print("Example 3: Aeration Blower Cooling")
    print("Calculating cooling required for blower discharge air...")
    blower_cooling = calculate_heat_duty(
        calculation_method="sensible_heat",
        fluid_name="air",
        flow_rate="5000 SCFM",  # Standard cubic feet per minute
        inlet_temp="180 degF",  # Discharge temperature
        outlet_temp="100 degF",  # Target temperature
    )
    print(f"Blower cooling duty: {blower_cooling['heat_duty_w']:.0f} W")
    print(f"Blower cooling duty: {blower_cooling['heat_duty_btu_hr']:.0f} BTU/hr")
    print()

    # Example 4: Sludge Heat Recovery
    print("Example 4: Sludge Heat Recovery")
    print("Sizing heat exchanger for sludge heating...")
    heat_recovery = size_heat_exchanger_area(
        required_heat_duty_q="300 HP",  # Available from boiler
        hot_fluid_name="water",
        hot_fluid_flow_rate="250 GPM",  # Hot water flow
        hot_fluid_inlet_temp="150 degF",  # Boiler water temperature
        cold_fluid_name="water",  # Sludge approximated as water
        cold_fluid_flow_rate="200 GPM",  # Sludge flow rate
        cold_fluid_inlet_temp="60 degF",  # Raw sludge temperature
        flow_arrangement="counterflow",
        tube_outer_diameter="0.75 in",
        tube_inner_diameter="0.625 in",
        fouling_factor_inner="0.001 hr*ft^2*F/BTU",  # Sludge side
        fouling_factor_outer="0.0005 hr*ft^2*F/BTU",  # Water side
    )
    print(f"Required heat exchanger area: {heat_recovery['area_m2']:.1f} m²")
    print(f"Required heat exchanger area: {heat_recovery['area_ft2']:.1f} ft²")
    print()

    # Example 5: Primary Clarifier Heat Loss
    print("Example 5: Primary Clarifier Heat Loss")
    print("Calculating heat loss from exposed clarifier surface...")
    clarifier_loss = calculate_surface_heat_transfer(
        geometry="horizontal_plate",
        dimensions={
            "length": "100 ft",  # Clarifier length
            "width": "30 ft",  # Clarifier width
        },
        internal_temperature="55 degF",  # Winter wastewater temp
        ambient_air_temperature="20 degF",  # Cold night temperature
        wind_speed="10 mph",
        surface_emissivity=0.95,  # Water surface
    )
    print(f"Clarifier heat loss: {clarifier_loss['heat_loss_w']:.0f} W")
    print(f"Clarifier heat loss: {clarifier_loss['heat_loss_btu_hr']:.0f} BTU/hr")
    print()

    # Example 6: Equipment Building Heat Loss
    print("Example 6: Equipment Building Heat Loss")
    print("Calculating heat loss through building slab...")
    building_loss = calculate_ground_heat_loss(
        foundation_type="slab_on_grade",
        foundation_length="100 ft",
        foundation_width="60 ft",
        foundation_depth="4 ft",
        wall_thickness="8 in",  # Concrete block
        floor_thickness="6 in",  # Slab thickness
        inside_temperature="70 degF",  # Indoor temperature
        outside_temperature="32 degF",  # Winter design temp
        ground_temperature="50 degF",  # Deep ground temperature
        wall_thermal_conductivity=0.5,  # Concrete block
        floor_thermal_conductivity=1.7,  # Concrete
        soil_thermal_conductivity=1.3,  # Moist soil
    )
    print(f"Building heat loss: {building_loss['heat_loss_w']:.0f} W")
    print(f"Building heat loss: {building_loss['heat_loss_btu_hr']:.0f} BTU/hr")
    print()

    print("All examples completed successfully!")


if __name__ == "__main__":
    main()