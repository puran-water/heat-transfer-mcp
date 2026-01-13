"""Wastewater treatment plant heat transfer examples."""

import sys
import os

# Add the parent directory to the path so we can import tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.surface_heat_transfer import calculate_surface_heat_transfer
from tools.buried_object_heat_loss import calculate_buried_object_heat_loss
from tools.heat_duty import calculate_heat_duty
from tools.size_shell_tube_heat_exchanger import size_shell_tube_heat_exchanger
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
    print("Sizing shell-tube heat exchanger for sludge heating...")
    # Convert units: 300 HP ≈ 223,710 W, 250 GPM ≈ 15.8 kg/s, 200 GPM ≈ 12.6 kg/s
    # 150°F ≈ 338.7 K, 60°F ≈ 288.7 K
    import json

    heat_recovery_json = size_shell_tube_heat_exchanger(
        heat_duty_W=223710,  # 300 HP in W
        hot_inlet_temp_K=338.7,  # 150°F
        cold_inlet_temp_K=288.7,  # 60°F
        hot_mass_flow_kg_s=15.8,  # 250 GPM water
        cold_mass_flow_kg_s=12.6,  # 200 GPM sludge (as water)
        hot_fluid="water",
        cold_fluid="water",
        tube_outer_diameter_m=0.01905,  # 3/4"
        tube_inner_diameter_m=0.01588,  # 5/8"
        tube_length_m=3.0,  # 10 ft tubes
        n_tube_passes=4,  # 4-pass for turbulent flow
        fouling_factor_tube_m2K_W=0.000176,  # Sludge side (~0.001 hr*ft²*F/BTU)
        fouling_factor_shell_m2K_W=0.000088,  # Water side (~0.0005 hr*ft²*F/BTU)
        auto_optimize=True,  # Auto-find optimal configuration
    )
    heat_recovery = json.loads(heat_recovery_json)
    if "error" not in heat_recovery:
        # Extract from optimization result or direct result
        result = heat_recovery.get("best_result", heat_recovery)
        print(f"Required heat exchanger area: {result['geometry']['area_required_m2']:.1f} m²")
        print(f"Number of tubes: {result['geometry']['n_tubes']}")
        print(f"Overall U-value: {result['thermal']['U_W_m2K']:.0f} W/m²K")
    else:
        print(f"Heat exchanger sizing failed: {heat_recovery['error']}")
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
