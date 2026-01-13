# Heat Exchanger Tool Architecture Refactoring Plan

## Problem Statement

The current heat exchanger tools led to erroneous guidance because:

1. **Generic defaults**: `heat_exchanger_design` assumes U=1000 W/m²K (varies 5x across HX types)
2. **Decoupled calculations**: U and ΔP treated independently (in reality coupled through velocity/Re)
3. **Redundant tools**: Multiple tools with overlapping functionality, inconsistent outputs
4. **No geometry-specific correlations**: Shell-tube and PHE require fundamentally different physics

The `size_plate_heat_exchanger` tool demonstrates the correct approach with:
- Geometry-specific correlations (Martin, Kumar)
- Thermal-hydraulic coupling (U and ΔP interdependent through Re)
- Iterative solver for constraints
- Comprehensive, verified outputs

## Solution: Type-Specific Tool Architecture

Replace legacy generic tools with type-specific tools, each with PHE-level detail.

### Target Architecture

```
KEEP (Type-Specific Sizing):
├── size_plate_heat_exchanger.py      ← EXISTS (gold standard)
├── size_shell_tube_heat_exchanger.py ← NEW (Phase 1)
├── size_double_pipe_heat_exchanger.py ← NEW (Phase 4, optional)

KEEP (Rating Mode):
├── rate_heat_exchanger.py            ← REFACTOR from calculate_heat_exchanger_performance

KEEP (Helpers):
├── calculate_pressure_drop.py        ← EXISTS
├── calculate_convection_coefficient.py ← EXISTS
├── hx_shell_side_h_kern.py           ← EXISTS (used by shell-tube tool)
├── estimate_hx_physical_dims.py      ← EXISTS (used by shell-tube tool)
├── heat_duty.py                      ← EXISTS

REMOVE (Legacy):
├── heat_exchanger_design.py          ← REMOVE (misleading defaults)
├── size_heat_exchanger_area.py       ← REMOVE (redundant, now integrated)
├── calculate_heat_exchanger_performance.py ← REFACTOR → rate_heat_exchanger.py

NEW (Shared Utilities):
├── utils/hx_common.py                ← NEW (LMTD, energy balance, output formatting)
```

---

## Phase 1: Create Shell-Tube Heat Exchanger Sizing Tool

**File:** `tools/size_shell_tube_heat_exchanger.py` (NEW)

### Function Signature

```python
def size_shell_tube_heat_exchanger(
    # Duty specification
    heat_duty_W: Optional[float] = None,

    # Temperature specs
    hot_inlet_temp_K: Optional[float] = None,
    hot_outlet_temp_K: Optional[float] = None,
    cold_inlet_temp_K: Optional[float] = None,
    cold_outlet_temp_K: Optional[float] = None,

    # Flow rates (required)
    hot_mass_flow_kg_s: float,
    cold_mass_flow_kg_s: float,

    # Fluids
    hot_fluid: str = "water",
    cold_fluid: str = "water",
    hot_fluid_pressure_Pa: float = 101325.0,
    cold_fluid_pressure_Pa: float = 101325.0,

    # Shell geometry
    shell_inner_diameter_m: float,
    baffle_spacing_m: float,
    baffle_cut_fraction: float = 0.25,
    n_baffles: Optional[int] = None,  # If None, calculated from tube length

    # Tube geometry
    tube_outer_diameter_m: float = 0.019,  # 3/4" OD default
    tube_inner_diameter_m: float = 0.016,
    tube_pitch_m: float = 0.025,
    tube_layout_angle: int = 30,  # 30, 45, 60, or 90 degrees
    tube_length_m: float = 3.0,
    tube_roughness_m: float = 0.00004,
    tube_material_conductivity_W_mK: float = 45.0,  # Carbon steel

    # Configuration
    n_tube_passes: int = 2,
    n_shell_passes: int = 1,
    tube_side_fluid: str = "cold",  # Which fluid in tubes: "hot" or "cold"

    # Constraints for solver
    max_pressure_drop_tube_kPa: Optional[float] = None,
    max_pressure_drop_shell_kPa: Optional[float] = None,
    max_velocity_tube_m_s: float = 3.0,
    max_velocity_shell_m_s: float = 1.5,

    # Fouling
    fouling_factor_tube_m2K_W: float = 0.0001,
    fouling_factor_shell_m2K_W: float = 0.0002,

    # Correlation selection
    shell_side_method: str = "Kern",  # "Kern" or "Zukauskas"
    tube_side_method: str = "auto",   # "auto", "Gnielinski", "Dittus_Boelter"

    # Solver options
    solve_for: Optional[str] = None,  # "n_tubes", "tube_length", None (rating)
    min_tubes: int = 10,
    max_tubes: int = 2000,

    # Parameter sweep
    sweep_n_tube_passes: Optional[List[int]] = None,

    strict: bool = False,
) -> str:
```

### Implementation Pattern (Following PHE Template)

```python
# 1. GEOMETRY: Use TEMA functions from ht.hx
from ht.hx import (
    Ntubes_Phadkeb, DBundle_for_Ntubes_Phadkeb,
    F_LMTD_Fakheri, check_tubing_TEMA, get_tube_TEMA,
    shell_clearance, baffle_thickness
)

# 2. TUBE-SIDE CONVECTION: Use ht.conv_internal
from ht.conv_internal import Nu_conv_internal

# 3. SHELL-SIDE CONVECTION: Use existing hx_shell_side_h_kern or direct
from ht.conv_tube_bank import Nu_Zukauskas_Bejan, dP_Kern

# 4. ITERATIVE SOLVER: Find tube count satisfying thermal + hydraulic constraints
for n_tubes in range(min_tubes, max_tubes):
    # Calculate velocities
    v_tube = m_tube / (rho_tube * n_tubes * A_tube_inner)
    v_shell = m_shell / (rho_shell * A_crossflow)

    # Calculate Reynolds (drives both thermal and hydraulic)
    Re_tube = rho_tube * v_tube * D_inner / mu_tube
    Re_shell = rho_shell * v_shell * D_equiv / mu_shell

    # THERMAL: Calculate h from correlations
    h_tube = Nu_conv_internal(...) * k_tube / D_inner
    h_shell = Nu_Zukauskas_Bejan(...) * k_shell / D_equiv

    # Overall U (tube wall + fouling)
    U = calculate_U(h_tube, h_shell, fouling, tube_wall)

    # Area available vs required
    A_available = n_tubes * pi * D_outer * tube_length_m
    A_required = Q / (U * LMTD * F_correction)

    # HYDRAULIC: Calculate pressure drops
    dP_tube = one_phase_dP(m_tube, rho_tube, mu_tube, D_inner, roughness, L * n_passes)
    dP_shell = dP_Kern(m_shell, rho_shell, mu_shell, DShell, LSpacing, pitch, Do, NBaffles)

    # Check constraints
    if A_available >= A_required and dP_tube <= max_dP_tube and dP_shell <= max_dP_shell:
        break
```

### Output Structure (Matching PHE Format)

```python
{
    "duty_W": float,
    "duty_kW": float,
    "LMTD_K": float,
    "F_correction": float,
    "effectiveness": float,
    "NTU": float,

    "geometry": {
        "type": "shell_tube",
        "n_tubes": int,
        "tube_length_m": float,
        "shell_diameter_m": float,
        "bundle_diameter_m": float,
        "n_baffles": int,
        "baffle_spacing_m": float,
        "tube_pitch_m": float,
        "tube_layout_angle": int,
        "area_per_tube_m2": float,
        "total_area_m2": float,
        "area_required_m2": float,
        "area_margin_pct": float,
        "tube_dimensions": {
            "outer_diameter_m": float,
            "inner_diameter_m": float,
            "wall_thickness_m": float
        }
    },

    "thermal": {
        "U_W_m2K": float,
        "h_tube_W_m2K": float,
        "h_shell_W_m2K": float,
        "Nu_tube": float,
        "Nu_shell": float,
        "Re_tube": float,
        "Re_shell": float,
        "Pr_tube": float,
        "Pr_shell": float,
        "correlation_tube": str,
        "correlation_shell": str,
        "R_wall_m2K_W": float,
        "fouling_tube_m2K_W": float,
        "fouling_shell_m2K_W": float
    },

    "hydraulic": {
        "velocity_tube_m_s": float,
        "velocity_shell_m_s": float,
        "pressure_drop_tube_kPa": float,
        "pressure_drop_shell_kPa": float,
        "friction_factor_tube": float,
        "correlation_tube_dP": str,
        "correlation_shell_dP": str
    },

    "temperatures": {
        "hot_inlet_C": float,
        "hot_outlet_C": float,
        "cold_inlet_C": float,
        "cold_outlet_C": float,
        "terminal_temp_diff_min_K": float,
        "terminal_temp_diff_max_K": float
    },

    "configuration": {
        "n_tube_passes": int,
        "n_shell_passes": int,
        "tube_side_fluid": str,
        "flow_arrangement": str
    },

    "heat_balance_verification": {
        "Q_from_LMTD_kW": float,
        "Q_from_hot_side_kW": float,
        "Q_from_cold_side_kW": float,
        "balance_satisfied": bool
    }
}
```

### Upstream Functions to Use

| Function | Library | Purpose |
|----------|---------|---------|
| `Nu_Zukauskas_Bejan` | ht.conv_tube_bank | Shell-side Nusselt |
| `dP_Kern` | ht.conv_tube_bank | Shell-side pressure drop |
| `Nu_conv_internal` | ht.conv_internal | Tube-side Nusselt (auto-selection) |
| `one_phase_dP` | fluids.friction | Tube-side pressure drop |
| `F_LMTD_Fakheri` | ht.hx | LMTD correction factor |
| `Ntubes_Phadkeb` | ht.hx | Accurate tube count from bundle diameter |
| `DBundle_for_Ntubes_Phadkeb` | ht.hx | Bundle diameter from tube count |
| `check_tubing_TEMA` | ht.hx | Validate TEMA tube sizes |

---

## Phase 2: Remove Legacy Tools and Update Server

### Files to Remove

| File | Reason |
|------|--------|
| `tools/heat_exchanger_design.py` | Generic U=1000 default misleads users |
| `tools/size_heat_exchanger_area.py` | Functionality now in type-specific tools |

### Files to Keep (No Changes)

| File | Reason |
|------|--------|
| `tools/hx_shell_side_h_kern.py` | Used by shell-tube tool |
| `tools/estimate_hx_physical_dims.py` | Used by shell-tube tool |
| `tools/heat_duty.py` | Lightweight utility |
| `tools/calculate_pressure_drop.py` | Supporting tool |
| `tools/convection_coefficient.py` | Supporting tool |

### Update `server.py`

```python
# REMOVE from OMNIBUS_TOOLS:
# - heat_exchanger_design  ← REMOVE

# REMOVE from TOOLS:
# - size_heat_exchanger_area  ← REMOVE
# - calculate_heat_exchanger_performance  ← BECOMES rate_heat_exchanger

# ADD to TOOLS:
# + size_shell_tube_heat_exchanger
# + rate_heat_exchanger

# Updated registration:
OMNIBUS_TOOLS = [
    tank_heat_loss,
    extreme_conditions,
    pipe_heat_management,
    parameter_optimization,
    # heat_exchanger_design REMOVED
]

TOOLS = [
    # ... existing helpers ...
    # Type-specific HX sizing tools
    size_plate_heat_exchanger,
    size_shell_tube_heat_exchanger,  # NEW
    rate_heat_exchanger,  # REFACTORED
    # ... other tools ...
]
```

---

## Phase 3: Create Rating Tool

**File:** `tools/rate_heat_exchanger.py` (REFACTOR from calculate_heat_exchanger_performance)

### Purpose

Rating = Given existing HX, find outlet temperatures and actual duty

### Function Signature

```python
def rate_heat_exchanger(
    # Existing HX parameters
    heat_transfer_area_m2: float,
    overall_U_W_m2K: float,

    # Operating conditions
    hot_inlet_temp_K: float,
    cold_inlet_temp_K: float,
    hot_mass_flow_kg_s: float,
    cold_mass_flow_kg_s: float,

    # Fluids
    hot_fluid: str = "water",
    cold_fluid: str = "water",

    # Configuration
    flow_arrangement: str = "counterflow",  # or "parallel", "shell_tube_1_2"
    n_shell_passes: int = 1,

    # Method
    method: str = "effectiveness_NTU",  # or "LMTD_iterative"
) -> str:
```

### Output Structure

```python
{
    "actual_duty_W": float,
    "actual_duty_kW": float,
    "effectiveness": float,
    "NTU": float,
    "temperatures": {
        "hot_inlet_C": float,
        "hot_outlet_C": float,
        "cold_inlet_C": float,
        "cold_outlet_C": float
    },
    "LMTD_K": float,
    "method_used": str
}
```

---

## Phase 4: Create Shared Utilities (Optional Enhancement)

**File:** `utils/hx_common.py` (NEW)

### Functions to Extract

```python
def calculate_lmtd(Thi, Tho, Tci, Tco, arrangement: str) -> float:
    """Calculate log mean temperature difference."""

def calculate_f_correction(Thi, Tho, Tci, Tco, shells: int) -> float:
    """Calculate LMTD correction factor using F_LMTD_Fakheri."""

def verify_heat_balance(Q_duty, U, A, LMTD, F=1.0) -> dict:
    """Verify Q = U * A * LMTD * F."""

def format_temperature_output(Thi, Tho, Tci, Tco) -> dict:
    """Standard temperature output format (K and C)."""

def check_temperature_crossover(Thi, Tho, Tci, Tco, arrangement: str) -> bool:
    """Check for invalid temperature crossover."""
```

---

## Phase 5: Double-Pipe Tool (Future, Optional)

**File:** `tools/size_double_pipe_heat_exchanger.py` (NEW, lower priority)

Simpler geometry than shell-tube:
- Inner pipe carries one fluid
- Annulus (outer pipe - inner pipe) carries other fluid
- Pure counterflow or parallel flow
- Use De = D_outer - D_inner for annulus

---

## Files to Create/Modify Summary

| File | Action | Phase |
|------|--------|-------|
| `tools/size_shell_tube_heat_exchanger.py` | CREATE | 1 |
| `tools/rate_heat_exchanger.py` | CREATE (refactor) | 3 |
| `tools/heat_exchanger_design.py` | DELETE | 2 |
| `tools/size_heat_exchanger_area.py` | DELETE | 2 |
| `tools/calculate_heat_exchanger_performance.py` | DELETE (replaced) | 3 |
| `server.py` | MODIFY | 2 |
| `utils/hx_common.py` | CREATE (optional) | 4 |
| `tests/test_shell_tube_heat_exchanger.py` | CREATE | 1 |

---

## Verification Plan

### Unit Tests for Shell-Tube Tool

```python
class TestShellTubeHeatExchanger:
    def test_basic_sizing(self):
        """Test basic shell-tube sizing with water-water."""

    def test_thermal_hydraulic_coupling(self):
        """Verify that changing tube count affects both U and dP."""

    def test_constraint_satisfaction(self):
        """Verify solver respects max_pressure_drop constraints."""

    def test_output_consistency(self):
        """Verify Q = U * A * LMTD * F."""

    def test_f_correction_factor(self):
        """Verify F < 1 for multi-pass configurations."""
```

### Integration Test

```python
def test_tool_comparison():
    """Compare PHE vs shell-tube for same duty."""
    # Same duty, same fluids
    # PHE should show: higher U, lower area, higher dP
    # Shell-tube should show: lower U, larger area, lower dP
```

### Manual Verification

1. Run `size_shell_tube_heat_exchanger` with typical industrial case
2. Verify heat balance: Q = U * A * LMTD * F
3. Verify pressure drops are reasonable (typically 10-50 kPa per side)
4. Compare U-value against typical ranges (500-2000 W/m²K for water-water)

---

## Success Criteria

1. **No misleading defaults**: All U-values calculated from geometry, never assumed
2. **Thermal-hydraulic coupling**: Changing geometry affects both U and ΔP
3. **Consistent output format**: All sizing tools return same JSON structure
4. **Heat balance verified**: Every output includes Q = U * A * LMTD verification
5. **Clear tool selection**: LLM can select appropriate tool based on HX type
6. **Legacy tools removed**: No path to generic/misleading calculations

---

## Implementation Order

1. **Phase 1** (Priority): Create `size_shell_tube_heat_exchanger.py` - most common industrial HX
2. **Phase 2** (Required): Remove legacy tools, update server.py
3. **Phase 3** (Required): Create `rate_heat_exchanger.py` for rating mode
4. **Phase 4** (Optional): Extract shared utilities to `utils/hx_common.py`
5. **Phase 5** (Future): Add `size_double_pipe_heat_exchanger.py` if needed
