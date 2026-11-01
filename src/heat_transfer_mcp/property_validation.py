"""Check selected thermo correlations before reading extrapolated properties."""

THERMO_OWN_METHODS_RULING_ID = 'PUBLIC-RULE-580dcf227f5b'
THERMO_OWN_METHODS_RULE = (
    "A named property_source='thermo' answers from thermo's own methods, never a CoolProp-backed one, whether or not "
    "CoolProp is installed: a method_P thermo set to COOLPROP takes the best one thermo ranks after it (its choice with "
    "CoolProp absent); a temperature method set to COOLPROP refuses, since Chemical reads those on construction; the "
    f"method evidence names the method and method_P that answered ({THERMO_OWN_METHODS_RULING_ID} OQ-C126-1)")


def thermo_chemical(fluid_name, T, P=101325.0):
    """THERMO_OWN_METHODS_RULE: thermo's Chemical at (T, P) with no property read through CoolProp, or ValueError."""
    from thermo import Chemical
    from thermo.utils import COOLPROP, TDependentProperty

    chem = Chemical(fluid_name, T=T, P=P)
    for name, prop in vars(chem).items():
        if not isinstance(prop, TDependentProperty):
            continue
        if prop.method == COOLPROP:
            raise ValueError(f"{name}: thermo selected COOLPROP as its temperature method, which Chemical reads on "
                             "construction; no answer of thermo's own is separable (THERMO_OWN_METHODS_RULE)")
        if getattr(prop, "method_P", None) == COOLPROP:
            prop.method_P = next((method for method in prop.ranked_methods_P
                                  if method != COOLPROP and method in prop.all_methods_P), None)
    return chem


def require_thermo_method(chemical, family):
    """Return method evidence or refuse outside the library's stated envelope."""
    prop = getattr(chemical, family)
    if family == "VolumeGas":
        method = prop.method_P
        valid = method is not None and prop.test_method_validity_P(chemical.T, chemical.P, method)
    else:
        method = prop.method
        valid = method is not None and prop.test_method_validity(chemical.T, method)
    if not valid:
        raise ValueError(f"{family}: T={chemical.T} K outside selected method {method} validity")
    evidence = {"method": method, "temperature_limits_K": prop.T_limits.get(method)}
    if family != "VolumeGas" and getattr(prop, "method_P", None) is not None:
        # a value read at T and P comes from method_P (thermo's own, THERMO_OWN_METHODS_RULE), not method
        evidence["method_P"] = prop.method_P
    return evidence
