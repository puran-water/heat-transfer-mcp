"""Published chevron-correlation envelopes, shared by heat and pressure tools.

Sources: ht.conv_plate and fluids.friction documentation for the named methods.
These are applicability limits, not extrapolation warnings.
"""


def require_plate_domain(method, reynolds, angle, enlargement=None, prandtl=None):
    if method in {"Martin_1999", "Martin_VDI"}:
        valid = 200 <= reynolds <= 10000 and 0 <= angle <= 80
        limits = "200 <= Re <= 10000; 0 <= angle <= 80 degrees"
    elif method == "Kumar":
        valid = 0.1 <= reynolds <= 10000 and 30 <= angle <= 65
        limits = "0.1 <= Re <= 10000; 30 <= angle <= 65 degrees"
    elif method == "Muley_Manglik":
        valid = reynolds > 1000 and 30 <= angle <= 60 and enlargement is not None and 1 <= enlargement <= 1.5
        limits = "Re > 1000; 30 <= angle <= 60 degrees; 1 <= enlargement <= 1.5"
    elif method == "Khan_Khan":
        valid = 500 <= reynolds <= 2500 and 30 <= angle <= 60 and prandtl is not None and 3.5 <= prandtl <= 6
        limits = "500 <= Re <= 2500; 30 <= angle <= 60 degrees; 3.5 <= Pr <= 6"
    else:
        raise ValueError(f"Unsupported plate correlation: {method}")
    if not valid:
        raise ValueError(f"{method} outside correlation validity: {limits}; got Re={reynolds}, angle={angle}, enlargement={enlargement}, Pr={prandtl}")
