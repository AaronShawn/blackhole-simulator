"""Independent, high-accuracy audit of the Novikov-Thorne / Page-Thorne (NT/PT)
thin-disk flux for a Schwarzschild black hole (M = 1, G = c = 1).

The audit answers three questions with a single, self-contained implementation:

  1. Is the closed form used in ``grref.py`` for
         B(r) = int_{r_in}^{r} (E - Omega L) (dL/dr') dr'
     correct?  (Compare with dense quadrature.)
  2. What is the peak radius of the NT flux, and of the Shakura-Sunyaev flux?
  3. Which geometric weight w(r) makes the energy sum rule
         (2 r_in/R_in) * int_{r_in}^{inf} 2 F(r) w(r) 2 pi r dr = Mdot * eta
     hold exactly, with eta = 1 - E(r_in)?

Quadrature uses the substitution x = 1 - r_in/r, which maps [r_in, inf) to a
finite interval on which every integrand below is smooth (the r^-3 tail of the
flux makes the mapped integrand vanish linearly at x = 1).  A midpoint rule on
4e6 sub-intervals is therefore close to machine precision for these integrands.
"""
import json
import os

import numpy as np

M = 1.0
R_IN = 6.0                      # ISCO of Schwarzschild, r = 6M
SQ3 = np.sqrt(3.0)
X_GRID_N = 4_000_001            # midpoint cells for the mapped integral


# ----------------------------------------------------------------------------
# Thermodynamics of circular geodesics (Schwarzschild, M = 1)
# ----------------------------------------------------------------------------
def energy(r):
    r = np.asarray(r, dtype=float)
    return (1.0 - 2.0 * M / r) / np.sqrt(1.0 - 3.0 * M / r)


def angmom(r):
    r = np.asarray(r, dtype=float)
    return np.sqrt(M * r) / np.sqrt(1.0 - 3.0 * M / r)


def omega(r):
    return np.sqrt(M / np.asarray(r, dtype=float) ** 3)


def dOmega(r):
    r = np.asarray(r, dtype=float)
    return -1.5 * np.sqrt(M) * r ** (-2.5)


def u_redshift(r):
    """E - Omega L  =  sqrt(1 - 3M/r) for circular geodesics."""
    r = np.asarray(r, dtype=float)
    return np.sqrt(1.0 - 3.0 * M / r)


ETA = 1.0 - float(energy(R_IN))


# ----------------------------------------------------------------------------
# The torque integral B(r) = int (E - Omega L) dL/dr dr
# ----------------------------------------------------------------------------
def integrand_B(r, r_in=R_IN):
    """(E - Omega L) dL/dr, reduced analytically: 0.5 r^-1/2 (r-6)/(r-3)."""
    r = np.asarray(r, dtype=float)
    return 0.5 * r ** (-0.5) * (r - 6.0 * M) / (r - 3.0 * M)


def B_closed(r, r_in=R_IN):
    """Closed form used by grref.py: (s - si) - (sqrt3/2) * lg."""
    r = np.asarray(r, dtype=float)
    s = np.sqrt(r)
    si = np.sqrt(r_in)
    lg = np.log((s - SQ3) / (s + SQ3)) - np.log((si - SQ3) / (si + SQ3))
    return (s - si) - 0.5 * SQ3 * lg


def B_closed_plus(r, r_in=R_IN):
    """Rejected variant kept as a counter-example: (s - si) + (sqrt3/2) * lg."""
    r = np.asarray(r, dtype=float)
    s = np.sqrt(r)
    si = np.sqrt(r_in)
    lg = np.log((s - SQ3) / (s + SQ3)) - np.log((si - SQ3) / (si + SQ3))
    return (s - si) + 0.5 * SQ3 * lg


def B_quad(r, r_in=R_IN, n=X_GRID_N):
    """Dense midpoint quadrature of integrand_B on a log grid."""
    s = np.logspace(np.log10(r_in), np.log10(r), n)
    y = integrand_B(s, r_in)
    return float(np.trapezoid(y, s))


# ----------------------------------------------------------------------------
# Fluxes -- all normalised so that the asymptotic tail is 3M/(8 pi) r^-3 Mdot
# ----------------------------------------------------------------------------
def flux_nt(r, r_in=R_IN):
    """Page-Thorne / Novikov-Thorne flux, F = Mdot/(4 pi r) (-dOmega/dr)
    (E - Omega L)^-2 B(r)."""
    r = np.asarray(r, dtype=float)
    return (-dOmega(r)) * B_closed(r, r_in) / u_redshift(r) ** 2 / (4.0 * np.pi * r)


def flux_nt_noB(r, r_in=R_IN):
    """The same, with the plus-sign variant of B (counter-example)."""
    r = np.asarray(r, dtype=float)
    return (-dOmega(r)) * B_closed_plus(r, r_in) / u_redshift(r) ** 2 / (4.0 * np.pi * r)


def flux_ss(r, r_in=R_IN, r_out=None):
    """Shakura-Sunyaev Newtonian flux, F = 3M/(8 pi) (1 - sqrt(r_in/r)) / r^3,
    optionally truncated at an outer edge."""
    r = np.asarray(r, dtype=float)
    f = 3.0 * M / (8.0 * np.pi) * (1.0 - np.sqrt(r_in / r)) / r ** 3
    if r_out is not None:
        f = np.where(r <= r_out, f, 0.0)
    return f


def integral_weighted(weight, flux=flux_nt, r_in=R_IN, n=X_GRID_N):
    """int_{r_in}^{inf} 2 * 2 pi r F(r) w(r) dr via x = 1 - r_in/r.

    Returns the integral divided by Mdot (the fluxes above are per unit Mdot).
    """
    i = np.arange(n, dtype=float) + 0.5
    x = i / n                                  # midpoints of (0, 1)
    inv = 1.0 - x                              # = r_in / r  > 0
    r = r_in / inv
    drdx = r_in / inv ** 2                     # dr/dx
    f = flux(r, r_in)
    return float(np.sum(4.0 * np.pi * r * f * weight(r) * drdx) / n)


def peak_radius(flux, r_in=R_IN, r_out=1.0e4, n=6_000_001):
    r = np.exp(np.linspace(np.log(r_in * (1.0 + 1e-12)), np.log(r_out), n))
    f = flux(r, r_in)
    return float(r[int(np.argmax(f))])


# ----------------------------------------------------------------------------
# The torque identity: total radiated power per unit radius
#   P'(r) = d/dr [ Mdot ( E(r) - (L(r) - L_in) Omega(r) ) ]
# integrates to Mdot (1 - E_in) exactly, and equals the Newtonian result
# Mdot GM/(2 r_in) when the Newtonian E, L, Omega are used.
# ----------------------------------------------------------------------------
def Pprime(r, r_in=R_IN):
    r = np.asarray(r, dtype=float)
    L = angmom(r)
    L_in = float(angmom(np.array([r_in]))[0])
    return -(L - L_in) * dOmega(r)             # per unit Mdot


def main():
    out = {}

    # --- 1. closed form vs quadrature ------------------------------------
    rows = []
    for rr in (6.5, 7.0, 8.0, 10.0, 20.0, 60.0, 200.0):
        q = B_quad(rr)
        m = float(B_closed(rr))
        p = float(B_closed_plus(rr))
        rows.append(dict(r=rr, quadrature=q, closed_minus=m, closed_plus=p,
                         rel_err_minus=(m - q) / q if q else 0.0,
                         rel_err_plus=(p - q) / q if q else 0.0))
    out["B_check"] = rows

    # --- 2. peak radii ---------------------------------------------------
    out["peaks"] = {
        "r_peak_NT_M": peak_radius(flux_nt),
        "r_peak_NT_plus_variant_M": peak_radius(flux_nt_noB, r_out=200.0),
        "r_peak_SS_M": peak_radius(flux_ss, r_in=R_IN, r_out=1.0e4),
        "r_peak_NT_over_r_in": peak_radius(flux_nt) / R_IN,
        "r_peak_SS_over_r_in": peak_radius(flux_ss, r_in=R_IN, r_out=1.0e4) / R_IN,
    }

    # --- 3. energy sum rule ----------------------------------------------
    out["eta"] = ETA
    out["E_R_IN"] = float(energy(np.array([R_IN]))[0])

    # Shakura-Sunyaev control: Newtonian efficiency is M/(2 r_in).
    eta_ss = 0.5 * M / R_IN
    out["SS_control"] = {
        "eta_SS": eta_ss,
        "weight_unity": integral_weighted(lambda r: np.ones_like(r), flux_ss) / eta_ss,
    }

    # Candidate geometric weights, w = (1-2/r)^a * (1-3/r)^b * (r)^c
    weights = {
        "1 (coordinate area)": lambda r: np.ones_like(r),
        "(1-2/r)^-1/2 (proper area)": lambda r: (1 - 2 / r) ** -0.5,
        "(1-2/r)^+1/2": lambda r: (1 - 2 / r) ** 0.5,
        "(1-3/r)^+1/2": lambda r: (1 - 3 / r) ** 0.5,
        "(1-3/r)^-1/2": lambda r: (1 - 3 / r) ** -0.5,
        "(1-3/r)^+1": lambda r: (1 - 3 / r),
        "(1-3/r)^-1": lambda r: (1 - 3 / r) ** -1,
        "(1-3/r)^+3/2": lambda r: (1 - 3 / r) ** 1.5,
        "(1-3/r)^-3/2": lambda r: (1 - 3 / r) ** -1.5,
        "E(r)": lambda r: energy(r),
        "(1-2/r)^-1/2 (1-3/r)^1/2 [comoving dtau]":
            lambda r: (1 - 2 / r) ** -0.5 * (1 - 3 / r) ** 0.5,
        "E(r) (1-2/r)^-1/2": lambda r: energy(r) * (1 - 2 / r) ** -0.5,
    }
    out["sum_rule_NT"] = {
        k: integral_weighted(v) / ETA for k, v in weights.items()
    }
    out["sum_rule_NT"]["P'(r) torque identity"] = (
        float(np.sum([0.0]))  # placeholder overwritten below
    )

    # torque identity, integrated on the same mapped grid
    i = np.arange(X_GRID_N, dtype=float) + 0.5
    x = i / X_GRID_N
    inv = 1.0 - x
    r = R_IN / inv
    out["sum_rule_NT"]["P'(r) torque identity"] = float(
        np.sum(Pprime(r) * (R_IN / inv ** 2)) / X_GRID_N / ETA
    )

    # --- 4. how fast does the NT/SS ratio approach unity? ----------------
    rr = np.array([50.0, 100.0, 1e3, 1e4, 1e5, 1e6])
    out["NT_over_SS"] = [dict(r=float(a), ratio=float(flux_nt(a) / flux_ss(a)))
                         for a in rr]

    # --- 5. flux values at a few radii (for the paper) -------------------
    out["flux_table"] = [
        dict(r=float(a),
             NT=float(flux_nt(a)), SS=float(flux_ss(a)),
             NT_over_SS=float(flux_nt(a) / flux_ss(a)))
        for a in np.array([6.5, 7.0, 8.0, 8.1667, 9.5509, 12.0, 16.0, 26.0])
    ]

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "audit_nt.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
