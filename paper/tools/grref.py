"""grref - general-relativistic reference model for the Schwarzschild ray tracer.

This module is the *numerical ground truth* used by the paper: every figure and
every validation number in ``paper/`` is produced from the functions below, and
the same equations are implemented in GLSL in ``web/src/shaders.js``.

Units: G = c = M = 1, so r_s = 2, photon sphere r = 3, ISCO r = 6,
critical impact parameter b_c = 3*sqrt(3) ~ 5.1962.

Conventions
-----------
* ``u`` is 1/r and ``phi`` the orbital-plane azimuth.
* The null-geodesic orbit equation is
      d2u/dphi2 = -u + 3 u^2,
  with first integral (du/dphi)^2 = 1/b^2 - u^2 (1 - 2u).
* ``b`` is the impact parameter |L|/E of the photon.
"""

from __future__ import annotations

import numpy as np

M = 1.0
RS = 2.0 * M
R_PHOTON = 3.0 * M
R_ISCO = 6.0 * M
B_CRIT = 3.0 * np.sqrt(3.0) * M


# --------------------------------------------------------------------------- #
#  Analytic circular-orbit quantities                                          #
# --------------------------------------------------------------------------- #
def omega(r):
    """Coordinate angular velocity of a circular geodesic."""
    return np.sqrt(M / np.asarray(r, dtype=float) ** 3)


def specific_energy(r):
    """E(r) = (1-2M/r)/sqrt(1-3M/r)."""
    r = np.asarray(r, dtype=float)
    return (1.0 - 2.0 * M / r) / np.sqrt(1.0 - 3.0 * M / r)


def specific_angular_momentum(r):
    """L(r) = sqrt(M r)/sqrt(1-3M/r)."""
    r = np.asarray(r, dtype=float)
    return np.sqrt(M * r) / np.sqrt(1.0 - 3.0 * M / r)


def ut_circular(r):
    """u^t of a circular-geodesic emitter, 1/sqrt(1-3M/r)."""
    r = np.asarray(r, dtype=float)
    return 1.0 / np.sqrt(1.0 - 3.0 * M / r)


def isco_efficiency():
    """Radiative efficiency of a zero-torque Schwarzschild disk: 1 - E(r_isco)."""
    return 1.0 - specific_energy(R_ISCO)


# --------------------------------------------------------------------------- #
#  Null geodesics                                                             #
# --------------------------------------------------------------------------- #
def rhs(state):
    """d/dphi (u, du/dphi) = (du/dphi, -u + 3u^2)."""
    u, du = state
    return np.array([du, -u + 3.0 * u * u])


def rk4_step(state, h):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * h * k1)
    k3 = rhs(state + 0.5 * h * k2)
    k4 = rhs(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rkf45_step(state, h):
    """Dormand-Prince 5(4) step with an embedded 4th-order error estimate."""
    c = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
    a = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    b5 = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
    b4 = [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    k = []
    for i in range(7):
        s = state.copy()
        for j, aij in enumerate(a[i]):
            s = s + h * aij * k[j]
        k.append(rhs(s))
    y5 = state + h * sum(b * ki for b, ki in zip(b5, k))
    y4 = state + h * sum(b * ki for b, ki in zip(b4, k))
    err = np.linalg.norm(y5 - y4)
    return y5, err


def _g(u, b):
    """Radicand of the first integral: (du/dphi)^2 = g(u)."""
    return 1.0 / b**2 - u * u * (1.0 - 2.0 * u)


def turning_point(b):
    """Smallest positive root of g(u)=0, i.e. the periapsis radius u=1/r.

    ``None`` means the photon is captured (b < b_c = 3 sqrt(3)).
    """
    if b <= B_CRIT * (1.0 + 1e-15):
        return None
    lo, hi = 1e-14, 1.0 / 3.0                     # g(0)>0 , g(1/3)<0 for b>b_c
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _g(mid, b) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cubic_roots(b):
    """Roots r0 < r1 < r2 of 2u^3 - u^2 + 1/b^2 = 0.

    ``r1`` is the turning point of a photon with b > b_c (the smallest positive
    root); ``r0`` is negative.  Writing

        g(u) = 2 (u - r0)(r1 - u)(r2 - u)

    keeps the square root regular at the turning point and avoids the
    catastrophic cancellation of the naive 1/b^2 - u^2 + 2u^3 form for large b.
    Returns ``None`` for captured photons (b <= b_c), where the two positive
    roots merge.
    """
    r1 = turning_point(b)
    if r1 is None:
        return None
    # 2u^3 - u^2 + c = 2 (u - r1)(u^2 + p u + q)
    p = r1 - 0.5
    q = r1 * r1 - 0.5 * r1
    disc = max(p * p - 4.0 * q, 0.0)
    a = 0.5 * (-p + np.sqrt(disc))
    c = 0.5 * (-p - np.sqrt(disc))
    return np.array(sorted([c, r1, a]))


def sqrt_g(u, b):
    """Stable sqrt of the first-integral radicand using the factored cubic."""
    roots = cubic_roots(b)
    if roots is None:
        return np.zeros_like(np.asarray(u, dtype=float))
    r0, r1, r2 = roots
    u = np.asarray(u, dtype=float)
    val = 2.0 * np.maximum(u - r0, 0.0) * np.maximum(r1 - u, 0.0) * np.maximum(r2 - u, 0.0)
    return np.sqrt(np.maximum(val, 0.0))


def _phi_integrand(u, b):
    """1/sqrt(g(u)) with the square-root endpoint singularity removed."""
    u1 = turning_point(b)
    if u1 is None:
        raise ValueError("captured ray has no turning point")
    if np.any(np.asarray(u) > u1):
        raise ValueError("u above the turning point")
    return 1.0 / np.sqrt(np.maximum(_g(u, b), 1e-300))


def phi_to_periapsis(b, u, n=4000):
    """Azimuth swept from u (=1/r) up to the periapsis (no singularity)."""
    u1 = turning_point(b)
    if u1 is None:
        return np.inf
    u = np.asarray(u, dtype=float)
    r0, r1, r2 = cubic_roots(b)
    u1 = r1
    A, B = r1 - r0, r2 - r1
    # substitute u = u1 - s^2 : du = -2 s ds and
    # sqrt(g) = s * sqrt(2 (A - s^2)(B + s^2))       (regular at s=0)
    s_max = np.sqrt(np.maximum(u1 - u, 0.0))
    out = np.empty_like(u)
    for i, sm in enumerate(np.atleast_1d(s_max)):
        s = np.linspace(0.0, sm, n)
        # du/sqrt(g) = 2 ds / sqrt(2 (A-s^2)(B+s^2))   (the 1/s factor cancels)
        f = 2.0 / np.sqrt(np.maximum(2.0 * (A - s * s) * (B + s * s), 1e-300))
        val = np.trapezoid(f, s)
        out.flat[i] = val
    return out if out.shape else float(out)


def deflection_exact(b, n=200000):
    """Exact light deflection for impact parameter b (M=1).

        alpha(b) = 2 int_0^{u_1} du / sqrt(1/b^2 - u^2 + 2u^3) - pi
    """
    if b <= B_CRIT:
        return np.inf
    r0, r1, r2 = cubic_roots(b)
    A, B = r1 - r0, r2 - r1
    s = np.linspace(0.0, np.sqrt(r1), n)
    f = 2.0 / np.sqrt(np.maximum(2.0 * (A - s * s) * (B + s * s), 1e-300))
    integral = np.trapezoid(f, s)
    return 2.0 * integral - np.pi


def capture_probability(b):
    """True if a photon from infinity with this impact parameter hits the horizon."""
    return b <= B_CRIT


def trajectory_xy(b, r_from, r_to, n=3000, branch=+1):
    """Cartesian trajectory of a photon between two radii (branch=+1 outgoing).

    Returns (x, y) arrays in the orbital plane; the periapsis sits at the
    origin of the azimuth, negative azimuth corresponds to the incoming leg.
    """
    u1 = turning_point(b)
    if u1 is None:
        raise ValueError("captured ray")
    r_end = r_to if branch > 0 else r_from
    u_grid = 1.0 / np.linspace(max(r_from, 1.0 / u1), r_end, n)
    u_grid = np.clip(u_grid, 1e-12, u1 * (1 - 1e-12))
    phi = phi_to_periapsis(b, u_grid)
    r = 1.0 / u_grid
    return r * np.cos(branch * phi), r * np.sin(branch * phi)


def trajectory_infall(b, r_start, u_end=0.5, h=2.0e-3, max_steps=400000):
    """Inward spiral of a captured photon (b <= b_c) down to the horizon.

    The orbit equation  u'' = -u + 3u^2  is integrated forward in phi with the
    branch  u' = +sqrt(g(u, b))  (u = 1/r grows inwards) and RK4 steps.  The
    trajectory starts at ``r_start`` and stops at u_end = 1/r_s = 0.5.
    """
    u = 1.0 / r_start
    a = float(np.sqrt(max(_g(u, b), 0.0)))          # du/dphi

    def deriv(uu, aa):
        return aa, -uu + 3.0 * uu * uu

    xs, ys, phi = [], [], 0.0
    for _ in range(max_steps):
        r = 1.0 / u
        xs.append(r * np.cos(phi))
        ys.append(r * np.sin(phi))
        if u >= u_end:
            break
        k1 = deriv(u, a)
        k2 = deriv(u + 0.5 * h * k1[0], a + 0.5 * h * k1[1])
        k3 = deriv(u + 0.5 * h * k2[0], a + 0.5 * h * k2[1])
        k4 = deriv(u + h * k3[0], a + h * k3[1])
        u += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        a += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        phi += h
        if not np.isfinite(u):
            break
    return np.array(xs), np.array(ys)


def trajectory_full(b, r_far=60.0, n=1400, h=2.0e-3, max_steps=400000):
    """Complete photon trajectory entering from ``r_far``, in the orbital plane.

    Scattered rays (b > b_c) are returned whole: the incoming leg, the
    periapsis and the outgoing leg are joined into one polyline by sampling the
    regularising variable  s = sqrt(u_1 - u),  which distributes points evenly
    in azimuth even for near-critical impact parameters.  Captured rays
    (b <= b_c) are integrated inwards until they cross the horizon.
    """
    if capture_probability(b):
        return trajectory_infall(b, r_far, h=h, max_steps=max_steps)
    u1 = float(turning_point(b))
    s_max = np.sqrt(max(u1 - 1.0 / r_far, 0.0))
    s = np.linspace(s_max, 0.0, n)
    u = np.clip(u1 - s * s, 1e-15, u1 * (1.0 - 1e-13))
    r = 1.0 / u
    phi = np.atleast_1d(phi_to_periapsis(b, u, n=1600)).astype(float)
    x_in, y_in = r * np.cos(-phi), r * np.sin(-phi)
    x_out, y_out = r[::-1] * np.cos(phi[::-1]), r[::-1] * np.sin(phi[::-1])
    return np.concatenate([x_in, x_out]), np.concatenate([y_in, y_out])


def tangent_xy(b, r):
    """Unit propagation direction at radius r on the outgoing branch."""
    u1 = turning_point(b)
    if u1 is None:
        raise ValueError("captured ray")
    u = 1.0 / r
    du_dphi = -float(sqrt_g(np.array([u]), b)[0])   # outgoing: u decreasing
    # position azimuth measured from the periapsis
    phi = float(np.atleast_1d(phi_to_periapsis(b, np.array([u])))[0])
    er = np.array([np.cos(phi), np.sin(phi)])
    ep = np.array([-np.sin(phi), np.cos(phi)])
    drdphi = -du_dphi / u**2
    v = drdphi * er + r * ep
    return v / np.linalg.norm(v), phi


def tangent_error(R, b, R_ref=1.0e7):
    """Angular error of the shader's escape approximation (tangent at R).

    The tracer stops at u < 1/R and uses the local tangent as the asymptotic
    direction; this returns the angle between that tangent and the tangent at
    r = R_ref (an excellent stand-in for the true asymptote).
    """
    t1, _ = tangent_xy(b, R)
    t2, _ = tangent_xy(b, R_ref)
    return float(np.arccos(np.clip(np.dot(t1, t2), -1.0, 1.0)))


def residual_deflection_weak(R, b):
    """Weak-field residual bending beyond radius R: (2M/b)(1 - R/sqrt(R^2+b^2))."""
    return (2.0 * M / b) * (1.0 - R / np.sqrt(R * R + b * b))


def trace_ode(b, r_start, u_escape, tol=1e-9, integrator="rk4", h=0.02,
              record=False):
    """Mirror of the GLSL integrator: RK4 in phi with adaptive steps.

    Starts at radius ``r_start`` on the *incoming* branch (u increasing) and
    stops at u < u_escape.  Returns the swept azimuth and, optionally, the
    trajectory, so that the shader's step control can be compared with the
    exact integral solution.
    """
    u = 1.0 / r_start
    g = _g(u, b)
    du = np.sqrt(max(g, 0.0))                    # incoming: u increasing
    state = np.array([u, du])
    phi = 0.0
    u1 = turning_point(b)
    if u1 is None:
        return dict(captured=True, phi=0.0, traj=None)
    traj = [(phi, state[0])] if record else None
    for _ in range(2_000_000):
        if state[0] > 0.5:
            return dict(captured=True, phi=phi, traj=traj)
        if state[0] <= 1e-12:                     # numerically escaped
            return dict(captured=False, phi=phi, traj=traj)
        if state[0] < u_escape and state[1] < 0.0:
            break
        if integrator == "rk4":
            new = rk4_step(state, h)
            err = 0.0
        else:
            new, err = rkf45_step(state, h)
            if err > 0:
                h = min(max(0.9 * h * (tol / max(err, 1e-16)) ** 0.2, 1e-8), 0.2)
        # land exactly on u_escape with a partial final step so that the swept
        # azimuth is comparable to the exact value to O(h^4)
        if state[1] < 0.0 and new[0] < u_escape <= state[0]:
            lo, hi = 0.0, h
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                trial = rk4_step(state, mid)
                if trial[0] > u_escape:
                    lo = mid
                else:
                    hi = mid
            state = rk4_step(state, 0.5 * (lo + hi))
            phi += 0.5 * (lo + hi)
            if record:
                traj.append((phi, state[0]))
            return dict(captured=False, phi=phi, traj=traj)
        state = new
        phi += h
        if record:
            traj.append((phi, state[0]))
    return dict(captured=False, phi=phi, traj=traj)


def deflection(b, **kw):
    """Deflection angle alpha(b) for a photon from infinity (exact integral)."""
    return deflection_exact(b, **kw)


def weak_field_series(b, order=4):
    """Post-Newtonian series for the light deflection (M=1)."""
    b = np.asarray(b, dtype=float)
    terms = [4.0 / b, 15.0 * np.pi / 4.0 / b**2, 128.0 / 3.0 / b**3,
             3465.0 * np.pi / 64.0 / b**4]
    return sum(terms[:order])


def impact_parameter_from_local_angle(psi, r0):
    """b measured by a static observer at r0 for a ray at local angle psi."""
    return r0 * np.sin(psi) / np.sqrt(1.0 - RS / r0)


def shadow_angle(r0):
    """Apparent angular radius of the shadow for a static observer at r0."""
    return np.arcsin(np.minimum(1.0, B_CRIT * np.sqrt(1.0 - RS / np.asarray(r0, float)) / np.asarray(r0, float)))


# --------------------------------------------------------------------------- #
#  Accretion-disk profiles                                                    #
# --------------------------------------------------------------------------- #
def flux_newtonian(r, r_in=R_ISCO, norm="peak"):
    """Shakura-Sunyaev / Newtonian zero-torque flux  F ~ r^-3 (1-sqrt(r_in/r))."""
    r = np.asarray(r, dtype=float)
    f = (1.0 - np.sqrt(r_in / r)) / r**3
    if norm == "peak":
        r_p = (49.0 / 36.0) * r_in
        f_p = (1.0 - np.sqrt(r_in / r_p)) / r_p**3
        return f / f_p
    return f


def _nt_integral(r, r_in=R_ISCO):
    """Closed form of  int_{r_in}^{r} (E - Omega L) dL/dr dr  (M = 1).

    Using  (E - Omega L) = sqrt(1-3/r)  and
           (E - Omega L) dL/dr = (1/2) r^-1/2 (r-6)/(r-3)
    the primitive is   sqrt(r) - sqrt(r_in) - (sqrt3/2) * log-ratio.
    """
    r = np.asarray(r, dtype=float)
    s = np.sqrt(r)
    si = np.sqrt(r_in)
    a = np.sqrt(3.0)
    lg = np.log((s - a) / (s + a)) - np.log((si - a) / (si + a))
    return 0.5 * (2.0 * (s - si) - a * lg)


def flux_page_thorne(r, r_in=R_ISCO, exponent=2, norm="peak"):
    """General-relativistic (Page-Thorne 1974) flux for a zero-torque disk.

        F(r) = 1/(4 pi r) * (-dOmega/dr) * I(r) / (E - Omega L)^exponent
        I(r) = int_{r_in}^{r} (E - Omega L) dL/dr dr'

    The denominator exponent is 2 in the published form; it is exposed here so
    that the Newtonian limit and the sum rule can be studied explicitly.
    """
    r = np.asarray(r, dtype=float)
    domega = -1.5 * np.sqrt(M) * r ** (-2.5)
    q = np.sqrt(1.0 - 3.0 * M / r)
    f = (-domega) * _nt_integral(r, r_in) / q**exponent / (4.0 * np.pi * r)
    if norm == "peak":
        rr = np.exp(np.linspace(np.log(r_in * 1.0000001), np.log(60 * r_in), 200000))
        ff = 1.5 * rr ** (-2.5) * _nt_integral(rr, r_in) / np.sqrt(1 - 3 / rr) ** exponent / (4 * np.pi * rr)
        f = f / np.nanmax(ff)
    return f


def profile_peak_radius(kind="nt", r_in=R_ISCO):
    r = np.exp(np.linspace(np.log(r_in * 1.0000001), np.log(60 * r_in), 400000))
    f = flux_page_thorne(r, r_in, norm="none") if kind == "nt" else flux_newtonian(r, r_in, norm="none")
    return float(r[int(np.nanargmax(f))])


def disk_luminosity(kind="nt", r_in=R_ISCO, faces=2, area="coordinate",
                    redshift=False, n=800001):
    """Radiated luminosity in units of Mdot c^2 for a zero-torque disk.

    ``area``     : 'coordinate' (2 pi r dr) or 'proper' (2 pi r dr / sqrt(1-2/r))
    ``redshift`` : multiply by the gravitational redshift factor (1-2/r),
                   which is what converts local flux into energy at infinity

    A correct profile satisfies L/(1-E(r_in)) = 1 for the appropriate
    convention (proper area + redshift, or coordinate area without it).
    """
    r = np.exp(np.linspace(np.log(r_in * 1.0000001), np.log(1e7), n))
    if kind == "nt":
        domega = -1.5 * np.sqrt(M) * r ** (-2.5)
        f = (-domega) * _nt_integral(r, r_in) / (1 - 3 * M / r) ** 2 / (4 * np.pi * r)
    else:
        f = 3.0 * M / (8 * np.pi) * (1 - np.sqrt(r_in / r)) / r**3
    w = 2 * np.pi * r
    if area == "proper":
        w = w / np.sqrt(1 - RS / r)
    if redshift:
        w = w * (1 - RS / r)
    return faces * np.trapezoid(f * w, r)


def nt_over_ss(r, r_in=R_ISCO):
    """Ratio of the Page-Thorne and Newtonian fluxes with classical prefactors.

    Both are evaluated with Mdot = 1 and no peak normalisation, so the ratio
    tends to one at large r - the exact Newtonian limit of the relativistic
    profile - and quantifies the relativistic correction near the ISCO.
    """
    classical_ss = (3.0 * M / (8.0 * np.pi)) * flux_newtonian(r, r_in, norm="none")
    return flux_page_thorne(r, r_in, norm="none") / classical_ss


def temperature_from_flux(f_hat, t_max):
    """T(r) = T_max * (F/F_max)^{1/4}."""
    return t_max * np.maximum(f_hat, 0.0) ** 0.25


# --------------------------------------------------------------------------- #
#  Relativistic transfer factors                                              #
# --------------------------------------------------------------------------- #
def doppler_factor(r, b_axis, spin=1.0, keplerian=True, r_obs=np.inf):
    """g = nu_obs/nu_emit for an emitter on a circular orbit.

        g = 1 / [ u^t sqrt(1-r_s/r_obs) (1 + Omega b_axis) ]
    """
    r = np.asarray(r, dtype=float)
    if keplerian:
        ut = 1.0 / np.sqrt(1.0 - 3.0 * M / r)
        om = spin / r**1.5
    else:
        ut = 1.0 / np.sqrt(1.0 - 2.0 * M / r)
        om = np.zeros_like(r)
    f_obs = 1.0 if not np.isfinite(r_obs) else np.sqrt(1.0 - RS / r_obs)
    return 1.0 / (ut * f_obs * (1.0 + om * b_axis))


def photon_from_local_direction(r, n_hat, e_loc=1.0):
    """4-momentum of a photon with a given *locally measured* direction.

    Static orthonormal frame of the metric (+---):
        e_(t) = (1-2M/r)^-1/2 d_t ,  e_(r) = (1-2M/r)^1/2 d_r , e_(phi) = d_phi / r
    so for a local direction n_hat = (n_r, n_phi) on the unit sphere
        p^t = E (1-2M/r)^-1/2 ,  p^r = E n_r (1-2M/r)^1/2 , p^phi = E n_phi / r
    and the covariant components follow from the metric.
    """
    nr, nph = n_hat
    f = 1.0 - RS / r
    p_t = e_loc * np.sqrt(f)
    p_r = e_loc * nr / np.sqrt(f)
    p_phi = e_loc * r * nph
    return np.array([p_t, p_r, p_phi])


def doppler_from_4vectors(r, n_hat, keplerian=True, r_obs=None, e_loc=1.0):
    """Independent (4-vector) evaluation of the frequency ratio g.

        g = (p . u_obs) / (p . u_em)

    Returns (g, b_axis) where ``b_axis`` is L_y/E of the photon, i.e. the
    quantity that appears in the closed form used by the shader.
    """
    p = photon_from_local_direction(r, n_hat, e_loc=e_loc)
    p_t, p_r, p_phi = p
    ut = 1.0 / np.sqrt(1.0 - (3.0 if keplerian else 2.0) * M / r)
    om = (1.0 / r**1.5) if keplerian else 0.0
    nu_em = p_t * ut + p_phi * ut * om
    r_obs_eff = r if r_obs is None else r_obs
    u_obs_t = 1.0 / np.sqrt(1.0 - RS / r_obs_eff)
    nu_obs = p_t * u_obs_t
    return nu_obs / nu_em, p_phi / p_t


# --------------------------------------------------------------------------- #
#  Blackbody colour (identical to the GLSL implementation)                    #
# --------------------------------------------------------------------------- #
def planck_xy(t):
    """CIE 1931 chromaticity of a Planck spectrum (Kim et al. cubic fits)."""
    t = np.clip(np.asarray(t, dtype=float), 1000.0, 40000.0)
    x = np.where(
        t < 4000.0,
        -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910,
        -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390,
    )
    y = np.where(
        t < 2222.0,
        -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683,
        np.where(
            t < 4000.0,
            -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867,
            3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483,
        ),
    )
    return x, np.maximum(y, 1e-4)


def planck_rgb(t):
    """Linear sRGB chromaticity of a Planck spectrum, unit luminance."""
    x, y = planck_xy(t)
    X = x / y
    Y = np.ones_like(X)
    Z = (1.0 - x - y) / y
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z
    rgb = np.stack([r, g, b], axis=-1)
    mn = rgb.min(axis=-1, keepdims=True)
    rgb = rgb - np.minimum(mn, 0.0)
    return rgb


def planck_spectrum(t, lam_nm):
    """Spectral radiance B_lambda(T) in arbitrary units (SI constants omitted)."""
    lam = np.asarray(lam_nm, dtype=float) * 1e-9
    h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return (2 * h * c**2 / lam**5) / (np.exp(h * c / (lam * kB * t)) - 1.0)


# --------------------------------------------------------------------------- #
#  Radiative transfer (mirrors the GLSL loop)                                 #
# --------------------------------------------------------------------------- #
def gaussian_column_weight(y1, y2, h):
    """<exp(-y^2/2H^2)> averaged over a straight segment y1 -> y2."""
    y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
    dy = y2 - y1
    out = np.empty_like(dy)
    small = np.abs(dy) < 1e-9
    out[small] = np.exp(-0.5 * (y1[small] / h) ** 2)
    from math import erf, sqrt
    k = 1.0 / (h * np.sqrt(2.0))
    erf_v = np.vectorize(erf)
    big = ~small
    out[big] = np.abs(np.sqrt(np.pi / 2.0) * h * (erf_v(y2[big] * k) - erf_v(y1[big] * k)) / dy[big])
    return out


def transfer_slab(tau_per_face, s0=1.0, n=400):
    """Front-to-back accumulation through an optically thick slab.

    Returns (I, T) after subdividing the slab into n sub-segments carrying
    tau_per_face each; I tends to s0 * (number of faces) for tau -> inf.
    """
    dt = tau_per_face / n
    acc, tr = 0.0, 1.0
    for _ in range(n):
        acc += tr * (1.0 - np.exp(-dt)) * s0
        tr *= np.exp(-dt)
    return acc, tr


# --------------------------------------------------------------------------- #
#  Numerics helpers                                                           #
# --------------------------------------------------------------------------- #
def convergence_table(b=6.0, steps=(64, 128, 256, 512, 1024, 2048, 4096, 8192)):
    """Azimuth error of the GLSL-style RK4 integrator vs the exact solution.

    The photon is started at r = 10^6 with impact parameter b; the exact
    azimuth swept until u falls below the escape value is
    ``2*phi_to_periapsis(b, u_escape)`` (symmetric about the periapsis).
    """
    u_escape = 1.0 / 1000.0
    phi_exact = 2.0 * float(np.atleast_1d(phi_to_periapsis(b, np.array([u_escape])))[0])
    rows = []
    for n in steps:
        h = 0.9 * phi_exact / n                       # whole sweep in n steps
        out = trace_ode(b, 1.0 / u_escape, u_escape, integrator="rk4", h=h)
        rows.append((n, abs(out["phi"] - phi_exact), h))
    return phi_exact, rows
