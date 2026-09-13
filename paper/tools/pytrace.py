"""A vectorised NumPy mirror of the GLSL geodesic kernel of the simulator.

The paper uses this module to obtain *independent* results for the same
physical model that the WebGL shader renders: identical orbit equation,
identical static-observer local frame, identical adaptive step controller and
identical escape criterion - but evaluated in IEEE double precision, without
any GPU-specific shortcut, and from a Python process that never touches the
shader sources.

Agreement between this tracer and the analytic quadratures of ``grref`` is a
check of the *physics*; agreement between this tracer and the rendered frames
is a check of the *GPU implementation*.

Conventions (geometric units, G = c = 1):

    M = 1,  r_s = 2M,  photon sphere 3M,  ISCO 6M,  b_c = 3 sqrt(3) M.

The disk lies in the plane y = 0 and rotates about +y.
"""

from __future__ import annotations

import numpy as np

M = 1.0
RS = 2.0
R_PHOTON = 3.0
R_ISCO = 6.0
B_CRIT = 3.0 * np.sqrt(3.0)


# --------------------------------------------------------------------------- #
def camera_basis(dist: float, elevation_deg: float):
    """Camera position and the (right, up, forward) basis of three.js lookAt."""
    el = np.radians(elevation_deg)
    pos = np.array([dist * np.cos(el), dist * np.sin(el), 0.0])
    back = pos / np.linalg.norm(pos)
    up_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(up_world, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    basis = np.stack([right, up, -back], axis=1)      # columns
    return pos, basis


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
def trace_pixels(dist=26.0, elevation=13.0, fov=58.0, nx=240, ny=135,
                 max_steps=600, tol=0.03, step_scale=1.0, escape_r=None,
                 r_in=R_ISCO, r_out=26.0, jitter=0.0, seed=12345,
                 want_path=None, want_path_frac=None, want_dir=False,
                 aspect=None,
                 y_ndc=None, ndc_x=None, ndc_y=None):
    """Trace one null geodesic per pixel; return per-pixel diagnostic arrays.

    Parameters mirror the uniform block of ``GEODESIC_FRAG``; ``want_path`` may
    hold a tuple of (row, col) pixel indices for which the full (x, y) ground
    track in the orbital plane is recorded as well.

    ``aspect`` overrides the horizontal/vertical pixel-scale ratio (default
    ``nx/ny``) and ``y_ndc`` replaces the vertical normalised device
    coordinate of the fan (default: the usual ``ny`` evenly spaced rows).
    Both exist so that a *single image row* can be traced with the exact pixel
    scale it would have inside a full ``nx`` x ``ny`` frame - the shadow-radius
    measurement below needs no more than that and this makes it two orders of
    magnitude cheaper than scanning the whole frame.

    Returns a dict with 2-D arrays (ny, nx):

    ``captured``  ray crossed the horizon            (bool)
    ``escaped``   ray reached r > escape radius      (bool)
    ``timeout``   ray ran out of the step budget     (bool)
    ``r_hit``     radius of the *first* disk-plane crossing inside the annulus
    ``n_cross``   number of annulus crossings (image order = n_cross)
    ``b``         conserved impact parameter b = L/E
    ``psi``       locally measured arrival angle at the observer [rad]
    ``phi_end``   swept azimuth at the end of the integration
    ``delay``     coordinate path length proxy  int r^2/sqrt(1-2M/r) dphi
    ``tdir``      unit propagation direction at the end of the integration
                  (the direction in which the star field is sampled), only
                  filled in when ``want_dir`` is set
    """
    if escape_r is None:
        escape_r = 140.0
    pos, basis = camera_basis(dist, elevation)
    r0 = float(np.linalg.norm(pos))
    u0 = 1.0 / r0
    f_obs = np.sqrt(max(1.0 - 2.0 * u0, 1e-6))
    E1 = pos / r0

    xs = (np.arange(nx) + 0.5) / nx * 2.0 - 1.0
    if y_ndc is None:
        ys = 1.0 - (np.arange(ny) + 0.5) / ny * 2.0
    else:
        ys = np.full(ny, float(y_ndc))
    # Optional crop of the image plane, in the same normalised device
    # coordinates the full frame uses: ``ndc_x = (x_left, x_right)`` and
    # ``ndc_y = (y_bottom, y_top)``.  Both default to the full frame, so the
    # crop only re-parameterises the very same ray grid; it exists so that the
    # paper can zoom into the neighbourhood of the critical curve, where the
    # higher-order images live.
    x0, x1 = (-1.0, 1.0) if ndc_x is None else (float(ndc_x[0]), float(ndc_x[1]))
    y0, y1 = (-1.0, 1.0) if ndc_y is None else (float(ndc_y[0]), float(ndc_y[1]))
    xs = x0 + (xs + 1.0) * 0.5 * (x1 - x0)
    if y_ndc is None:
        ys = y0 + (ys + 1.0) * 0.5 * (y1 - y0)
    X, Y = np.meshgrid(xs, ys)
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        X = X + rng.uniform(-1, 1, X.shape) * jitter / nx
        Y = Y + rng.uniform(-1, 1, X.shape) * jitter / ny
    tan_half = np.tan(np.radians(fov) * 0.5)
    if aspect is None:
        # square pixels: one pixel must subtend the same angle in x and in y
        aspect = (nx / float(ny)) * ((y1 - y0) / (x1 - x0))

    d = (basis[:, 0] * (X * tan_half * aspect)[..., None]
         + basis[:, 1] * (Y * tan_half)[..., None]
         + basis[:, 2])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    a_dir = np.sum(d * E1, axis=-1)
    btan = np.sqrt(np.maximum(1.0 - a_dir ** 2, 0.0))
    radial_ray = btan < 1e-5
    btan_safe = np.maximum(btan, 1e-5)

    cr = np.cross(np.broadcast_to(pos, d.shape), d)
    cr2 = np.sum(cr * cr, axis=-1)
    cN = np.empty_like(cr)
    good = cr2 > 1e-12
    cN[good] = cr[good] / np.sqrt(cr2[good])[..., None]
    if not good.all():                                  # degenerate columns
        axis = np.array([0.0, 1.0, 0.0]) if abs(E1[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        fallback = np.cross(E1, axis)
        cN[~good] = fallback / np.linalg.norm(fallback)
    E2 = np.cross(cN, E1)
    E2 /= np.linalg.norm(E2, axis=-1, keepdims=True)
    flip = np.sum(d * E2, axis=-1) < 0.0
    E2 = np.where(flip[..., None], -E2, E2)

    du = -u0 * (a_dir / btan_safe) * f_obs
    b_imp = 1.0 / np.sqrt(np.maximum(du * du + u0 * u0 * (1.0 - 2.0 * u0), 1e-12))
    psi = np.arcsin(np.clip(b_imp * f_obs / r0, 0.0, 1.0))

    u = np.full((ny, nx), u0)
    duv = du.copy()
    phi = np.zeros((ny, nx))
    captured = np.zeros((ny, nx), dtype=bool)
    escaped = np.zeros((ny, nx), dtype=bool)
    r_hit = np.full((ny, nx), np.nan)
    n_cross = np.zeros((ny, nx), dtype=np.int32)
    delay = np.zeros((ny, nx))

    # radial and captured-by-construction columns: a < 0 falls straight in
    captured |= radial_ray & (a_dir < 0.0)
    escaped |= radial_ray & (a_dir >= 0.0)

    u_min = 1.0 / max(escape_r, 1.6 * r0)

    paths = {}
    if want_path is not None:
        for (rr, cc) in want_path:
            paths[(rr, cc)] = [(0.0, u0)]

    for _ in range(max_steps):
        live = ~(captured | escaped)
        if not live.any():
            break
        now_cap = live & (u > 0.5)
        captured |= now_cap
        now_esc = live & (u < u_min) & (duv < 0.0)
        escaped |= now_esc
        live = live & ~(now_cap | now_esc)
        if not live.any():
            break

        hmax = 0.35 - 0.30 * _smoothstep(0.0, 0.30, u)
        h = np.clip(tol * u / np.maximum(np.abs(duv), 1e-7), 0.002, hmax) * step_scale
        h = np.where(live, h, 0.0)

        def f(uu, aa):
            return aa, -uu + 3.0 * uu * uu

        k1 = f(u, duv)
        k2 = f(u + 0.5 * h * k1[0], duv + 0.5 * h * k1[1])
        k3 = f(u + 0.5 * h * k2[0], duv + 0.5 * h * k2[1])
        k4 = f(u + h * k3[0], duv + h * k3[1])
        u_n = u + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        du_n = duv + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

        c1, s1 = np.cos(phi), np.sin(phi)
        phi2 = phi + h
        c2, s2 = np.cos(phi2), np.sin(phi2)
        with np.errstate(divide="ignore", invalid="ignore"):
            p1 = (c1[..., None] * E1 + s1[..., None] * E2) / np.maximum(u, 1e-7)[..., None]
            p2 = (c2[..., None] * E1 + s2[..., None] * E2) / np.maximum(u_n, 1e-7)[..., None]

        y1, y2 = p1[..., 1], p2[..., 1]
        crossing = live & (y1 * y2 < 0.0)
        if crossing.any():
            t = y1 / np.where(np.abs(y1 - y2) < 1e-30, 1e-30, y1 - y2)
            pm = p1 + t[..., None] * (p2 - p1)
            rm = np.linalg.norm(pm, axis=-1)
            inside = crossing & (rm > r_in * 0.90) & (rm < r_out * 1.05)
            first = inside & np.isnan(r_hit)
            r_hit = np.where(first, rm, r_hit)
            n_cross = n_cross + inside.astype(np.int32)
            # coordinate-time proxy: the largest contribution is the
            # "photon travel time" r^2/sqrt(1-rs/r) dphi along the segment
            seg = 0.5 * (1.0 / np.maximum(u, 1e-7) ** 2 + 1.0 / np.maximum(u_n, 1e-7) ** 2)
            grav = 1.0 / np.sqrt(np.maximum(1.0 - 2.0 * np.maximum(u, u_n), 1e-6))
            delay = delay + np.where(live, seg * grav * np.abs(h), 0.0)
        else:
            delay = delay + np.where(live, 0.0, 0.0)

        # record requested orbital-plane tracks
        if paths:
            for key in paths:
                rr, cc = key
                paths[key].append((float(phi2[rr, cc]), float(u_n[rr, cc])))

        phi, u, duv = phi2, u_n, du_n
        run_off = live & (u < 1e-6)
        u = np.where(run_off, 1e-6, u)
        escaped |= run_off

    timeout = ~(captured | escaped)

    tdir = None
    if want_dir:
        # p(phi) = (cos(phi) E1 + sin(phi) E2) / u  ->  dp/dphi is the tangent
        # of the orbit; normalising it gives the sky direction that the shader
        # uses for the background star lookup.
        c, s = np.cos(phi), np.sin(phi)
        uu = np.maximum(u, 1e-7)
        pp = (c[..., None] * E1 + s[..., None] * E2) / uu[..., None]
        dp = ((-s[..., None] * E1 + c[..., None] * E2) / uu[..., None]
              - pp * (duv / uu)[..., None])
        nrm = np.linalg.norm(dp, axis=-1, keepdims=True)
        tdir = dp / np.maximum(nrm, 1e-30)

    return dict(captured=captured, escaped=escaped, timeout=timeout, r_hit=r_hit,
                n_cross=n_cross, b=b_imp, psi=psi, phi_end=phi, delay=delay,
                paths=paths, pos=pos, basis=basis, r0=r0, tdir=tdir)


# --------------------------------------------------------------------------- #
def analytic_shadow_angle(r0):
    """sin(psi_c) = b_c sqrt(1 - rs/r0) / r0."""
    return float(np.arcsin(min(1.0, B_CRIT * np.sqrt(1.0 - RS / r0) / r0)))


def shadow_radius_px(dist, elevation, fov, nx, ny, tol=0.01, max_steps=1500):
    """Independently measure the apparent shadow radius with this tracer.

    The measurement reproduces the one used by the running application: scan
    the central image row for the last dark pixel on either side.

    ``ny`` is the height of the *viewport* the row belongs to; it only fixes
    the pixel scale.  Only that single row of rays is integrated.
    """
    out = trace_pixels(dist=dist, elevation=elevation, fov=fov, nx=nx, ny=1,
                       aspect=nx / float(ny), y_ndc=0.0,
                       max_steps=max_steps, tol=tol)
    row = out["captured"][0]
    dark = np.where(row)[0]                 # captured pixels are the dark shadow
    if dark.size == 0:
        return dict(measured_px=np.nan, analytic_px=analytic_px(dist, fov, nx, ny),
                    rel=np.nan)
    lo, hi = dark.min(), dark.max()
    measured = 0.5 * (hi - lo) + 0.5
    return dict(measured_px=float(measured), analytic_px=analytic_px(dist, fov, nx, ny),
                rel=float(measured / analytic_px(dist, fov, nx, ny) - 1.0))


def analytic_px(dist, fov, nx, ny):
    """Pixel radius of the shadow edge for a height = ny, fov = fov viewport."""
    ang = analytic_shadow_angle(dist)
    return (ny * 0.5) * np.tan(ang) / np.tan(np.radians(fov) * 0.5)


if __name__ == "__main__":                              # tiny self-test
    import time
    t0 = time.time()
    res = trace_pixels(nx=160, ny=90, max_steps=500)
    dt = time.time() - t0
    frac = res["captured"].mean()
    print("160x90 in %.2f s, captured fraction %.4f" % (dt, frac))
    print("shadow angle %.8f rad" % analytic_shadow_angle(26.0))
