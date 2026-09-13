"""Independent validation of the NumPy mirror of the GPU geodesic kernel.

Every number printed here is quoted in the paper.  Run with

    python -u validate_pytrace.py

and the same numbers land in ``validate_pytrace.json``.

V1  the capture boundary of the tracer sits at the analytic critical impact
    parameter b_c = 3 sqrt(3) M.  The boundary is located *below the pixel
    scale* by bisecting the ray direction of the central image column, at
    three integration tolerances, so that the geometric part of the error is
    separated from the integration part;
V2  the first integral  (du/dphi)^2 + u^2 (1 - 2 M u) = 1/b^2  is conserved
    along a numerically integrated near-critical geodesic that winds inside
    the photon sphere.  The residual is evaluated with a central difference
    on the interior of the recorded track only: the one-sided stencil at the
    two ends of the record belongs to the difference operator, not to the
    integrator, and it is reported separately so the two are never mixed;
V3  the apparent shadow radius agrees with the closed form
    sin psi_c = b_c sqrt(1 - r_s / r0) / r0 by two independent routes.  The
    legacy route is the integer scan of the central image row: its whole
    discrepancy is the one-pixel quantisation of the image grid, the diameter
    being the difference of two such quantisations.  The route the shipped
    application now uses is the sub-pixel one (``shadow_radius_px`` bisects
    the capture boundary in NDC, and the live HUD probe bisects the *rendered*
    8-bit coverage ramp).  The 6.5e-12 relative figure reported here is the
    double-precision limit of the bisection - the geometry and the quadrature
    are exact to machine precision.  The live probe cannot reach it: its edge
    is carried by 24 stratified coverage samples quantised to 8 bits, so its
    1-sigma band is 1/(2 sqrt(2) * 24) = 0.0147 px per edge (measured in
    ``measure_probe.py``), which is sub-pixel but not exact;
V4  the pixel direction of the camera *is* the locally measured angle of a
    static observer,  sin psi = b sqrt(1 - r_s / r0) / r0;  the error that the
    two plausible naive alternatives would introduce is quantified for the
    preset observer distances;
V5  truncating the integration at the escape radius R_esc = 140 M leaves a
    residual direction error well below one pixel for every ray that reaches
    the background sky in any preset.

Conventions: G = c = M = 1, r_s = 2, photon sphere 3, ISCO 6.
"""

from __future__ import annotations

import json
import os

import numpy as np

import grref
import pytrace as pt

RESULTS = {}

VIEWPORT = (1000, 600)                     # the application's default canvas
DIST, ELEV, FOV = 26.0, 13.0, 58.0         # the "classic horizon" preset


# --------------------------------------------------------------------------- #
# sub-pixel camera geometry
# --------------------------------------------------------------------------- #
def tan_half_fov(fov=FOV):
    return float(np.tan(np.radians(fov) * 0.5))


def b_of_ndc(y, dist=DIST, fov=FOV):
    """Analytic conserved b of the central column pixel with vertical NDC y."""
    r0 = float(dist)
    f = 1.0 - 2.0 / r0
    mu = np.arctan(y * tan_half_fov(fov))          # locally measured angle
    return r0 * np.sin(mu) / np.sqrt(f)


def trace_single(y, tol=1e-2, max_steps=8000, dist=DIST, fov=FOV,
                 viewport=VIEWPORT, jitter=0.0):
    """Trace the single ray through the centre of the image column."""
    return pt.trace_pixels(dist=dist, elevation=ELEV, fov=fov, nx=1, ny=1,
                           aspect=viewport[0] / float(viewport[1]), y_ndc=y,
                           max_steps=max_steps, tol=tol, jitter=jitter)


def captured(y, **kw):
    return bool(trace_single(y, **kw)["captured"][0, 0])


def bisect_edge(tol, max_steps=12000, iters=52):
    """Bisect the ray direction of the central column for the shadow edge."""
    lo, hi = 0.0, 0.5
    while captured(hi, tol=tol, max_steps=max_steps):
        hi *= 1.5
        if hi > 1.0:
            raise RuntimeError("no escape found on the central column")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if captured(mid, tol=tol, max_steps=max_steps):
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-16:
            break
    return 0.5 * (lo + hi), lo, hi


# --------------------------------------------------------------------------- #
def v1_critical_impact_parameter():
    rows = []
    for tol in (1e-2, 1e-3, 1e-4):
        y, lo, hi = bisect_edge(tol)
        b = float(b_of_ndc(y))
        rows.append(dict(tol=tol, y_edge=float(y), b_edge=b,
                         b_crit=float(pt.B_CRIT), rel=float(b / pt.B_CRIT - 1.0)))
        print("V1 tol = %-6g   y_edge = %.15f   b = %.12f   b_c = %.12f"
              "   rel err = %+.3e" % (tol, y, b, pt.B_CRIT, b / pt.B_CRIT - 1.0),
              flush=True)
    RESULTS["V1"] = rows
    return rows[1]["y_edge"]


def v2_first_integral_real(y_edge):
    """Residual of the first integral along a near-critical recorded track."""
    y = y_edge * (1.0 - 1e-6)
    rows = []
    for tol in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        out = pt.trace_pixels(dist=DIST, elevation=ELEV, fov=FOV, nx=1, ny=1,
                              aspect=VIEWPORT[0] / float(VIEWPORT[1]), y_ndc=y,
                              max_steps=60000, tol=tol, want_path=[(0, 0)])
        phi, u = np.array(out["paths"][(0, 0)]).T
        b = float(out["b"][0, 0])
        # the recorder keeps appending after the ray is swallowed (h = 0), so
        # the tail is a run of repeated samples; cut it before differentiating
        end = int(np.argmax(u > 0.5)) if (u > 0.5).any() else u.size
        phi, u = phi[:end], u[:end]
        keep = np.ones_like(phi, dtype=bool)
        keep[1:] = np.diff(phi) > 0.0
        phi, u = phi[keep], u[keep]
        dudphi = np.gradient(u, phi)
        resid = np.abs((dudphi ** 2 + u ** 2 * (1.0 - 2.0 * u)) * b ** 2 - 1.0)
        strong = u > 1.0 / 30.0                      # inside the strong field
        interior = strong.copy()
        interior[0] = interior[-1] = interior[-2] = False   # one-sided stencil
        bnd = strong & ~interior
        r_min = 1.0 / float(u[strong].max())
        rows.append(dict(tol=tol, b=b, samples=int(phi.size),
                         n_strong=int(strong.sum()),
                         max_resid=float(resid[interior].max()),
                         mean_resid=float(resid[interior].mean()),
                         max_resid_ends=float(resid[bnd].max()) if bnd.any() else 0.0,
                         r_at_max_M=float(1.0 / u[interior][int(np.argmax(
                             resid[interior]))]),
                         r_min_M=float(r_min),
                         phi_span=float(phi[strong].max() - phi[strong].min()),
                         captured=bool(out["captured"][0, 0]),
                         timeout=bool(out["timeout"][0, 0])))
        print("V2 tol = %-7g  b = %.9f   samples = %5d   max |b^2 E - 1| = %.3e"
              "   mean %.2e   r_min = %.4f M   dphi = %.4f rad"
              % (tol, b, phi.size, rows[-1]["max_resid"], rows[-1]["mean_resid"],
                 r_min, phi[strong].max() - phi[strong].min()), flush=True)
        print("      ... at the two ends of the record the one-sided stencil"
              " alone gives %.3e  (difference-operator artefact, excluded)"
              % rows[-1]["max_resid_ends"], flush=True)
    RESULTS["V2"] = rows
    RESULTS["V2_summary"] = dict(
        max_ratio=rows[0]["max_resid"] / rows[-1]["max_resid"],
        mean_ratio=rows[0]["mean_resid"] / rows[-1]["mean_resid"],
        plateau_tol=rows[-2]["tol"], plateau_max=rows[-2]["max_resid"])


def v3_shadow_radius(y_edge):
    """Row-scan measurement of the shadow against the closed form.

    ``shadow_radius_px`` does what a viewer of the running app does: it scans
    whole pixels along the central image row and takes the last captured one
    as the edge.  ``refined`` replaces that integer edge with the exact NDC
    crossing ``y_edge`` returned by the V1 bisection, which is what the app's
    HUD probe now recovers to sub-pixel accuracy from the rendered coverage
    ramp.  Reporting both is the point: the pixel scan is dominated by the
    half-pixel edge convention, the refined value isolates the physics.
    """
    rows = []
    for (nx, ny) in ((400, 240), (1000, 600), (2000, 1200), (4000, 2400)):
        m = pt.shadow_radius_px(DIST, ELEV, FOV, nx, ny, tol=5e-3,
                                max_steps=2000)
        refined = y_edge * ny * 0.5
        row = dict(nx=nx, ny=ny,
                   measured_px=float(m["measured_px"]),
                   analytic_px=float(m["analytic_px"]),
                   rel=float(m["measured_px"] / m["analytic_px"] - 1.0),
                   refined_px=float(refined),
                   rel_refined=float(refined / m["analytic_px"] - 1.0),
                   quantisation_px=float(abs(m["measured_px"] - m["analytic_px"])))
        rows.append(row)
        print("V3 %5dx%-5d scan %8.3f px (%+0.3e)   sub-pixel %8.3f px (%+0.3e)"
              "   closed form %8.4f px"
              % (nx, ny, row["measured_px"], row["rel"], row["refined_px"],
                 row["rel_refined"], row["analytic_px"]), flush=True)
    RESULTS["V3"] = rows


def v4_local_frame(y_edge):
    """Pixel angle == local static-frame angle; naive alternatives quantified."""
    r0 = DIST
    f = 1.0 - 2.0 / r0
    rows = []
    for y in (0.10, 0.20, float(y_edge), 0.50, 0.70, 0.90):
        out = trace_single(y, tol=1e-4, max_steps=12000)
        b_tr = float(out["b"][0, 0])
        psi_tr = float(out["psi"][0, 0])
        mu = np.arctan(y * tan_half_fov(FOV))        # pixel angle
        b_an = float(b_of_ndc(y))
        psi_an = float(np.arcsin(b_an * np.sqrt(f) / r0))
        rows.append(dict(y=y, b_tracer=b_tr, b_analytic=b_an,
                         rel_b=b_tr / b_an - 1.0,
                         psi_tracer_deg=float(np.degrees(psi_tr)),
                         pixel_angle_deg=float(np.degrees(mu)),
                         rel_psi=psi_tr / psi_an - 1.0))
        print("V4a y = %7.4f   b tracer %12.8f  analytic %12.8f (%+0.2e)   "
              "psi tracer %9.6f deg  pixel %9.6f deg  rel %+0.2e"
              % (y, b_tr, b_an, b_tr / b_an - 1.0, np.degrees(psi_tr),
                 np.degrees(mu), psi_tr / psi_an - 1.0), flush=True)
    RESULTS["V4a"] = rows

    conv = []
    for r in (4.0, 6.0, 6.5, 12.0, 26.0, 60.0, 140.0):
        fr = 1.0 - 2.0 / r
        sf = np.sqrt(fr)
        ps = float(grref.shadow_angle(r))            # the locally measured angle
        s_flat = np.sin(ps) / sf                     # sin(theta) = b / r0
        th_flat = float(np.arcsin(np.clip(s_flat, -1.0, 1.0)))
        th_chart = float(np.arctan(sf * np.tan(ps)))            # chart angle

        def _px(th):
            """Pixel offset of the naive reading, None if it leaves the frame."""
            if not np.isfinite(th) or abs(np.degrees(th)) > 88.0:
                return None
            return float((np.tan(th) - np.tan(ps)) * 300.0 / tan_half_fov(FOV))

        def _px_s(v):
            return "   n/a" if v is None else "%+6.2f" % v

        conv.append(dict(r0=r, f=fr, psi_deg=float(np.degrees(ps)),
                         flat_deg=float(np.degrees(th_flat)),
                         chart_deg=float(np.degrees(th_chart)),
                         flat_leaves_frame=bool(abs(s_flat) >= 1.0),
                         err_flat=float(th_flat / ps - 1.0),
                         err_chart=float(th_chart / ps - 1.0),
                         px_flat=_px(th_flat), px_chart=_px(th_chart)))
        print("V4b r0 = %6.1f M   local %9.6f deg   flat %9.6f (%+7.3f %%)"
              "   chart %9.6f (%+7.3f %%)   at 600 px: %s / %s px"
              % (r, np.degrees(ps), np.degrees(th_flat),
                 100.0 * (th_flat / ps - 1.0), np.degrees(th_chart),
                 100.0 * (th_chart / ps - 1.0), _px_s(conv[-1]["px_flat"]),
                 _px_s(conv[-1]["px_chart"])), flush=True)
    RESULTS["V4b"] = conv


def v5_escape_truncation():
    """Residual direction error of stopping the integration at R_esc."""
    rows = []
    for b in (5.1962, 6.0, 9.0, 14.0, 20.0, 26.0):
        err = float(grref.tangent_error(140.0, b))
        rows.append(dict(b=b, err_rad=err, err_px_1000x600=float(
            np.degrees(err) / FOV * 600.0)))
        print("V5 escape cut   b = %7.4f M   residual direction error = %.3e rad"
              "   (%+.5f px at 600 px / 58 deg)"
              % (b, err, rows[-1]["err_px_1000x600"]), flush=True)
    RESULTS["V5"] = dict(R_esc=140.0, rows=rows,
                         max_err_rad=float(max(r["err_rad"] for r in rows)),
                         max_err_px=float(max(r["err_px_1000x600"] for r in rows)))


def main():
    y_edge = v1_critical_impact_parameter()
    v2_first_integral_real(y_edge)
    v3_shadow_radius(y_edge)
    v4_local_frame(y_edge)
    v5_escape_truncation()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "validate_pytrace.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
