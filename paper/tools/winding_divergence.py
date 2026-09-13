"""Near-critical winding: the logarithmic divergence of the swept azimuth.

A photon that grazes the photon sphere at r_ph = 3M sweeps a total azimuth
between its two asymptotes that diverges logarithmically,

    phi_end(b) -> -ln(kappa) + const ,     kappa^2 = 1/b_c^2 - 1/b^2 ,

where kappa measures how far the periapsis sits from the photon sphere in
u = 1/r (the near-critical expansion of the effective potential is a downward
parabola,  g(u) = (1/3 - u)^2 - kappa^2 + O(mu^3), so the quadrature reduces to
arcosh(mu_0 / kappa)).  Because the *image-plane* offset of a pixel from the
critical curve is quadratic in kappa,

    delta_X = X/X_c - 1 ~ (dX/db) (b - b_c) / X_c  ~  C kappa^2 ,

the same divergence seen across the image plane keeps the *unit* slope in
ln delta_X (Bozza's near-critical deflection law alpha ~ -ln(b/b_c - 1) + const
has exactly this -1 coefficient),

    d phi_end / d ln kappa   = -2 ,
    d phi_end / d ln delta_X = -1 ,

which is the quantitative statement of "the higher-order images are compressed
into a logarithmically narrow band hugging the critical curve": every decade
of delta_X buys only ln 10 = 2.3026 rad ~ 0.366 of an extra turn.  (The -1
slope is *not* -1/2: delta_X is quadratic in kappa, but each factor of kappa
costs a full e-fold of swept azimuth, so the two powers cancel.)  This script
measures both slopes directly from the exact double-precision quadrature of
``grref`` and writes

    paper/figures/f25_winding_divergence.{pdf,png}
    paper/tools/winding_divergence.json

Nothing is transcribed by hand: every number in the figure comes from the
quadrature below, and every number quoted in the paper is read back from the
JSON this script writes.

    python paper/tools/winding_divergence.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grref as G                                          # noqa: E402
import figstyle as S                                       # noqa: E402
from figstyle import plt, C_ACCENT, C_ACCENT2, C_ACCENT3, C_ACCENT4, C_DIM  # noqa: E402

R0 = 26.0                 # observer radius, M
FOV = 58.0                # vertical field of view of the app's default preset
RS = 2.0
B_C = 3.0 * np.sqrt(3.0)  # = 5.196152422706632
TOOLS = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
#  Exact, cancellation-free mappings
# --------------------------------------------------------------------------- #
def kappa_from_b(b):
    """kappa = sqrt(1/b_c^2 - 1/b^2), evaluated without cancellation.

    The naive difference of two 1/b^2 of size 1/27 loses half of double
    precision as soon as b - b_c ~ 1e-14, which is exactly the regime this
    experiment probes; the factored form keeps full relative accuracy.
    """
    b = np.asarray(b, dtype=float)
    val = (b - B_C) * (b + B_C) / (B_C * B_C * b * b)
    return np.sqrt(np.maximum(val, 0.0))


def image_delta(b, r0=R0, fov=FOV):
    """delta = X/X_c - 1 for the pixel whose ray has impact parameter b.

    X = tan(psi)/tan(fov/2) with sin(psi) = b sqrt(1 - r_s/r0)/r0 is the
    normalised device coordinate of the ray at the observer.  Subtracting two
    nearly equal tangents would again destroy the small difference, so the
    numerator is rewritten as

        s sqrt(1 - s_c^2) - s_c sqrt(1 - s^2) = (s - s_c)(s + s_c)
                                                / (s sqrt(1-s_c^2)
                                                   + s_c sqrt(1-s^2))

    with s - s_c = (b - b_c) f / r0 available exactly.
    """
    b = np.asarray(b, dtype=float)
    f = np.sqrt(1.0 - RS / r0)
    s = b * f / r0
    s_c = B_C * f / r0
    c2 = np.sqrt(np.maximum(1.0 - s * s, 0.0))
    c2_c = np.sqrt(1.0 - s_c * s_c)
    num = (s - s_c) * (s + s_c)
    den = (s * c2_c + s_c * c2) * c2 * c2_c
    d_tan = num / den                       # = tan(psi) - tan(psi_c)
    return d_tan * c2_c / s_c               # divided by tan(psi_c)


def b_from_image_delta(delta_x, r0=R0):
    """Invert the exact map b -> delta_X by bisection on the impact parameter.

    ``image_delta`` is strictly increasing in b and the near-critical branch
    (delta_X -> 0 as b -> b_c+) is the one the paper cares about, so a plain
    bisection on [b_c, b_c (1 + 1e-1)] in log-space converges to machine
    precision in a few dozen steps and needs no external root finder.  This is
    what makes the spiral panel honest: the four curves are labelled by the
    *image-plane* offset, not by the (twice as large) relative offset of b.
    """
    target = float(delta_x)
    # bisect in u = ln(delta_b) with b = b_c (1 + e^u); u -> -inf is the
    # critical curve itself and u = 0 is b = 2 b_c, far outside the band
    lo, hi = np.log(1e-14), 0.0
    def f(u):
        return float(image_delta(np.array([B_C * (1.0 + np.exp(u))]), r0, FOV)[0]) - target
    # image_delta is increasing in b, hence increasing in u
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return B_C * (1.0 + np.exp(0.5 * (lo + hi)))


def phi_end_numeric(b, r0=R0, n=200001):
    """Total azimuth swept from r0 in to the periapsis and back out to r0.

    ``grref.phi_to_periapsis`` regularises the square-root endpoint with
    s = sqrt(u_1 - u) and integrates the resulting smooth integrand by the
    trapezoidal rule, so the value is a genuine double-precision quadrature of
    the exact orbit integral - not an ODE trajectory with a step limit.
    """
    out = np.empty_like(np.atleast_1d(np.asarray(b, dtype=float)))
    for i, bi in enumerate(np.atleast_1d(np.asarray(b, dtype=float))):
        half = G.phi_to_periapsis(float(bi), np.array([1.0 / r0]), n=n)
        out.flat[i] = 2.0 * float(np.atleast_1d(half)[0])
    # a scalar argument returns a Python float (numpy >= 2 refuses float() on
    # a one-element array, which would otherwise be a silent API trap)
    return out if np.ndim(b) > 0 else float(out[0])


def phi_end_parabolic(b, r0=R0):
    """Closed form of the same quantity in the near-critical parabola model.

    With mu = 1/3 - u the radicand becomes mu^2 - kappa^2, hence

        phi_half = arcosh(mu_0/kappa),   mu_0 = 1/3 - 1/r0 ,

    which for kappa << mu_0 is -ln kappa + ln(2 mu_0) + O(kappa^2).
    """
    k = np.maximum(kappa_from_b(b), 1e-300)
    mu0 = 1.0 / 3.0 - 1.0 / r0
    return 2.0 * np.arccosh(np.maximum(mu0 / k, 1.0 + 1e-15))


def _slope(x, y):
    """Least-squares slope of y against ln x, plus the residual RMS."""
    lx = np.log(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (s, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - (s * lx + c)
    return float(s), float(c), float(np.sqrt(np.mean(res ** 2)))


def _loglog_slope(x, y):
    """Least-squares slope of ln y against ln x, plus the residual RMS.

    Fitting in the log-log plane is what makes the exponent of a power law
    y = C x^p meaningful; a linear fit of y against ln x would be
    dimensionally meaningless and is what produced the earlier bogus exponent.
    """
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    (p, c), *_ = np.linalg.lstsq(A, ly, rcond=None)
    res = ly - (p * lx + c)
    return float(p), float(c), float(np.sqrt(np.mean(res ** 2)))


# --------------------------------------------------------------------------- #
#  Measurement
# --------------------------------------------------------------------------- #
def measure():
    delta_b = np.logspace(-12.0, -2.0, 61)      # b = b_c (1 + delta_b)
    b = B_C * (1.0 + delta_b)
    kappa = kappa_from_b(b)
    delta_x = image_delta(b)
    phi_num = phi_end_numeric(b)
    phi_par = phi_end_parabolic(b)

    # asymptotic regime: kappa small enough that the parabola model holds to
    # better than a per-cent while still being far from the double-precision
    # floor.  Fitting window stated explicitly so the number is reproducible.
    sel = (kappa > 1e-6) & (kappa < 1e-2)
    slope_k, c_k, rms_k = _slope(kappa[sel], phi_num[sel])
    slope_x, c_x, rms_x = _slope(delta_x[sel], phi_num[sel])

    # power law relating the two small quantities, fitted in the log-log plane
    # so that the exponent is the genuine d ln delta_X / d ln kappa ~ 2
    p_scale, c_scale, rms_scale = _loglog_slope(kappa[sel], delta_x[sel])

    # exact-difference check against the naive evaluation of delta_X.  The
    # reference is the *critical* pixel offset tan(psi_c)/tan(fov/2), which is
    # the physical zero of delta_X - not the last (far-from-critical) sample,
    # which is b = b_c (1 + 1e-2).
    naive = np.tan(np.arcsin(np.clip(b * np.sqrt(1.0 - RS / R0) / R0, 0, 1))) \
        / np.tan(np.radians(FOV) * 0.5)
    x_crit = np.tan(np.arcsin(np.clip(
        B_C * np.sqrt(1.0 - RS / R0) / R0, 0, 1))) / np.tan(np.radians(FOV) * 0.5)
    naive = naive / x_crit - 1.0

    # where the naive route is still well conditioned it must agree with the
    # factored one to double precision; the breakdown is confined to the
    # near-critical band, which is precisely what the experiment probes
    def naive_rel(d_b):
        bn = np.array([B_C * (1.0 + d_b)])
        nv = np.tan(np.arcsin(np.clip(bn * np.sqrt(1.0 - RS / R0) / R0, 0, 1))) \
            / np.tan(np.radians(FOV) * 0.5)
        exact = np.atleast_1d(image_delta(bn))
        return float(abs((nv / x_crit - 1.0)[0] / exact[0] - 1.0))

    naive_rel_checks = {("1e%+d" % int(np.round(np.log10(v)))): naive_rel(v)
                        for v in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)}

    # exact linear coefficient of the image-plane map in the critical band
    dX_over_db = float(image_delta(B_C * (1.0 + 1e-8))) / 1e-8

    # azimuth available inside one decade-wide image band, quoted in the paper
    def phi_at_dx(target):
        j = int(np.argmin(np.abs(np.log(delta_x) - np.log(target))))
        return dict(delta_x=float(delta_x[j]), phi_end=float(phi_num[j]),
                    loops=float(phi_num[j] / (2.0 * np.pi)))

    band = dict(inner=phi_at_dx(1e-8), outer=phi_at_dx(1e-3))
    band["delta_phi"] = band["inner"]["phi_end"] - band["outer"]["phi_end"]
    band["delta_loops"] = band["delta_phi"] / (2.0 * np.pi)

    # the four orbital-plane spirals of panel (d), labelled by the exact
    # image-plane offset (bisection, not the near-critical expansion)
    spiral = []
    for d in (1e-3, 1e-4, 1e-5, 1e-6):
        bb = b_from_image_delta(d)
        ph = float(np.atleast_1d(phi_end_numeric(bb))[0])
        spiral.append(dict(delta_x=d, b_over_bc=bb / B_C, phi_end=ph,
                           loops=ph / (2.0 * np.pi)))

    return dict(
        r0=R0, fov=FOV, b_c=B_C,
        delta_b=delta_b.tolist(), b=b.tolist(), kappa=kappa.tolist(),
        delta_x=delta_x.tolist(), phi_end=phi_num.tolist(),
        phi_end_parabolic=phi_par.tolist(),
        band_1e3_to_1e8=band,
        spirals=spiral,
        fit=dict(
            window_kappa=[float(kappa[sel].min()), float(kappa[sel].max())],
            slope_vs_ln_kappa=slope_k, intercept_vs_ln_kappa=c_k, rms_vs_ln_kappa=rms_k,
            slope_vs_ln_delta_x=slope_x, intercept_vs_ln_delta_x=c_x,
            rms_vs_ln_delta_x=rms_x,
            exponent_delta_x_in_kappa=p_scale, rms_of_power_law=rms_scale,
        ),
        # reference slopes the measurement should be compared against
        slope_reference_ln_kappa=-2.0,
        slope_reference_ln_delta_x=-1.0,
        # worst relative discrepancy of the naive delta_X against the factored
        # one - quantifies why the stable form is needed
        delta_x_naive_max_rel=float(np.max(np.abs(naive / delta_x - 1.0))),
        delta_x_naive_rel_at=naive_rel_checks,
        delta_x_per_delta_b=dX_over_db,
    )


# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
def figure(data):
    S.apply_style(dark=False)
    fig = plt.figure(figsize=(9.6, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    kappa = np.array(data["kappa"])
    delta_x = np.array(data["delta_x"])
    phi = np.array(data["phi_end"])
    phi_par = np.array(data["phi_end_parabolic"])
    fit = data["fit"]

    # -- (a) phi_end vs ln kappa -------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogx(kappa, phi, "o", ms=3.6, color=C_ACCENT, label="精确求积 $\\varphi_{\\rm end}(b)$", zorder=3)
    ax.semilogx(kappa, phi_par, "-", lw=1.2, color=C_ACCENT2,
                label="抛物近似 $2\\,{\\rm arcosh}(\\mu_0/\\kappa)$")
    kk = np.array([3e-6, 3e-3])
    ref = fit["slope_vs_ln_kappa"] * (np.log(kk) - np.log(kk[0])) + np.interp(np.log(kk[0]),
             np.log(kappa), phi)
    ax.semilogx(kk, ref, "--", lw=1.0, color=C_DIM,
                label="拟合斜率 $%.3f$   （参考 $-2$）" % fit["slope_vs_ln_kappa"])
    ax.set_xlabel("$\\kappa = \\sqrt{1/b_c^2 - 1/b^2}$  （近日点离光子球的 $u$ 距离）")
    ax.set_ylabel("$\\varphi_{\\rm end}$  [rad]")
    ax.set_title("(a) 方位角随 $\\kappa$ 对数发散", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=7.2)
    S.grid(ax, "both")

    # -- (b) delta_X vs kappa : the quadratic map --------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    ax.loglog(kappa, delta_x, "o", ms=3.6, color=C_ACCENT3)
    p = fit["exponent_delta_x_in_kappa"]
    kk = np.array([kappa.min() * 1.5, kappa.max() / 1.5])
    anchor = np.interp(np.log(kk[0]), np.log(kappa), np.log(delta_x))
    ax.loglog(kk, np.exp(anchor + p * (np.log(kk) - np.log(kk[0]))), "--", lw=1.1,
              color=C_DIM, label="拟合 $\\delta_X \\propto \\kappa^{%.3f}$   （参考 $2$）" % p)
    ax.set_xlabel("$\\kappa$")
    ax.set_ylabel("$\\delta_X = X/X_c - 1$  （像平面偏移）")
    ax.set_title("(b) 像平面偏移是 $\\kappa$ 的二次映射", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.2)
    S.grid(ax, "both")

    # -- (c) phi_end vs ln delta_X : the -1 slope --------------------------- #
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogx(delta_x, phi, "o", ms=3.6, color=C_ACCENT4,
                label="精确求积")
    dx = np.array([delta_x.min(), delta_x.max()])
    ax.semilogx(dx, fit["slope_vs_ln_delta_x"] * (np.log(dx) - np.log(dx[0]))
                + np.interp(np.log(dx[0]), np.log(delta_x), phi), "-", lw=1.2,
                color=C_ACCENT2,
                label="拟合斜率 $%.4f$   （参考 $-1$）" % fit["slope_vs_ln_delta_x"])
    ax.semilogx(dx, -1.0 * (np.log(dx) - np.log(dx[0]))
                + np.interp(np.log(dx[0]), np.log(delta_x), phi), ":", lw=1.4,
                color=C_DIM, label="参考斜率 $-1$")
    ax.set_xlabel("$\\delta_X$  （相对临界曲线的像平面距离）")
    ax.set_ylabel("$\\varphi_{\\rm end}$  [rad]")
    ax.set_title("(c) 每十倍 $\\delta_X$ 只多绕 $\\ln 10\\,(=2.303$ rad$)$", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7.2)
    S.grid(ax, "both")

    # -- (d) orbital-plane spirals ------------------------------------------ #
    ax = fig.add_subplot(gs[1, 1])
    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(2.0 * np.cos(th), 2.0 * np.sin(th), color="#101018", zorder=1)
    ax.plot(3.0 * np.cos(th), 3.0 * np.sin(th), "--", lw=1.0, color=C_DIM,
            label="光子球 $r_{\\rm ph}=3M$", zorder=2)
    for d, col in zip([1e-3, 1e-4, 1e-5, 1e-6],
                      [C_ACCENT2, C_ACCENT3, C_ACCENT4, C_ACCENT]):
        # the exact impact parameter whose *image-plane* offset is d, so the
        # legend label matches the panel-(c) abscissa exactly
        bb = b_from_image_delta(d)
        x, y = G.trajectory_full(bb, r_far=R0, n=2600)
        # every ray leaves the *observer*, so rotate the orbital plane until
        # the first sample sits back on the +x axis: the divergence is then
        # visible as the angle between the outgoing ends of the four curves
        th0 = np.arctan2(y[0], x[0])
        ca, sa = np.cos(-th0), np.sin(-th0)
        x, y = ca * x - sa * y, sa * x + ca * y
        ax.plot(x, y, lw=1.0, color=col, zorder=3,
                label="$\\delta_X=10^{%d}$: $\\varphi_{\\rm end}=%.2f$"
                      % (int(np.round(np.log10(d))),
                         float(np.atleast_1d(phi_end_numeric(bb))[0])))
    ax.plot([R0], [0.0], "o", ms=4.5, color="#1b1f2a", zorder=4)
    ax.text(R0 * 0.95, 1.5, "观测者", fontsize=7.0, ha="right", color="#1b1f2a")
    ax.set_xlim(-7, 27)
    ax.set_ylim(-14, 17)                    # headroom for the legend
    ax.set_aspect("equal")
    ax.set_xlabel("$x/M$")
    ax.set_ylabel("$y/M$")
    ax.set_title("(d) 轨道平面内的环绕：像阶次的来源（均自观测者出发）", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=6.4, ncol=2, framealpha=0.9,
              borderpad=0.35, columnspacing=0.9, handlelength=1.5)
    S.grid(ax, "both", alpha=0.25)

    fig.suptitle("近临界光线的对数发散：$\\varphi_{\\rm end}\\simeq-\\ln\\kappa$，像平面斜率为 $-1$",
                 fontsize=10.5, y=0.975)
    return S.save(fig, "f25_winding_divergence")


def main():
    data = measure()
    f = data["fit"]
    print("[winding] window kappa in [%.3e, %.3e]" % tuple(f["window_kappa"]))
    print("[winding] d phi / d ln kappa  = %.6f  (rms %.2e)" % (f["slope_vs_ln_kappa"], f["rms_vs_ln_kappa"]))
    print("[winding] d phi / d ln deltaX = %.6f  (rms %.2e)" % (f["slope_vs_ln_delta_x"], f["rms_vs_ln_delta_x"]))
    print("[winding] dev. from -2 / -1 refs  = %.2e / %.2e"
          % (abs(f["slope_vs_ln_kappa"] + 2.0), abs(f["slope_vs_ln_delta_x"] + 1.0)))
    print("[winding] deltaX ~ kappa^%.4f" % f["exponent_delta_x_in_kappa"])
    print("[winding] max rel. error of the naive delta_X = %.2e" % data["delta_x_naive_max_rel"])
    print("[winding]   naive rel. err by delta_b: %s"
          % "  ".join("%s=%.1e" % kv for kv in data["delta_x_naive_rel_at"].items()))
    print("[winding] d delta_X / d delta_b = %.6f  (critical band)" % data["delta_x_per_delta_b"])
    bnd = data["band_1e3_to_1e8"]
    print("[winding] deltaX 1e-3 -> 1e-8 : phi %.4f -> %.4f rad, extra %.4f rad = %.4f turns"
          % (bnd["outer"]["phi_end"], bnd["inner"]["phi_end"], bnd["delta_phi"], bnd["delta_loops"]))
    for sp in data["spirals"]:
        print("[winding]   deltaX=%.0e  b/b_c=%.9f  phi_end=%.4f rad = %.3f turns"
              % (sp["delta_x"], sp["b_over_bc"], sp["phi_end"], sp["loops"]))
    i = int(np.argmin(np.abs(np.array(data["kappa"]) - 1e-6)))
    print("[winding] phi_end(kappa=%.3e) = %.5f rad  (parabola %.5f, rel %.2e)"
          % (data["kappa"][i], data["phi_end"][i], data["phi_end_parabolic"][i],
             abs(data["phi_end"][i] / data["phi_end_parabolic"][i] - 1.0)))
    with open(os.path.join(TOOLS, "winding_divergence.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    png = figure(data)
    print("[winding] wrote", png)


if __name__ == "__main__":
    main()
