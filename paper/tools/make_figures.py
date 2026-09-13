"""Generate the analysis / validation figures of the paper.

    python paper/tools/make_figures.py

Writes vector PDFs to paper/figures/ and 200-dpi PNGs to paper/figures/png/.
Everything is computed from ``grref`` - no hand-entered numbers.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grref as G  # noqa: E402
import figstyle as S  # noqa: E402
from figstyle import plt, C_ACCENT, C_ACCENT2, C_ACCENT3, C_ACCENT4, C_WARM, C_DIM  # noqa: E402


# --------------------------------------------------------------------------- #
def f01_geodesics():
    S.apply_style(dark=True)
    fig = plt.figure(figsize=(7.4, 3.4))
    ax = fig.add_subplot(121, projection="polar")
    ax2 = fig.add_subplot(122)

    bs = [4.0, 5.0, 5.196152422706632 + 1e-6, 5.6, 6.5, 9.0, 14.0]
    labels = ["$b=4$（捕获）", "$b=5$（捕获）", "$b\\to b_c$（临界）", "$b=5.6$", "$b=6.5$", "$b=9$", "$b=14$"]
    for b, lab in zip(bs, labels):
        try:
            x, y = G.trajectory_full(b, 60.0)
        except ValueError:
            continue
        r = np.hypot(x, y)
        keep = r >= 2.0
        th = np.unwrap(np.arctan2(y, x))
        lw = 1.6 if abs(b - G.B_CRIT) < 1e-3 else 1.0
        style = "-" if not G.capture_probability(b) else "--"
        ax.plot(th[keep], r[keep], style, lw=lw, label=lab)
        ax2.plot(x[keep], y[keep], style, lw=lw)

    ax.set_ylim(0, 26)
    ax.set_yticks([5, 10, 15, 20])
    ax.set_yticklabels(["5", "10", "15", "20"])
    ax.set_title("轨道平面内的光子轨迹 $r(\\varphi)$", pad=12)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.16, -0.08), fontsize=7, ncol=2)

    th = np.linspace(0, 2 * np.pi, 600)
    for rad, col, lab, ls in ((2.0, "#e0574a", "视界 $r_s$", "-"),
                              (3.0, C_WARM, "光子球 $3M$", "--"),
                              (6.0, C_ACCENT2, "ISCO $6M$", ":")):
        ax.plot(th, np.full_like(th, rad), color=col, lw=1.0, ls=ls, alpha=0.9)
    ax2.add_patch(plt.Circle((0, 0), 2.0, color="#0a0d14", ec="#e0574a", lw=1.0, zorder=5))
    ax2.add_patch(plt.Circle((0, 0), 3.0, fill=False, ec=C_WARM, lw=0.9, ls="--"))
    ax2.add_patch(plt.Circle((0, 0), 6.0, fill=False, ec=C_ACCENT2, lw=0.9, ls=":"))
    ax2.set_xlim(-26, 26)
    ax2.set_ylim(-20, 20)
    ax2.set_aspect("equal")
    ax2.set_title("笛卡尔视图（原点为黑洞）")
    ax2.set_xlabel("$x/M$")
    ax2.set_ylabel("$y/M$")
    for a in (ax2,):
        a.tick_params(labelsize=8)
    return S.save(fig, "f01_geodesic_rays")


# --------------------------------------------------------------------------- #
def f02_deflection():
    S.apply_style()
    bs = np.exp(np.linspace(np.log(G.B_CRIT * 1.0002), np.log(3e4), 400))
    alpha = np.array([G.deflection_exact(b, n=40000) for b in bs])
    series = G.weak_field_series(bs)
    weak1 = 4.0 / bs

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    ax.loglog(bs, alpha, color=C_ACCENT2, label="精确积分 $\\alpha(b)$")
    ax.loglog(bs, weak1, "--", color=C_DIM, label="$4M/b$（一阶）")
    ax.loglog(bs, series, ":", color=C_ACCENT, label="四阶级数")
    ax.axvline(G.B_CRIT, color="#b5533f", lw=1.0)
    ax.text(G.B_CRIT * 1.05, 6, "$b_c=3\\sqrt{3}M$", color="#b5533f", fontsize=8, rotation=90, va="top")
    ax.set_xlabel("碰撞参数 $b/M$")
    ax.set_ylabel("偏折角 $\\alpha$ [rad]")
    ax.set_title("光线偏折角")
    ax.legend(loc="upper right")
    S.grid(ax)

    ax = axs[1]
    rel = np.abs(alpha - series) / series
    ax.loglog(bs, rel, color=C_ACCENT3)
    ax.axhline(1e-3, color=C_DIM, ls="--", lw=0.8)
    ax.text(bs[8], 1.3e-3, "0.1%", color=C_DIM, fontsize=8)
    ax.set_xlabel("碰撞参数 $b/M$")
    ax.set_ylabel("$|\\alpha_{\\rm exact}-\\alpha_{\\rm series}|/\\alpha$")
    ax.set_title("与解析级数的相对偏差")
    ax.set_ylim(1e-10, 1)
    S.grid(ax)
    return S.save(fig, "f02_deflection")


# --------------------------------------------------------------------------- #
def f03_shadow():
    S.apply_style()
    r0 = np.linspace(6.2, 200, 500)
    psi = np.degrees(G.shadow_angle(r0))
    # measured values come from paper/tools/r0_scan.json, written by
    # tools/verify.py while it drives the shipped application (1600x900).
    scan = None
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "r0_scan.json"), encoding="utf-8") as fh:
            scan = json.load(fh)
    except (OSError, ValueError):
        scan = None
    if scan is None:
        raise RuntimeError("f03 needs paper/tools/r0_scan.json (run tools/verify.py)")
    meas = np.array([[float(r["r0"]), float(r["measured"]), float(r["analytic"])]
                     for r in scan["rows"]])
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    ax.plot(r0, psi, color=C_ACCENT2, label="解析解 $\\arcsin(\\,b_c\\sqrt{1-r_s/r_0}\\,/\\,r_0)$")
    ax.set_xlabel("观测者半径 $r_0/M$")
    ax.set_ylabel("阴影张角半径 [deg]")
    ax.set_title("视界阴影的角半径")
    ax.legend(loc="upper right")
    S.grid(ax)

    ax = axs[1]
    # Residual in pixels.  The edge estimate is limited by the coverage
    # quantisation of the in-app probe, sigma = 1 / (2 sqrt(2) N_p); with
    # N_p = SHADOW_PROBE_SAMPLES = 24 (web/src/main.js) that is 0.0147 px,
    # independent of the viewport.
    sigma_px = 1.0 / (2.0 * np.sqrt(2.0) * 24.0)
    resid = meas[:, 1] - meas[:, 2]
    rel = resid / meas[:, 2] * 100
    ax.axhline(0, color=C_DIM, lw=0.8)
    ax.axhspan(-sigma_px, sigma_px, color=C_ACCENT2, alpha=0.20,
               label="覆盖率量化带 $\\pm\\sigma_{\\rm px}=0.0147$\\,px")
    ax.semilogx(meas[:, 0], resid, "o", ms=5.0, color=C_ACCENT,
                label="实机探针 $-$ 闭式")
    for x, y, r in zip(meas[:, 0], resid, rel):
        ax.annotate("%+.2f\\,px\n%+.2f\\%%" % (y, r), (x, y),
                    textcoords="offset points", xytext=(4, 6), fontsize=5.6,
                    color=C_ACCENT)
    ax.set_xlabel("观测者半径 $r_0/M$")
    ax.set_ylabel("残差 $R_{\\rm px}^{\\rm meas}-R_{\\rm px}^{\\rm cl}$  [px]")
    ax.set_title("阴影半径的实机残差（$1600\\times900$）")
    ax.set_xlim(6.5, 260)
    ax.set_ylim(-0.16, 0.16)
    ax.legend(loc="lower left")
    S.grid(ax)
    return S.save(fig, "f03_shadow_radius")


# --------------------------------------------------------------------------- #
def f04_phase():
    S.apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    us = np.linspace(1e-4, 0.5, 1200)
    for b, col, lab in ((4.0, C_DIM, "$b<b_c$"), (G.B_CRIT, "#b5533f", "$b=b_c$ (分界)"),
                        (6.0, C_ACCENT2, "$b=6M$"), (10.0, C_ACCENT, "$b=10M$"),
                        (20.0, C_ACCENT3, "$b=20M$")):
        g = G.sqrt_g(us, b)
        m = us <= (G.turning_point(b) if G.turning_point(b) else 0.5)
        ax.plot(us[m], g[m], color=col, label=lab,
                lw=1.6 if lab.startswith("$b=b_c$") else 1.1)
    ax.set_xlabel("$u=1/r$  [1/M]")
    ax.set_ylabel("$|\\mathrm{d}u/\\mathrm{d}\\varphi|$")
    ax.set_title("零测地线的相平面（首次积分）")
    ax.legend(loc="upper right")
    S.grid(ax)
    return S.save(fig, "f04_phase_portrait")


# --------------------------------------------------------------------------- #
def f05_convergence():
    S.apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    colors = [C_ACCENT2, C_ACCENT, C_ACCENT3]
    for b, col in zip((6.0, 12.0, 26.0), colors):
        ns = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
        _, rows = G.convergence_table(b, steps=tuple(ns))
        err = np.array([r[1] for r in rows])
        ax.loglog(ns, err, "o-", ms=3.4, color=col, label="$b=%gM$" % b)
    ref_n = np.array([100.0, 3e4])
    ax.loglog(ref_n, 3e-5 * (ref_n / 100.0) ** -4, "--", color=C_DIM, lw=1.0,
              label="$\\mathcal{O}(h^4)$（四阶参考）")
    ax.set_xlabel("每条光线的 RK4 步数 $n$")
    ax.set_ylabel("方位角误差 $|\\Delta\\varphi|$ [rad]")
    ax.set_title("积分器的收敛性")
    ax.legend(loc="lower left")
    ax.set_ylim(1e-14, 1e-4)
    S.grid(ax)
    return S.save(fig, "f05_integrator_convergence")


# --------------------------------------------------------------------------- #
def f06_escape():
    S.apply_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    R = np.exp(np.linspace(np.log(30.0), np.log(3000.0), 60))
    for b, col in zip((6.0, 12.0, 20.0, 26.0), (C_ACCENT2, C_ACCENT, C_ACCENT3, C_ACCENT4)):
        err = np.array([G.tangent_error(float(r), b) for r in R])
        ax.loglog(R, np.degrees(err), color=col, label="$b=%gM$" % b)
    ax.axvline(140.0, color="#b5533f", lw=1.0)
    ax.text(150, 3e-6, "默认 $R_{\\rm esc}=140M$", color="#b5533f", fontsize=8)
    ax.axhline(np.degrees(0.066 / 60), color=C_DIM, ls="--", lw=0.8)
    ax.text(35, np.degrees(0.066 / 60) * 1.25, "0.001 px（0.066°/px）", color=C_DIM, fontsize=7.5)
    ax.set_xlabel("背景采样半径 $R_{\\rm esc}/M$")
    ax.set_ylabel("背景方向误差 [deg]")
    ax.set_title("逃逸半径近似带来的方向误差")
    ax.legend(loc="upper right")
    S.grid(ax)
    return S.save(fig, "f06_escape_radius")


# --------------------------------------------------------------------------- #
def f07_disk_profiles():
    S.apply_style()
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7))
    r = np.exp(np.linspace(np.log(6.001), np.log(60), 2000))
    f_ss = G.flux_newtonian(r)
    f_nt = G.flux_page_thorne(r)

    ax = axs[0]
    ax.plot(r, f_ss, color=C_ACCENT2, label="牛顿 SS 剖面")
    ax.plot(r, f_nt, color=C_ACCENT, label="相对论 NT 剖面")
    ax.plot([G.profile_peak_radius("ss")], [1.0], "v", color=C_ACCENT2, ms=5)
    ax.plot([G.profile_peak_radius("nt")], [1.0], "v", color=C_ACCENT, ms=5)
    ax.set_xlabel("$r/M$")
    ax.set_ylabel("$F/F_{\\max}$")
    ax.set_title("归一化通量剖面")
    ax.legend(loc="upper right")
    S.grid(ax)

    ax = axs[1]
    tmax = 9000.0
    ax.plot(r, G.temperature_from_flux(f_ss, tmax), color=C_ACCENT2, label="牛顿 SS")
    ax.plot(r, G.temperature_from_flux(f_nt, tmax), color=C_ACCENT, label="相对论 NT")
    ax.set_xlabel("$r/M$")
    ax.set_ylabel("$T$ [K]")
    ax.set_title("有效温度 $T\\propto F^{1/4}$")
    ax.legend(loc="upper right")
    S.grid(ax)

    ax = axs[2]
    ax.semilogx(r, G.nt_over_ss(r), color=C_ACCENT3)
    ax.axhline(1.0, color=C_DIM, ls="--", lw=0.8)
    ax.set_xlabel("$r/M$")
    ax.set_ylabel("$F_{\\rm NT}/F_{\\rm SS}$")
    ax.set_title("相对论修正（$r\\to\\infty$ 时 $\\to 1$）")
    ax.set_xlim(6, 60)
    S.grid(ax)
    return S.save(fig, "f07_disk_profiles")


# --------------------------------------------------------------------------- #
def f08_efficiency():
    S.apply_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    r = np.exp(np.linspace(np.log(6.0001), np.log(1.0e6), 40000))
    # Sum rule for the radiated power.  For the Newtonian SS73 profile the
    # energy received at infinity per unit coordinate area is the emitted
    # flux itself, so the weight is w = 1 and the integral is exactly
    # eta_SS = 1/12.  For the general-relativistic Page-Thorne profile the
    # published F(r) is the flux per unit *proper* area in the emitter frame;
    # converting it to the energy crossing a coordinate sphere per unit
    # coordinate time costs the product of the gravitational redshift
    # (1-2/r) and the proper/coordinate area ratio (1-2/r)^-1/2, i.e.
    # w = E(r) = (1-2M/r)/(1-3M/r)^1/2, the circular-orbit specific energy.
    # With that weight the sum rule closes on
    # eta = 1-E(6M) to 5e-11 (see paper/tools/audit_nt.py); the unweighted
    # coordinate-area integral overshoots it by 1.916%.
    for kind, col, lab in (("ss", C_ACCENT2, "牛顿 SS（$w=1$）"),
                           ("nt", C_ACCENT, "相对论 NT（$w=E(r)$）")):
        if kind == "nt":
            f = G.flux_page_thorne(r, 6.0, norm="none")
            w = G.specific_energy(r)
            target = G.isco_efficiency()
        else:
            f = 3.0 / (8 * np.pi) * (1 - np.sqrt(6 / r)) / r**3
            w = np.ones_like(r)
            target = 1.0 / 12.0
        dl = 2.0 * f * 2 * np.pi * r * w        # both faces, weighted area
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (dl[1:] + dl[:-1]) * np.diff(r))])
        ax.semilogx(r, cum, color=col, label=lab)
        ax2.loglog(r[1:], np.abs(cum[1:] - target) / target, color=col, label=lab)
    eta = G.isco_efficiency()
    ax.axhline(eta, color="#b5533f", ls="--", lw=1.0)
    ax.text(6.6, eta * 1.05, "相对论效率 $\\eta = 1-E(6M) = 5.72\\%$", color="#b5533f", fontsize=8)
    ax.axhline(1 / 12, color=C_ACCENT2, ls=":", lw=1.0)
    ax.text(6.6, 1 / 12 * 0.955, "牛顿极限 $M/2r_{\\rm in} = 8.33\\%$", color=C_ACCENT2, fontsize=8)
    ax.set_xlabel("$r/M$")
    ax.set_ylabel("累积辐射功率 [$\\dot M c^2$]")
    ax.set_title("(a) 累积辐射功率（求和规则）")
    ax.set_xlim(6, 1.0e6)
    ax.legend(loc="lower right")
    S.grid(ax)

    ax2.set_xlabel("$r/M$")
    ax2.set_ylabel("相对残差 $|P(r)-\\eta|/\\eta$")
    ax2.set_title("(b) 尾部残差：$O(1/r)$ 收敛")
    ax2.axhline(1.0e-2, color="#888", ls="-.", lw=0.8)
    rref = np.array([10.0, 1.0e5])
    ax2.plot(rref, 12.0 / rref, color="#888", ls="--", lw=0.8)
    ax2.text(3.0e4, 12.0 / 3.0e4 * 1.6, "$12/r$", color="#666", fontsize=7)
    ax2.legend(loc="lower left")
    S.grid(ax2)
    return S.save(fig, "f08_disk_efficiency")


# --------------------------------------------------------------------------- #
def f09_doppler_map():
    S.apply_style()
    r = np.linspace(6, 26, 420)
    psi = np.linspace(-np.pi, np.pi, 560)
    RR, PP = np.meshgrid(r, psi, indexing="ij")
    # photon angular momentum component along the spin axis, for a face-on view
    # |b_axis| <= |b| and reaches its maximum for the edge-on parts of the disk;
    # here we plot the equatorial-plane (b_axis = -r cos(psi) for a distant
    # observer) configuration used in the paper's illustrative figure.
    b_axis = -RR * np.cos(PP)
    g = G.doppler_factor(RR, b_axis, keplerian=True, r_obs=np.inf)

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    # 420 x 560 quads stored as vector polygons make a ~6 MB PDF for no visible
    # gain: rasterise this one artist at 400 dpi and keep the contours vector.
    im = ax.pcolormesh(np.degrees(psi), r, g, cmap="RdBu_r", shading="auto",
                       vmin=2 - np.nanpercentile(g, 99.5), vmax=np.nanpercentile(g, 99.5),
                       rasterized=True)
    cs = ax.contour(np.degrees(psi), r, g, levels=[0.7, 0.85, 1.0, 1.2, 1.4],
                    colors="k", linewidths=0.5)
    ax.clabel(cs, fmt="%.2f", fontsize=7)
    ax.set_xlabel("盘方位角 $\\psi$ [deg]")
    ax.set_ylabel("$r/M$")
    ax.set_title("频移因子 $g(r,\\psi)$（开普勒盘）")
    fig.colorbar(im, ax=ax, label="$g=\\nu_{\\rm obs}/\\nu_{\\rm emit}$", pad=0.02)

    ax = axs[1]
    for rr, col in zip((6.0, 8.0, 12.0, 20.0), (C_ACCENT, C_ACCENT2, C_ACCENT3, C_ACCENT4)):
        m = np.isclose(RR[:, 0], rr, atol=0.02)
        if not m.any():
            continue
        i = int(np.argmin(np.abs(r - rr)))
        gg = G.doppler_factor(rr, -rr * np.cos(psi), keplerian=True)
        ax.plot(np.degrees(psi), gg**4, color=col, label="$r=%gM$" % rr)
    ax.set_xlabel("盘方位角 $\\psi$ [deg]")
    ax.set_ylabel("集束因子 $g^4$")
    ax.set_title("多普勒集束（$I\\propto g^4$）")
    ax.legend(loc="upper right")
    ax.set_xlim(-180, 180)
    S.grid(ax)
    return S.save(fig, "f09_doppler_map", pdf_dpi=400)


# --------------------------------------------------------------------------- #
def f10_spectrum():
    S.apply_style()
    lam = np.linspace(120, 1400, 900)
    t0 = 9000.0
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    ax = axs[0]
    for g, col in zip((0.55, 0.8, 1.0, 1.25, 1.6),
                      (C_ACCENT2, C_ACCENT3, C_DIM, C_ACCENT, C_ACCENT4)):
        spec = G.planck_spectrum(g * t0, lam)
        ax.plot(lam, spec / spec.max(), color=col, label="$g=%.2f$" % g)
    ax.set_xlabel("波长 $\\lambda$ [nm]")
    ax.set_ylabel("$B_\\lambda(\\lambda; gT)/B_{\\max}$")
    ax.set_title("多普勒/引力移动后的黑体谱")
    ax.legend(loc="upper right", ncol=2)
    S.grid(ax)

    ax = axs[1]
    colors = G.planck_rgb(np.array([3000.0, 5000.0, 7000.0, 9000.0, 14000.0, 25000.0]))
    x, y = G.planck_xy(np.linspace(1600, 30000, 400))
    ax.plot(x, y, color="#555", lw=1.0)
    rgb = np.clip(colors / np.maximum(colors.max(axis=1, keepdims=True), 1e-9), 0, 1)
    ax.scatter(x[::40], y[::40], s=6, c=rgb[::10][:len(x[::40])], marker="o")
    for t, col in zip((3000, 9000, 25000), (C_ACCENT, C_DIM, C_ACCENT2)):
        xx, yy = G.planck_xy(np.array([float(t)]))
        ax.plot(xx, yy, "o", ms=6, color=col, mec="k", mew=0.4)
        ax.annotate("%d K" % t, (xx[0], yy[0]), textcoords="offset points",
                    xytext=(6, 5), fontsize=8, color=col)
    ax.set_xlabel("CIE $x$")
    ax.set_ylabel("CIE $y$")
    ax.set_title("普朗克轨迹与盘色温范围")
    ax.set_xlim(0.2, 0.75)
    ax.set_ylim(0.2, 0.5)
    S.grid(ax, alpha=0.2)
    return S.save(fig, "f10_spectrum_colour")


# --------------------------------------------------------------------------- #
def f11_transfer():
    S.apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axs[0]
    tau = np.linspace(0, 6, 400)
    for faces, col, lab in ((1, C_ACCENT2, "单次穿越"), (2, C_ACCENT, "两次穿越（近侧 + 透镜远侧）")):
        ax.plot(tau, faces * (1 - np.exp(-tau)), color=col, label=lab)
    ax.plot(tau, np.minimum(tau, 1.0), ":", color=C_DIM, label="光学薄极限 $\\tau$")
    ax.set_xlabel("面光学厚度 $\\tau$")
    ax.set_ylabel("出射强度 / 源函数")
    ax.set_title("前向累积辐射转移")
    ax.legend(loc="lower right")
    S.grid(ax)

    ax = axs[1]
    y = np.linspace(-3, 3, 400)
    for hr, col in zip((0.02, 0.05, 0.10, 0.20), (C_ACCENT2, C_ACCENT3, C_ACCENT, C_ACCENT4)):
        ax.plot(np.exp(-0.5 * (y / hr) ** 2), y, color=col, label="$H/r=%.2f$" % hr)
    ax.set_xlabel("$\\exp(-y^2/2H^2)$")
    ax.set_ylabel("$y/r$")
    ax.set_title("垂直高斯结构")
    ax.legend(loc="upper right", fontsize=7.5)
    S.grid(ax)
    return S.save(fig, "f11_radiative_transfer")


# --------------------------------------------------------------------------- #
def f12_image_orders():
    """Which image order does each pixel see?  Count equatorial crossings."""
    S.apply_style(dark=True)
    r0 = 26.0
    fov = np.radians(58.0)
    n = 420
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    order = np.full((n, n), np.nan)
    b_grid = np.full((n, n), np.nan)
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            psi = np.arctan(np.hypot(xx, yy) * np.tan(fov / 2))
            b = G.impact_parameter_from_local_angle(psi, r0)
            b_grid[j, i] = b
            if b <= G.B_CRIT:
                order[j, i] = 0                     # shadow
                continue
            # count how many times the ray crosses the equatorial plane
            u1 = G.turning_point(b)
            if u1 is None:
                order[j, i] = 0
                continue
            phi_half = float(np.atleast_1d(G.phi_to_periapsis(b, np.array([1.0 / r0])))[0])
            sweeps = 2 * phi_half / np.pi
            order[j, i] = 1 if sweeps < 1.5 else (2 if sweeps < 2.6 else 3)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    cmap = plt.get_cmap("inferno", 4)
    im = ax.imshow(order, origin="lower", extent=[-1, 1, -1, 1], cmap=cmap,
                   interpolation="nearest", alpha=0.95)
    ax.add_patch(plt.Circle((0, 0), np.tan(G.shadow_angle(r0)) / np.tan(fov / 2),
                            fill=False, ec="#7fd4ff", lw=1.0, ls="--"))
    cb = fig.colorbar(im, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625], pad=0.02)
    cb.ax.set_yticklabels(["阴影", "直接像", "二次像", "高阶像"])
    ax.set_xlabel("$x$ [视场]")
    ax.set_ylabel("$y$ [视场]")
    ax.set_title("视线与赤道面交会的像阶（$r_0=26M$）")
    return S.save(fig, "f12_image_orders")


# --------------------------------------------------------------------------- #
def f13_lensing_map():
    """Grid of source directions warped by the lens (static observer)."""
    S.apply_style(dark=True)
    r0 = 26.0
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
    half = np.radians(12.0)
    for ax, title, dmax in ((axs[0], "未透镜化", 0.0), (axs[1], "引力透镜化", 1.0)):
        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)
        ax.set_aspect("equal")
        ax.set_title(title)
    # radial grid of source angles
    for rad in np.linspace(2.0, 12.0, 6):
        for az in np.linspace(0, 2 * np.pi, 145):
            sx, sy = rad * np.cos(az), rad * np.sin(az)
            axs[0].plot(sx, sy, ".", ms=1.0, color="#7f8ea6")
            psi = np.arctan(rad / r0)
            b = G.impact_parameter_from_local_angle(psi, r0)
            if b <= G.B_CRIT:
                continue
            t_in, _ = G.tangent_xy(b, r0)
            t_asym, _ = G.tangent_xy(b, 1e7)
            d = np.arccos(np.clip(np.dot(t_in, t_asym), -1, 1)) * 180 / np.pi
            psi_obs = np.degrees(np.arctan(rad / r0)) - d
            ro = r0 * np.tan(np.radians(psi_obs))
            bx, by = ro * np.cos(az), ro * np.sin(az)
            col = "#ff9b52" if rad < 7 else "#66c2ff"
            axs[1].plot(bx, by, ".", ms=1.0, color=col)
    t = np.linspace(0, 2 * np.pi, 400)
    for ax in axs:
        rs = np.degrees(G.shadow_angle(r0))
        rr = r0 * np.tan(np.radians(rs))
        if ax is axs[1]:
            ax.fill(rr * np.cos(t), rr * np.sin(t), color="#000", zorder=5)
            ax.plot(rr * np.cos(t), rr * np.sin(t), color="#e0574a", lw=1.0, zorder=6)
        ax.set_xlabel("$x$ [M]")
    axs[0].set_ylabel("$y$ [M]")
    return S.save(fig, "f13_lensing_map")


# --------------------------------------------------------------------------- #
def main() -> int:
    only = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")]
    if not only and "--only" in sys.argv:
        i = sys.argv.index("--only")
        only = [a for a in sys.argv[i + 1:] if not a.startswith("-")]
    made = []
    for fn in (f01_geodesics, f02_deflection, f03_shadow, f04_phase, f05_convergence,
               f06_escape, f07_disk_profiles, f08_efficiency, f09_doppler_map,
               f10_spectrum, f11_transfer, f12_image_orders, f13_lensing_map):
        if only and fn.__name__.split("_")[0] not in only:
            continue
        try:
            made.append(fn())
            print("[ok]", os.path.basename(made[-1]))
        except Exception as exc:                       # keep going, report clearly
            print("[!!]", fn.__name__, "->", type(exc).__name__, exc)
    print("figures written:", len(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
