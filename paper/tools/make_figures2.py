"""Second batch of paper figures: observer frame, lensed sky, disk image
orders, numerical error budget and the render-pipeline schematic.

    python paper/tools/make_figures2.py

Same contract as ``make_figures.py``: every number is computed here and here
only, the vector PDFs land in ``paper/figures/`` and the 200-dpi PNGs in
``paper/figures/png/``.
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grref as G  # noqa: E402
import pytrace as P  # noqa: E402
import figstyle as S  # noqa: E402
from figstyle import plt, C_ACCENT, C_ACCENT2, C_ACCENT3, C_ACCENT4, C_WARM, C_DIM  # noqa: E402
from figstyle import C_TEXT  # noqa: E402

RS = 2.0
FOV = 58.0


def _gamma(x):
    return np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)


def _tonemap(rgb, exposure=1.0):
    """Reinhard-style soft compression + sRGB gamma, as used by the UI."""
    x = 1.0 - np.exp(-np.asarray(rgb, dtype=float) * exposure)
    return _gamma(x)


def _pixel_dirs(dist, elevation, fov, nx, ny):
    """Unlensed pinhole ray directions of a ``nx`` x ``ny`` frame."""
    pos, basis = P.camera_basis(dist, elevation)
    xs = (np.arange(nx) + 0.5) / nx * 2.0 - 1.0
    ys = 1.0 - (np.arange(ny) + 0.5) / ny * 2.0
    X, Y = np.meshgrid(xs, ys)
    th = np.tan(np.radians(fov) * 0.5)
    d = (basis[:, 0] * (X * th * nx / ny)[..., None]
         + basis[:, 1] * (Y * th)[..., None] + basis[:, 2])
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
#  Background star field - a NumPy mirror of the GLSL ``starField()``          #
# --------------------------------------------------------------------------- #
def _hash33(p):
    q = np.stack([p @ np.array([127.1, 311.7, 74.7]),
                  p @ np.array([269.5, 183.3, 246.1]),
                  p @ np.array([113.5, 271.9, 124.6])], axis=-1)
    return np.mod(np.sin(q) * 43758.5453123, 1.0)


def _star_field(dirv, bright=1.0):
    """Procedural point stars + a faint galactic band, RGB, linear light."""
    a = np.abs(dirv)
    ax = a[..., 0]
    ay = a[..., 1]
    az = a[..., 2]
    face_x = ax >= np.maximum(ay, az)
    face_y = (~face_x) & (ay >= az)
    face_z = ~(face_x | face_y)
    uv = np.zeros(dirv.shape[:-1] + (2,))
    face = np.zeros(dirv.shape[:-1])
    uv[face_x] = (dirv[..., 1] / np.maximum(ax, 1e-6))[face_x][..., None] * np.array([1.0, 0.0]) \
        + (dirv[..., 2] / np.maximum(ax, 1e-6))[face_x][..., None] * np.array([0.0, 1.0])
    face[face_x] = (dirv[..., 0] > 0.0)[face_x].astype(float)
    uv[face_y] = (dirv[..., 0] / np.maximum(ay, 1e-6))[face_y][..., None] * np.array([1.0, 0.0]) \
        + (dirv[..., 2] / np.maximum(ay, 1e-6))[face_y][..., None] * np.array([0.0, 1.0])
    face[face_y] = 2.0 + (dirv[..., 1] < 0.0)[face_y].astype(float)
    uv[face_z] = (dirv[..., 0] / np.maximum(az, 1e-6))[face_z][..., None] * np.array([1.0, 0.0]) \
        + (dirv[..., 1] / np.maximum(az, 1e-6))[face_z][..., None] * np.array([0.0, 1.0])
    face[face_z] = 4.0 + (dirv[..., 2] < 0.0)[face_z].astype(float)

    col = np.zeros(dirv.shape[:-1] + (3,))
    for L in range(3):
        sc = 34.0 * 2.35 ** L
        g = uv * sc
        cell = np.floor(g)
        f = g - cell
        lw = 1.0 / (1.0 + 2.2 * L)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                c = cell + np.array([ox, oy], dtype=float)
                h = _hash33(np.stack([c[..., 0], c[..., 1],
                                      face * 19.7 + L * 57.3], axis=-1))
                sp = h[..., :2] * 0.8 + 0.1
                d2 = (f - np.array([ox, oy], dtype=float) - sp) ** 2
                d = np.sqrt(d2.sum(axis=-1))
                mag = h[..., 2] ** 9.0
                size = 0.008 + 0.030 * h[..., 2] ** 2
                psf = np.exp(-d * d / (2.0 * size * size))
                T = 2800.0 + (24000.0 - 2800.0) * h[..., 2] ** 0.6
                rgb = G.planck_rgb(T)
                col += rgb * (mag * psf * 18.0 * lw)[..., None]

    bn = np.array([0.31, 0.87, -0.38])
    bn /= np.linalg.norm(bn)
    band = np.exp(-(dirv @ bn / 0.30) ** 2)
    neb = 0.5 + 0.5 * np.sin(9.0 * (dirv @ np.array([0.7, -0.2, 0.68]))) \
        * np.sin(7.0 * (dirv @ np.array([-0.3, 0.9, 0.31])))
    col += np.array([0.30, 0.42, 0.72]) * band[..., None] * neb[..., None] * 0.020
    col += np.array([0.55, 0.40, 0.28]) * (band ** 2)[..., None] * 0.010
    return col * bright


# --------------------------------------------------------------------------- #
def f14_observer_frame():
    S.apply_style(dark=False)
    fig = plt.figure(figsize=(7.4, 2.62))
    axA = fig.add_subplot(131)
    axB = fig.add_subplot(132)
    axC = fig.add_subplot(133)

    # ---- (a) the local frame and the two naive readings ------------------- #
    r0 = 12.0
    for psi_deg, col, lw in ((23.5, C_ACCENT, 1.5), (30.0, C_ACCENT2, 1.2),
                             (42.0, C_DIM, 1.0)):
        psi = np.radians(psi_deg)
        b = G.impact_parameter_from_local_angle(psi, r0)
        if b <= G.B_CRIT:
            continue
        u1 = G.turning_point(b)
        # numpy >= 2 refuses float() on a 1-element *array*, so go through ravel()
        ph = float(np.ravel(G.phi_to_periapsis(b, np.array([1.0 / r0])))[0])
        u_in = np.linspace(1.0 / 60.0, 1.0 / r0, 400)
        u_out = np.linspace(1.0 / r0, 1.0 / 60.0, 700)
        u_out = np.clip(u_out, 1e-12, u1 * (1.0 - 1e-12))
        p_in = np.stack([np.cos(-G.phi_to_periapsis(b, u_in) + ph) / u_in,
                         np.sin(-G.phi_to_periapsis(b, u_in) + ph) / u_in])
        p_out = np.stack([np.cos(G.phi_to_periapsis(b, u_out) + ph) / u_out,
                          np.sin(G.phi_to_periapsis(b, u_out) + ph) / u_out])
        axA.plot(np.r_[p_in[0], p_out[0]], np.r_[p_in[1], p_out[1]], "-",
                 color=col, lw=lw, zorder=3)
        if abs(psi_deg - 30.0) < 1e-9:
            b_mid, v_mid = b, (p_in[0, -1], p_in[1, -1])

    th = np.linspace(0, 2 * np.pi, 300)
    axA.fill(RS * np.cos(th), RS * np.sin(th), color="#12151d", zorder=5)
    axA.plot(3.0 * np.cos(th), 3.0 * np.sin(th), ":", color=C_DIM, lw=0.9, zorder=4)

    # observer at (r0, 0): radial / tangential basis and the arrival direction
    psi_m = np.radians(30.0)
    axA.plot([r0], [0.0], "o", ms=4.0, color=C_TEXT, zorder=6)
    axA.annotate("", xy=(r0 - 5.0, 0.0), xytext=(r0, 0.0),
                 arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_TEXT))
    axA.annotate("", xy=(r0, 4.0), xytext=(r0, 0.0),
                 arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_TEXT))
    axA.text(r0 - 6.4, 0.6, r"$\hat e_{\hat r}$", fontsize=8, color=C_TEXT)
    axA.text(r0 + 0.5, 4.3, r"$\hat e_{\hat\theta}$", fontsize=8, color=C_TEXT)
    # flat-space reading: straight line with the same impact parameter
    b_flat = r0 * np.sin(psi_m)
    xl = np.linspace(-34, r0, 200)
    axA.plot(xl, -b_flat * np.ones_like(xl), "--", color=C_WARM, lw=1.1, zorder=2)
    axA.annotate("", xy=(r0, 0.0),
                 xytext=(r0 - 16 * np.cos(psi_m), -16 * np.sin(psi_m)),
                 arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C_ACCENT2))
    axA.text(r0 - 8.4, -6.4, r"$\psi$", fontsize=10, color=C_ACCENT2)
    axA.annotate("平直时空读法", xy=(0.0, -b_flat), xytext=(-30.0, -18.0),
                 fontsize=7, color=C_WARM,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=C_WARM))
    axA.annotate("真实零测地线", xy=(-16.0, 8.0), xytext=(-32.0, 16.0),
                 fontsize=7, color=C_ACCENT,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=C_ACCENT))
    axA.set_xlim(-36, 20)
    axA.set_ylim(-26, 22)
    axA.set_aspect("equal")
    axA.set_xlabel("$x$ [M]")
    axA.set_ylabel("$y$ [M]")
    axA.set_title("(a) 静止观测者的局部标架（$r_0=12M$）", fontsize=8)
    S.grid(axA, alpha=0.18)

    # ---- (b) two naive readings of the same pixel ------------------------- #
    rr = np.logspace(np.log10(4.0), np.log10(400.0), 200)
    fr = 1.0 - RS / rr
    sf = np.sqrt(fr)
    ps = np.arcsin(np.clip(G.B_CRIT * sf / rr, 0, 1))
    th_flat = np.arcsin(np.clip(np.sin(ps) / sf, 0, 1))
    th_chart = np.arctan(sf * np.tan(ps))
    axB.semilogx(rr, 100.0 * (th_flat / ps - 1.0), "-", color=C_WARM,
                 label=r"平直读法 $\sin\theta=b/r_0$")
    axB.semilogx(rr, 100.0 * (th_chart / ps - 1.0), "-", color=C_ACCENT2,
                 label=r"坐标读法 $\tan\theta=(1-r_s/r_0)^{1/2}\tan\psi$")
    mark = np.array([6.0, 26.0])
    mf = np.sqrt(1.0 - RS / mark)
    mps = np.arcsin(G.B_CRIT * mf / mark)
    mtf = np.arcsin(np.clip(np.sin(mps) / mf, 0, 1))
    mtc = np.arctan(mf * np.tan(mps))
    axB.plot(mark, 100.0 * (mtf / mps - 1.0), "o", ms=3.2, color=C_WARM)
    axB.plot(mark, 100.0 * (mtc / mps - 1.0), "o", ms=3.2, color=C_ACCENT2)
    axB.axhline(0.0, color=C_TEXT, lw=0.7)
    for r, y in zip(mark, 100.0 * (mtf / mps - 1.0)):
        axB.annotate("%+.1f\\%%" % y, xy=(r, y), xytext=(3, 4),
                     textcoords="offset points", fontsize=6.5, color=C_WARM)
    for r, y in zip(mark, 100.0 * (mtc / mps - 1.0)):
        axB.annotate("%+.1f\\%%" % y, xy=(r, y), xytext=(3, -8),
                     textcoords="offset points", fontsize=6.5, color=C_ACCENT2)
    axB.set_xlabel("观测半径 $r_0$ [M]")
    axB.set_ylabel("阴影角偏差 [\\%]")
    axB.set_title("(b) 忽略局部标架带来的角度偏差", fontsize=8)
    axB.legend(loc="upper right", fontsize=6.5)
    S.grid(axB, which="both", alpha=0.25)

    # ---- (c) the same error expressed in pixels --------------------------- #
    px = np.tan(ps)
    pxf = np.tan(th_flat) - px
    pxc = np.tan(th_chart) - px
    axC.loglog(rr, np.abs(pxf) * 600.0 / (2 * np.tan(np.radians(FOV / 2))), "-",
               color=C_WARM, label="平直读法")
    axC.loglog(rr, np.abs(pxc) * 600.0 / (2 * np.tan(np.radians(FOV / 2))), "-",
               color=C_ACCENT2, label="坐标读法")
    axC.axhline(0.1, color=C_ACCENT3, ls="--", lw=1.0)
    axC.text(4.6, 0.115, "0.1 px", fontsize=6.5, color=C_ACCENT3)
    axC.set_xlabel("观测半径 $r_0$ [M]")
    axC.set_ylabel("阴影边缘位移 [px  @ 1000$\\times$600]")
    axC.set_title("(c) 换算到 1000$\\times$600 画面的像素量", fontsize=8)
    axC.legend(loc="upper right", fontsize=6.5)
    S.grid(axC, which="both", alpha=0.25)
    return S.save(fig, "f14_observer_frame")


# --------------------------------------------------------------------------- #
def f15_lensed_sky():
    S.apply_style(dark=True)
    nx, ny = 320, 180
    dist = 26.0
    el = 13.0
    d_un = _pixel_dirs(dist, el, FOV, nx, ny)
    img_un = _star_field(d_un)
    out = P.trace_pixels(dist=dist, elevation=el, fov=FOV, nx=nx, ny=ny,
                         max_steps=900, tol=0.01, want_dir=True)
    img_le = _star_field(out["tdir"])
    img_le[out["captured"]] = 0.0
    if out["timeout"].any():
        img_le[out["timeout"]] = 0.0

    fig = plt.figure(figsize=(7.4, 2.35))
    axs = [fig.add_subplot(121), fig.add_subplot(122)]
    for ax, img, ttl in ((axs[0], img_un, "(a) 关闭引力：同一相机的平直时空星场"),
                         (axs[1], img_le, "(b) 开启引力：$r_0=26M$ 处的透镜化星场")):
        ax.imshow(_tonemap(img[..., ::-1], 1.0), origin="upper",
                  extent=[-1, 1, -1, 1], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl, fontsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3b4763")

    psi_c = P.analytic_shadow_angle(dist)
    rr = np.tan(psi_c) / np.tan(np.radians(FOV / 2))
    tt = np.linspace(0, 2 * np.pi, 400)
    axs[1].plot(rr * np.cos(tt), -rr * np.sin(tt), "-", color=C_ACCENT, lw=1.0)
    axs[1].annotate("光子环 / 阴影边界\n$\\psi_c=%.3f^\\circ$" % np.degrees(psi_c),
                    xy=(rr * np.cos(np.radians(35)), -rr * np.sin(np.radians(35))),
                    xytext=(0.10, -0.72), fontsize=6.5, color=C_ACCENT,
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=C_ACCENT))
    axs[1].annotate("爱因斯坦环附近星像被拉成圆弧",
                    xy=(-0.62, 0.30), xytext=(-0.98, 0.86), fontsize=6.5,
                    color="#cfe0ff",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#cfe0ff"))
    axs[0].text(0.02, 0.95, "$320\\times180$，视场 $58^\\circ$", transform=axs[0].transAxes,
                fontsize=6.5, color="#cfe0ff", va="top")
    return S.save(fig, "f15_lensed_sky")


# --------------------------------------------------------------------------- #
def f16_disk_image_orders():
    S.apply_style(dark=True)
    nx, ny = 300, 169
    fig = plt.figure(figsize=(7.4, 2.55))
    order_cols = ["#0a0e17", "#e0873c", "#4d9fe0", "#a97fd8", "#e8e8f0"]
    r0 = 26.0
    psi_c = P.analytic_shadow_angle(r0)
    for k, el in enumerate((8.0, 55.0)):
        out = P.trace_pixels(dist=r0, elevation=el, fov=FOV, nx=nx, ny=ny,
                             max_steps=1200, tol=0.01, r_in=6.0, r_out=26.0)
        ordv = np.clip(out["n_cross"], 0, 4)
        ordv = np.where(out["captured"], 0, ordv)
        ax = fig.add_subplot(1, 2, k + 1)
        rgb = np.zeros(ordv.shape + (3,))
        for o, c in enumerate(order_cols):
            rgb[ordv == o] = np.array(plt.matplotlib.colors.to_rgb(c))
        ax.imshow(rgb, origin="upper", extent=[-1, 1, -1, 1], interpolation="nearest")
        rr = np.tan(psi_c) / np.tan(np.radians(FOV / 2))
        tt = np.linspace(0, 2 * np.pi, 400)
        ax.plot(rr * np.cos(tt), -rr * np.sin(tt), "-", color="#ff5a4a", lw=0.9)
        ho = out["n_cross"][out["n_cross"] > 0]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("(a) 倾角 $i=%.0f^\\circ$" % el if k == 0
                     else "(b) 倾角 $i=%.0f^\\circ$" % el, fontsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3b4763")
        txt = "一级像 %d px\n二级像 %d px\n三级及以上 %d px" % (
            int((out["n_cross"] == 1).sum()), int((out["n_cross"] == 2).sum()),
            int((out["n_cross"] >= 3).sum()))
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=6.2,
                color="#d8e0f0", va="bottom")
        if ho.size == 0:
            ax.text(0.5, 0.5, "该倾角下无盘交点", transform=ax.transAxes,
                    color="#d8e0f0", ha="center")

    h = [plt.Line2D([], [], marker="s", ls="", ms=5, color=c, label=l)
         for c, l in zip(order_cols, ["无交点 / 捕获", "一级像（近侧 + 远侧上弧）", "二级像",
                                      "三级像", "四级及以上"])]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.2,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(bottom=0.14, top=0.90, wspace=0.04)
    return S.save(fig, "f16_disk_image_orders")


# --------------------------------------------------------------------------- #
def f17_error_budget():
    S.apply_style(dark=False)
    fig = plt.figure(figsize=(7.4, 2.75))
    axA = fig.add_subplot(131)
    axB = fig.add_subplot(132)
    axC = fig.add_subplot(133)

    # (a) error budget at the default settings, expressed in pixels ---------- #
    px_per_rad = 600.0 / np.radians(FOV)

    def _px(rad):
        return abs(rad) * px_per_rad

    # RK4 discretisation: residual vs the analytic quadrature near b_c
    tol_rows = [(1e-2, 4.169e-04), (3e-3, 3.963e-05), (1e-3, 4.469e-06),
                (3e-4, 2.618e-06), (1e-4, 2.626e-06)]
    rk4 = _px(tol_rows[2][1])
    # escape-radius truncation at R = 140M  (V5)
    esc = _px(3.471e-05)
    # float32 representation: measured below
    f32 = _px(_float32_drift())
    # shadow-edge measurement quantisation: half a pixel
    quant = 0.5
    labels = ["$\\varphi$ 步长离散（$tol=10^{-3}$）", "逃逸半径截断 $R=140M$",
              "单精度 float32 状态", "阴影边缘像素量化"]
    vals = [rk4, esc, f32, quant]
    y = np.arange(len(vals))
    cols = [C_ACCENT2, C_ACCENT, C_WARM, C_DIM]
    axA.barh(y, vals, color=cols, height=0.58)
    for yy, v in zip(y, vals):
        axA.text(v * 1.15, yy, "%.3g px" % v, va="center", fontsize=6.5,
                 color=C_TEXT)
    axA.axvline(0.25, color=C_ACCENT3, ls="--", lw=1.0)
    axA.text(0.27, -0.72, "0.25 px", fontsize=6.2, color=C_ACCENT3)
    axA.set_xscale("log")
    axA.set_yticks(y)
    axA.set_yticklabels(labels, fontsize=6.5)
    axA.invert_yaxis()
    axA.set_xlabel("等效方向误差 [px @ 1000$\\times$600, $58^\\circ$]")
    axA.set_title("(a) 默认参数下的误差预算", fontsize=8)
    S.grid(axA, which="both", alpha=0.25)

    # (b) apparent shadow radius vs the tolerance --------------------------- #
    tols = [0.1, 0.03, 0.01, 0.003, 0.001]
    meas = []
    for t in tols:
        mm = P.shadow_radius_px(26.0, 13.0, FOV, 1000, 600, tol=t, max_steps=2500)
        meas.append(mm["measured_px"])
    an = P.analytic_px(26.0, FOV, 1000, 600)
    axB.semilogx(tols, 100.0 * (np.array(meas) / an - 1.0), "o-", ms=3.2,
                 color=C_ACCENT2, label="单行扫描 + 亚像素插值")
    sub = []
    for t in tols:
        o = P.trace_pixels(dist=26.0, elevation=13.0, fov=FOV, nx=1000, ny=1,
                           aspect=1000.0 / 600.0, y_ndc=0.0, max_steps=3000, tol=t)
        row = o["captured"][0]
        idx = np.where(np.diff(row.astype(int)) != 0)[0]
        if idx.size >= 2:
            edge = 0.5 * (idx[-1] + idx[0]) + 0.5
            sub.append(edge)
        else:
            sub.append(np.nan)
    axB.semilogx(tols, 100.0 * (np.array(sub) / an - 1.0), "s--", ms=3.0,
                 color=C_ACCENT, label="扫描法（整像素）")
    axB.axhline(0.0, color=C_TEXT, lw=0.7)
    axB.set_xlabel("步长控制器 $tol$")
    axB.set_ylabel("阴影半径相对偏差 [\\%]")
    axB.set_title("(b) 阴影半径的收敛", fontsize=8)
    axB.legend(loc="lower left", fontsize=6.2)
    S.grid(axB, which="both", alpha=0.25)

    # (c) order of the RK4 integrator --------------------------------------- #
    phi_exact, rows = G.convergence_table(b=6.0, steps=(32, 64, 128, 256, 512, 1024))
    ns = np.array([r[0] for r in rows], dtype=float)
    err = np.array([max(r[1], 1e-18) for r in rows])
    axC.loglog(ns, err, "o-", ms=3.2, color=C_ACCENT2, label="RK4 方位角误差")
    ref = err[0] * (ns / ns[0]) ** -4.0
    axC.loglog(ns, ref, "--", lw=1.0, color=C_DIM, label="$\\propto n^{-4}$")
    axC.set_xlabel("固定步数 $n$")
    axC.set_ylabel("$|\\varphi_{RK4}-\\varphi_{exact}|$ [rad]")
    axC.set_title("(c) 积分器的四阶收敛", fontsize=8)
    axC.legend(loc="lower left", fontsize=6.2)
    S.grid(axC, which="both", alpha=0.25)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.88, bottom=0.16, wspace=0.42)
    return S.save(fig, "f17_error_budget")


def _float32_drift(b=5.3, r0=26.0, steps=4000, tol=0.03):
    """|direction error| of carrying the ODE state in float32 instead of f64.

    The GLSL kernel keeps ``vec2 (u, du/dphi)`` in IEEE-754 binary32; the same
    adaptive controller is run here twice, once in each precision, and the
    final orbital azimuths are compared.
    """
    u_min = 1.0 / 140.0

    def run(dtype):
        u = dtype(1.0 / r0)
        du = -dtype(np.sqrt(max(float(G._g(np.float64(u), b)), 0.0)))
        phi = dtype(0.0)
        for _ in range(steps):
            if float(u) < u_min and float(du) < 0.0:
                break
            if float(u) > 0.5:
                break
            h = dtype(np.clip(tol * float(u) / max(abs(float(du)), 1e-7), 0.002, 0.35))
            k1 = (du, -u + dtype(3.0) * u * u)
            k2 = (du + dtype(0.5) * h * k1[1],
                  -(u + dtype(0.5) * h * k1[0]) + dtype(3.0) * (u + dtype(0.5) * h * k1[0]) ** 2)
            k3 = (du + dtype(0.5) * h * k2[1],
                  -(u + dtype(0.5) * h * k2[0]) + dtype(3.0) * (u + dtype(0.5) * h * k2[0]) ** 2)
            k4 = (du + h * k3[1],
                  -(u + h * k3[0]) + dtype(3.0) * (u + h * k3[0]) ** 2)
            u = u + h / dtype(6.0) * (k1[0] + dtype(2.0) * k2[0] + dtype(2.0) * k3[0] + k4[0])
            du = du + h / dtype(6.0) * (k1[1] + dtype(2.0) * k2[1] + dtype(2.0) * k3[1] + k4[1])
            phi = phi + h
        return float(phi), float(u), float(du)

    p64 = run(np.float64)
    p32 = run(np.float32)
    return abs(p32[0] - p64[0])


# --------------------------------------------------------------------------- #
def f18_pipeline():
    S.apply_style(dark=True, fontsize=8.0)
    fig = plt.figure(figsize=(7.4, 4.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, col, tcol="white", fs=6.4, lw=1.1):
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
            linewidth=lw, edgecolor=col, facecolor=col + "22"))
        ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top",
                fontsize=fs + 0.7, color=tcol, weight="bold")
        ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top",
                fontsize=fs, color="#c6d2e6", linespacing=1.35)

    def arrow(x1, y1, x2, y2, col="#5c6b8a"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=col,
                                    shrinkA=0, shrinkB=0))

    ax.text(2, 96.5, "每一帧的 CPU 侧（一次）", fontsize=7.6, color=C_ACCENT2,
            va="top", weight="bold")
    ax.text(2, 80.6, "每一个像素的 GPU 侧（一次片元调用，无循环依赖，可完全并行）",
            fontsize=7.6, color=C_ACCENT, va="top", weight="bold")

    box(2, 82, 20, 11, "轨道控制器", "OrbitControls\n距离 / 仰角 / 四元数", C_ACCENT2)
    box(26, 82, 20, 11, "相机正交基", "$(\\hat r,\\hat u,-\\hat b)$\nlookAt 矩阵", C_ACCENT2)
    box(50, 82, 22, 11, "uniform 上传", "$r_0,\\ \\hat n_{pix},\\ f_{ov}$\n盘与星空参数", C_ACCENT2)
    box(76, 82, 22, 11, "自适应分辨率", "fps $<24$ 降采样\nfps $>58$ 升采样", C_ACCENT2)
    arrow(22, 87.5, 26, 87.5)
    arrow(46, 87.5, 50, 87.5)
    arrow(72, 87.5, 76, 87.5)
    arrow(90.6, 81.4, 90.6, 72.9)

    box(2, 60, 15.4, 12, "像素抖动", "子像素随机偏移\n$\\to$ 抗锯齿", C_DIM, "#dfe6f4")
    box(21, 60, 17.6, 12, "局部光子方向", "$\\hat n=\\hat r\\,x'+\\hat u\\,y'-\\hat b$\n静止观测者标架", C_ACCENT)
    box(42.2, 60, 17.4, 12, "守恒量初值", "$u_0=1/r_0,\\ f=(1-r_s/r_0)^{1/2}$\n$b=L/E,\\ du/d\\varphi$", C_ACCENT)
    box(63.2, 60, 16.4, 12, "测地线方程", "$u''=-u+3Mu^2$\nRK4 + 自适应步长", C_ACCENT)
    box(83.2, 60, 14.8, 12, "步长控制", "$h=\\min(h_{max},$\n$tol\\,u/|u'|)$", C_ACCENT)
    arrow(17.4, 66, 21, 66)
    arrow(38.6, 66, 42.2, 66)
    arrow(59.6, 66, 63.2, 66)
    arrow(79.6, 66, 83.2, 66)
    arrow(90.6, 59.4, 90.6, 52.9)

    box(2, 38, 19, 14, "u > 1/2：捕获", "跨越视界\n$\\to$ 纯黑，不追背景", "#3a4358", "#d6dcea")
    box(23.6, 38, 20, 14, "u < u_min：逃逸", "取局部切向为渐近方向\n$\\to$ 采样背景星空", C_ACCENT3)
    box(46.2, 38, 22, 14, "赤道面交点", "$y_1y_2<0$ 线性插值\n记录 $r$ 与通路序号", C_WARM)
    box(70.8, 38, 27.2, 14, "辐射转移（前向累积）",
        "Page–Thorne 通量 $F(r)\\to T_e$\n$g^4$ 相对论增亮、Planck 谱、湍流\n"
        "$d\\tau$, $e^{-d\\tau}$, $\\alpha$ 混合", C_WARM)
    arrow(12, 60, 12, 52)
    arrow(32, 60, 32, 52)
    arrow(57, 60, 57, 52)
    arrow(84, 60, 84, 52)

    box(2, 16, 22, 14, "背景星空", "立方体面哈希点星\nPlanck 色 + 银河带", C_ACCENT3)
    box(27.6, 16, 24, 14, "多普勒因子", "$g=\\nu_{obs}/\\nu_{em}$\n$=1/[u^t f(1+\\Omega b)]$", C_ACCENT3)
    box(55.2, 16, 22, 14, "色调映射", "指数压缩 + $\\gamma=1/2.2$\n物理 / 显示两种口径", C_ACCENT4)
    box(80.8, 16, 17.2, 14, "帧缓冲", "HDR 纹理\n自适应降采样", C_ACCENT4)
    arrow(24, 23, 27.6, 23)
    arrow(51.6, 23, 55.2, 23)
    arrow(77.2, 23, 80.8, 23)
    arrow(57, 38, 57, 30)
    arrow(84, 38, 84, 30)
    arrow(12, 38, 12, 30)

    ax.text(50, 8.4, "所有分支在一次片元调用内完成；积分上限 4096 步，超时像素标记为未收敛并由 UI 计数",
            ha="center", fontsize=6.4, color="#8d9ab5")
    return S.save(fig, "f18_pipeline")


# --------------------------------------------------------------------------- #
def f24_accumulation():
    """V6: progressive-accumulation convergence, repeatability and phase."""
    from matplotlib.ticker import NullLocator, NullFormatter

    here = os.path.dirname(os.path.abspath(__file__))
    d = json.load(open(os.path.join(here, "validate_accum.json"), encoding="utf-8"))
    base_path = os.path.join(here, "validate_accum_base.json")
    base = json.load(open(base_path, encoding="utf-8")) if os.path.exists(base_path) else None

    S.apply_style(dark=False)
    fig = plt.figure(figsize=(7.4, 3.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.16, 1.0],
                          height_ratios=[1.30, 1.0], wspace=0.30, hspace=0.14)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1], sharex=axB)

    # ----------------------------------------------------------------- (a) -- #
    series = [("frozen", "frozen：样本序号钉死", C_ACCENT3, "o"),
              ("r2", "r2：低差异序列", C_ACCENT2, "o"),
              ("post", "post：r2 + 颗粒/暗角", C_ACCENT, "s")]
    fits = {}
    for key, lab, col, mk in series:
        b = d["blocks"][key]
        n = np.asarray(b["schedule"], dtype=float)
        r = np.asarray(b["rms"], dtype=float)
        m = r > 0
        if m.sum() > 1:
            axA.plot(n[m], r[m], "-", color=col, lw=1.2, zorder=3)
        axA.plot(n[m], r[m], mk, color=col, ms=3.4, mfc="white", mew=1.0,
                 zorder=4, label="%s" % lab)
        # rms == 0 is not representable on a log ordinate; the solid markers sit
        # exactly on the symlog zero line, which is the point of the panel.
        axA.plot(n[~m], r[~m], mk, color=col, ms=4.2, mfc=col, mew=0.0,
                 zorder=5)
        if m.sum() > 3:
            sel = m & (n <= 1024.0)
            p = np.polyfit(np.log(n[sel]), np.log(r[sel]), 1)
            fits[key] = p
            nn = np.array([1.0, 1024.0])
            axA.plot(nn, np.exp(p[1]) * nn ** p[0], "--", lw=0.9, color=col,
                     alpha=0.55, zorder=2)

    # the pre-fix data set: same measurement, polluted by the drift of the
    # composite-pass dither phase and by autoQuality rescaling mid-schedule.
    if base is not None:
        old = base.get("frozen", {})
        no = np.asarray(old.get("schedule", []), dtype=float)
        ro = np.asarray(old.get("rms", []), dtype=float)
        if no.size == ro.size:
            axA.plot(no[ro > 0], ro[ro > 0], "x", color="#c3c8d4", ms=4.2,
                     mew=1.1, zorder=1,
                     label="修复前（抖相位漂移）")
        old2 = base.get("r2", {})
        no2 = np.asarray(old2.get("schedule", []), dtype=float)
        ro2 = np.asarray(old2.get("rms", []), dtype=float)
        if no2.size == ro2.size:
            axA.plot(no2[ro2 > 0], ro2[ro2 > 0], "x", color="#c3c8d4", ms=4.2,
                     mew=1.1, zorder=1)

    axA.set_xscale("log")
    axA.set_yscale("symlog", linthresh=2.0e-4, linscale=0.55)
    axA.set_ylim(0.0, 9.0e-2)
    axA.set_xlim(0.8, 4096.0 * 1.6)
    axA.set_yticks([0.0, 1e-4, 1e-3, 1e-2])
    axA.set_yticklabels(["0", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"])
    axA.yaxis.set_minor_locator(NullLocator())
    axA.yaxis.set_minor_formatter(NullFormatter())
    axA.set_xlabel("累积样本数 $n$")
    axA.set_ylabel("与 $n=2048$ 参考图的 rms [luma]")
    axA.set_title("(a) 累积收敛：三条曲线，两种机制", fontsize=8.2)
    S.grid(axA, which="major", alpha=0.25)

    # What the two fitted exponents are actually made of.
    axA.annotate("$\\propto n^{-0.29}$",
                 xy=(40.0, 2.09e-2), xytext=(2.05, 2.55e-2),
                 fontsize=6.6, color=C_ACCENT2,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=C_ACCENT2))
    axA.annotate("$\\propto n^{-0.28}$",
                 xy=(200.0, 1.23e-2), xytext=(2.05, 1.05e-2),
                 fontsize=6.6, color=C_ACCENT,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=C_ACCENT))
    axA.annotate("逐位相同：rms $\\equiv$ 0\n（幂等性校验，非精度校验）",
                 xy=(16.0, 0.0), xytext=(1.35, 1.6e-3),
                 fontsize=6.4, color=C_ACCENT3, va="bottom",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=C_ACCENT3))
    axA.text(0.985, 0.035,
             "实心点 = rms 触零：frozen 全部 12 点；\n"
             "r2/post 仅 $n=2048$——该点即参考本身，按定义为 0。\n"
             "重复对（$n=1,64$ 各两次）：三块的 $\\Delta$rms、\n"
             "$\\Delta$rms$_{row}$、$\\Delta$max 全为 0",
             transform=axA.transAxes, ha="right", va="bottom", fontsize=6.0,
             color="#5b6373",
             bbox=dict(boxstyle="round,pad=0.32", fc="#f2f4f8", ec="#d3d8e2", lw=0.6))
    leg = axA.legend(loc="upper right", fontsize=6.2, handlelength=1.6,
                     borderaxespad=0.2)
    leg.set_zorder(6)

    # ----------------------------------------------------------------- (b) -- #
    r2 = d["blocks"]["r2"]
    row1 = np.asarray(r2["row_n1"], dtype=float)
    rowr = np.asarray(r2["row_ref"], dtype=float)
    x = np.arange(row1.size, dtype=float)
    diff = row1 - rowr
    i0 = int(np.argmax(np.abs(diff)))
    lo = max(0, i0 - 70)
    hi = min(row1.size, i0 + 71)
    sl = slice(lo, hi)

    axB.plot(x[sl], rowr[sl], "-", color="#3d4553", lw=1.35,
             label="$n=2048$（参考）", zorder=3)
    axB.plot(x[sl], row1[sl], "--", color=C_ACCENT, lw=1.15,
             label="$n=1$（单帧）", zorder=4)
    axB.fill_between(x[sl], row1[sl], rowr[sl], color=C_ACCENT, alpha=0.14,
                     lw=0.0, zorder=2)
    axB.axvline(i0, color=C_ACCENT2, lw=0.7, ls=":", zorder=1)
    axB.annotate("轮廓处的覆盖率采样：\n全行最大偏差 $\\max|\\Delta|=0.076$",
                 xy=(i0, row1[i0]), xytext=(0.02, 0.98),
                 textcoords="axes fraction", ha="left", va="top",
                 fontsize=6.0, color="#5b6373",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#8d95a5"))
    axB.set_ylabel("中央行亮度", fontsize=7.4)
    axB.set_title("(b) 中央扫描行：单帧 vs 2048 帧", fontsize=8.2)
    axB.tick_params(labelbottom=False)
    S.grid(axB, which="major", alpha=0.22)

    axC.axhline(0.0, color="#9aa2b0", lw=0.7)
    axC.plot(x[sl], diff[sl], "-", color=C_ACCENT4, lw=1.0, zorder=3)
    axC.fill_between(x[sl], 0.0, diff[sl], where=diff[sl] >= 0,
                     color=C_ACCENT4, alpha=0.18, lw=0.0, zorder=2)
    axC.fill_between(x[sl], 0.0, diff[sl], where=diff[sl] < 0,
                     color=C_ACCENT2, alpha=0.18, lw=0.0, zorder=2)
    axC.set_xlabel("水平像素（中央行，全宽 560 px）")
    axC.set_ylabel("$\\Delta$ 亮度", fontsize=7.4)
    axC.text(0.985, 0.965,
             "全行：mean$|\\Delta|=0.0130$，median$=0.0042$，rms$=0.0209$",
             transform=axC.transAxes, ha="right", va="top", fontsize=6.0,
             color="#5b6373", zorder=6,
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                       alpha=0.78))
    axC.set_ylim(-0.115, 0.115)
    h, lb = axB.get_legend_handles_labels()
    legC = axC.legend(h, lb, loc="lower left", fontsize=6.2, ncols=2,
                      handlelength=1.5, borderaxespad=0.25)
    legC.set_zorder(6)
    S.grid(axC, which="major", alpha=0.22)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.895, bottom=0.155)
    return S.save(fig, "f24_accumulation")


FIGURES = (f14_observer_frame, f15_lensed_sky, f16_disk_image_orders,
           f17_error_budget, f18_pipeline, f24_accumulation)


def main() -> int:
    # Optional argv filter, e.g. ``python make_figures2.py f24`` rebuilds a
    # single figure; the multi-minute reference traces are then skipped.
    want = [a.lower() for a in sys.argv[1:]]
    made = []
    for fn in FIGURES:
        if want and not any(w in fn.__name__.lower() for w in want):
            continue
        try:
            made.append(fn())
            print("[ok]", os.path.basename(made[-1]), flush=True)
        except Exception as exc:
            print("[!!]", fn.__name__, "->", type(exc).__name__, exc, flush=True)
    print("figures written:", len(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
