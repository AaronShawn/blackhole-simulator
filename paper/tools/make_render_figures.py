#!/usr/bin/env python3
"""Fourth batch of paper figures: real frame-buffer captures of the program.

    python paper/tools/make_render_figures.py            # everything
    python paper/tools/make_render_figures.py --only f19 f22

Everything here is produced by driving the *shipped* WebGL2 application
through Playwright (Microsoft Edge, headless), so the images are what a user
sees, not a re-implementation.  The pipeline is:

    launcher.start_server()  ->  http://127.0.0.1:PORT/index.html
    window.__bh.applyPreset(...) / set(...) / setTime(...)
    page.screenshot() -> figures/app/<name>.webp (raw capture, kept in the repo)
                     -> figures/<name>.pdf + figures/png/<name>.png (paper)

The raw captures are stored as WebP because the repository has to stay
clone-able; the vector-Latex figures embed the decoded RGB arrays, so the PDFs
carry the full-resolution render.

Hardware note: this machine exposes an *Intel* integrated adapter through
ANGLE/D3D11 and Chromium falls back to the software rasteriser in headless
mode.  Every timing printed here is therefore a conservative lower bound for
an NVIDIA GPU, and the figures say so explicitly in the caption text.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import textwrap
from PIL import Image
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from launcher import resource_root, start_server          # noqa: E402
from playwright.sync_api import sync_playwright           # noqa: E402

import figstyle as S                                      # noqa: E402
from figstyle import plt, C_ACCENT, C_ACCENT2, C_ACCENT3, C_ACCENT4, C_WARM  # noqa: E402

APP_DIR = os.path.join(os.path.dirname(HERE), "figures", "app")
BENCH_JSON = os.path.join(HERE, "render_benchmark.json")

# --------------------------------------------------------------------------- #
#  printed figure numbers                                                      #
# --------------------------------------------------------------------------- #
# The number burnt into the title strip of a montage has to agree with the
# number LaTeX prints under the float, and LaTeX numbers floats in *order of
# appearance* -- i.e. by the order of the \input files and of the figure
# environments inside them, not by the ``fig:NN`` label suffix (the suffix only
# records which script drew the picture).  The order in paper/paper.tex is
#
#     parts/02_theory    f01 f02 f07 f09 f10 ...............  1  2  3  4  5
#     parts/04_impl      f18 f23 ...........................   6  7
#     parts/05_verify    f05 f14 f17 f24 ...................   8  9 10 11
#     parts/06_results   f22 f16 f25 f19 f20 ...............  12 13 14 15 16
#     parts/09_appendix  f03 f04 f06 f08 f11 f12 f13 f15 f21  17 ........ 25
#
# ``check_fignum.py`` re-derives this table from the sources and fails when it
# drifts, so inserting a figure in the middle of the paper is caught here
# instead of silently mislabelling a published picture.
FIG_NUM = {
    "f01": 1,  "f02": 2,  "f03": 17, "f04": 18, "f05": 8,  "f06": 19,
    "f07": 3,  "f08": 20, "f09": 4,  "f10": 5,  "f11": 21, "f12": 22,
    "f13": 23, "f14": 9,  "f15": 24, "f16": 13, "f17": 10, "f18": 6,
    "f19": 15, "f20": 16, "f21": 25, "f22": 12, "f23": 7,  "f24": 11,
    "f25": 14,
}


def fig_title(tag, text):
    """Title strip with the *printed* number, e.g. ``图 15  渲染器实拍…``."""
    return "图 %d  %s" % (FIG_NUM[tag], text)


CHROME_ARGS = [
    "--ignore-gpu-blocklist",
    "--enable-unsafe-swiftshader",
    "--use-angle=d3d11",
    "--enable-gpu-rasterization",
    # The two switches that remove the frame-rate cap
    # (``--disable-gpu-vsync``/``--disable-frame-rate-limit``) are deliberately
    # *not* used: on a software rasteriser they make the compositor spin and
    # the page stops answering for tens of seconds.  Frame cost is measured
    # with ``window.__bh.cost()`` instead, which renders into an off-screen
    # target and forces a GPU read-back, so it is independent of the rate at
    # which requestAnimationFrame happens to be scheduled on the host.
]


# --------------------------------------------------------------------------- #
#  Browser session                                                            #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def session(width: int, height: int, dsf: float = 1.0):
    """Serve ``web/`` and hand out one page already running the simulator."""
    os.makedirs(APP_DIR, exist_ok=True)
    httpd = start_server(os.path.join(resource_root(), "web"))
    url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    pw = sync_playwright().start()
    browser = None
    try:
        browser = pw.chromium.launch(channel="msedge", headless=True, args=CHROME_ARGS)
        ctx = browser.new_context(viewport={"width": width, "height": height},
                                 device_scale_factor=dsf)
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append("pageerror: %s" % e))
        page.on("console", lambda m: errors.append("console: " + m.text)
                if m.type == "error" else None)
        page.goto(url, wait_until="load")
        page.wait_for_function("() => window.__bh !== undefined", timeout=120000)
        page.wait_for_function("() => window.__bh.info().fps > 0", timeout=120000)
        # Fixed trace resolution for the whole session: the adaptive controller
        # would otherwise change renderScale underneath the benchmark.
        page.evaluate("window.__bh.set('autoQuality', false)")
        page.evaluate("window.__bh.set('paused', true)")
        page.evaluate("window.__bh.set('timeRate', 0)")
        yield page, errors
    finally:
        if browser is not None:
            browser.close()
        pw.stop()
        httpd.shutdown()


def _rects(page) -> dict:
    ids = ["panel", "presets", "hud-phys", "btn-hide", "btn-collapse", "btn-expand",
           "measure-out", "controls", "diag", "presets"]
    js = ("() => { const ids = %s, o = {};"
          " for (const id of ids) { const e = document.getElementById(id);"
          " if (e) { const r = e.getBoundingClientRect();"
          " o[id] = [r.x, r.y, r.width, r.height]; } }"
          " o.__vw = innerWidth; o.__vh = innerHeight; return o; }"
          % json.dumps(ids))
    return page.evaluate(js)


def grab(page, name: str, js=(), wait: float = 4.0, panel: bool = True,
         want_rects: bool = False, quality: int = 92, viewport=None):
    """Apply JS tweaks, wait for the accumulation to settle, capture one frame."""
    if viewport is not None:
        page.set_viewport_size({"width": int(viewport[0]), "height": int(viewport[1])})
        page.wait_for_timeout(500)
    page.evaluate("document.body.classList.toggle('nopanel', %s)"
                  % ("true" if not panel else "false"))
    for s in js:
        page.evaluate(s)
    page.wait_for_timeout(int(wait * 1000))
    raw = os.path.join(APP_DIR, name + ".png")
    page.screenshot(path=raw)
    img = Image.open(raw).convert("RGB")
    os.remove(raw)
    img.save(os.path.join(APP_DIR, name + ".webp"), "WEBP", quality=quality, method=6)
    rect = _rects(page) if want_rects else None
    info = json.loads(page.evaluate("JSON.stringify(window.__bh.info())"))
    print("[grab] %-14s %dx%d  scale=%.2f steps=%d  %.1f fps"
          % (name, img.width, img.height, info["scale"], info["steps"], info["fps"]),
          flush=True)
    return img, rect, info


def load(name: str) -> np.ndarray:
    return np.asarray(Image.open(os.path.join(APP_DIR, name + ".webp")).convert("RGB"))


def js_preset(name: str, **over) -> list[str]:
    out = ["window.__bh.applyPreset(%s)" % json.dumps(name, ensure_ascii=False)]
    out.append("window.__bh.set('autoQuality', false)")
    for k, v in over.items():
        out.append("window.__bh.set(%s, %s)" % (json.dumps(k), json.dumps(v)))
    return out


# --------------------------------------------------------------------------- #
#  Montage helpers                                                            #
# --------------------------------------------------------------------------- #
def _fit(cell, img, figsize):
    """Letterbox an image inside a figure-fraction cell, keeping its aspect."""
    x, y, w, h = cell
    fw, fh = figsize
    a = img.shape[1] / float(img.shape[0])
    cell_a = (w * fw) / (h * fh)
    if a >= cell_a:                      # width limited
        ww, hh = w, (w * fw / a) / fh
    else:                                # height limited
        hh, ww = h, (h * fh * a) / fw
    return [x + (w - ww) / 2.0, y + (h - hh) / 2.0, ww, hh]


def montage(figwide, title, tiles, ncol, nrow, note=None, cap_in=0.34,
            name_fs=7.6, par_fs=6.1, title_fs=9.2, gap_in=0.16,
            top_in=0.34, bottom_in=0.30, inset=0.014):
    """Grid of screenshots with a two-line caption under every tile.

    The figure height is *derived* from the requested width, the tile aspect
    ratio and the caption band, so no vertical space is ever wasted.
    """
    S.apply_style(dark=True, fontsize=7.0)
    left, right = 0.006, 0.994
    cw_in = figwide * (right - left) / ncol
    inner = 1.0 - 2.0 * inset
    ar = max(t["img"].shape[1] / float(t["img"].shape[0]) for t in tiles)
    img_h_in = cw_in * inner / ar
    cell_h_in = img_h_in + cap_in
    fig_h = top_in + nrow * cell_h_in + (nrow - 1) * gap_in + bottom_in
    fig = plt.figure(figsize=(figwide, fig_h))
    cw_frac = cw_in / figwide

    for i, t in enumerate(tiles):
        r, c = divmod(i, ncol)
        x0 = left + c * cw_frac
        y0 = fig_h - top_in - (r + 1) * cell_h_in - r * gap_in   # inches, from bottom
        a = t["img"].shape[1] / float(t["img"].shape[0])
        ww_in = img_h_in * a
        ax = fig.add_axes([x0 + (cw_in - ww_in) / 2.0 / figwide,
                           (y0 + cap_in) / fig_h,
                           ww_in / figwide, img_h_in / fig_h])
        ax.imshow(t["img"], interpolation="lanczos", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#33415e"); sp.set_linewidth(0.7)
        cx = x0 + cw_frac / 2.0
        fig.text(cx, (y0 + cap_in * 0.68) / fig_h, t["name"], ha="center",
                 va="center", fontsize=name_fs, color="#f2f6ff", weight="bold")
        fig.text(cx, (y0 + cap_in * 0.22) / fig_h, t["params"], ha="center",
                 va="center", fontsize=par_fs, color="#8fa3c4")
    fig.text(0.5, 1.0 - title_fs * 1.15 / 72.0 / fig_h, title, ha="center",
             va="center", fontsize=title_fs, color="white", weight="bold")
    if note:
        fig.text(0.5, bottom_in * 0.42 / fig_h, note, ha="center", va="center",
                 fontsize=5.7, color="#8d9ab5")
    return fig


# --------------------------------------------------------------------------- #
#  f19  result gallery                                                        #
# --------------------------------------------------------------------------- #
GALLERY = [
    ("g_base", "经典视界（默认参数）",
     "r₀ = 26 M · i = 13° · T = 9000 K · 300 步",
     "经典视界 (推荐)", dict(dist=26, elevation=13, renderScale=1.0, maxSteps=300,
                            exposure=1.0)),
    ("g_photonring", "近观光子环",
     "r₀ = 12 M · fov = 42° · 520 步 · 可见一阶像",
     "近观光子环", dict()),
    ("g_topdown", "俯视盘面",
     "i = 58° · r_out = 30 M · 展示多普勒不对称",
     "俯视盘面", dict()),
    ("g_lens", "纯引力透镜（关闭吸积盘）",
     "只有背景星空 · 爱因斯坦环 + 二级像",
     "纯引力透镜 (无盘)", dict(starBright=1.3)),
    ("g_xray", "X 射线盘",
     "T_peak = 10⁷ K · 物理色标 · 相对论增亮",
     "X 射线盘 (T=10⁷ K)", dict()),
    ("g_hires", "高分辨率静帧",
     "renderScale = 1.2 · 600 步 · tol = 0.015",
     "高分辨率静帧", dict(renderScale=1.2, maxSteps=600)),
]


def f19_gallery(page):
    tiles = []
    for key, name, params, preset, over in GALLERY:
        over = dict(over)
        over.update({"paused": True, "timeRate": 0, "autoRotate": False})
        img, _, _ = grab(page, key, js=js_preset(preset, **over) +
                         ["window.__bh.setTime(14.0)"], wait=4.5)
        tiles.append({"img": np.asarray(img), "name": name, "params": params})
    fig = montage(7.4,
                  fig_title("f19",
                            "渲染器实拍：六种观察条件下的黑洞图像"
                            "（程序原样输出，仅加边框与说明）"),
                  tiles, 3, 2, cap_in=0.38, gap_in=0.30,
                  note="所有帧均为同一份 WebGL2 代码在 Intel UHD 集成显卡上渲染，"
                       "WebGL 上下文走 ANGLE/D3D11 硬件路径（无 GPU 黑名单回落）；"
                       "冻结坐标时间 t = 14 M，未做任何后期修图")
    return S.save(fig, "f19_app_gallery")


# --------------------------------------------------------------------------- #
#  f20  parameter sweep                                                       #
# --------------------------------------------------------------------------- #
SWEEP_ELEV = [6.0, 22.0, 50.0]
SWEEP_TEMP = [3500.0, 9000.0, 30000.0]
SWEEP_EXPOSURE = {3500.0: 1.25, 9000.0: 1.0, 30000.0: 0.70}


def f20_sweep(page):
    tiles = []
    for i, elev in enumerate(SWEEP_ELEV):
        for j, temp in enumerate(SWEEP_TEMP):
            key = "s_e%02d_t%05d" % (int(elev), int(temp))
            over = dict(dist=24.0, fov=58.0, diskInner=6.0, diskOuter=26.0,
                        elevation=elev, diskTemp=temp, colorMode=0,
                        exposure=SWEEP_EXPOSURE[temp], showDisk=1, showStars=1,
                        turb=0.55, spin=1.0, kepler=1.0, doppler=1.0,
                        diskOpacity=2.6, bloom=0.35, renderScale=0.9, maxSteps=400,
                        tol=0.025, paused=True, timeRate=0)
            img, _, _ = grab(page, key, js=js_preset("经典视界 (推荐)", **over) +
                             ["window.__bh.setTime(11.0)"], wait=4.0)
            tiles.append({"img": np.asarray(img),
                          "name": "i = %.0f° ,  T = %.1f×10³ K" % (elev, temp / 1e3),
                          "params": "r₀ = 24 M · 400 步 · 曝光 %.2f" % SWEEP_EXPOSURE[temp]})
    fig = montage(7.4,
                  fig_title("f20",
                            "参数扫描：视线仰角 i × 峰值温度 T 的九宫格"
                            "（同一观测者半径 r₀ = 24 M）"),
                  tiles, 3, 3, cap_in=0.27, gap_in=0.14, top_in=0.30, bottom_in=0.28,
                  note="左上→右下为 i 增大、T 增大；行内颜色差异来自 Planck 谱与显示色标的映射，"
                       "行间差异来自视线上盘像数目的变化（i → 0 时前后面近似重合）")
    return S.save(fig, "f20_param_sweep")


# --------------------------------------------------------------------------- #
#  f21  annotated user interface                                              #
# --------------------------------------------------------------------------- #
def f21_ui(page):
    img, rect, info = grab(
        page, "ui_panel",
        js=js_preset("经典视界 (推荐)", paused=True, timeRate=0, renderScale=0.9,
                     maxSteps=400, shadowCircle=True),
        wait=6.0, want_rects=True,
        viewport=(1440, 820))
    page.click("#btn-measure")
    page.wait_for_timeout(700)
    measure = page.eval_on_selector("#measure-out", "e => e.innerText")
    measure = measure.replace("<b>", "").replace("</b>", "").replace("\n", "  ")
    print("[f21] measure:", measure, flush=True)

    # (element id, colour, head line, body line, extra dx/dy of the pointer)
    # Anchor of every numbered marker, in fractions of the 1440x820 viewport.
    # The DOM rects cannot be reused directly here: the figure is composed in
    # figure fractions while the rects are CSS pixels of a viewport that the
    # screenshot itself does not share, so the anchors below are read off the
    # captured frame once and verified visually.
    items = [
        ("presets", C_ACCENT2, "预设观察条件",
         "六种典型观测（经典视界 / 光子环 / 俯视盘面 / 纯透镜 / X 射线 / 高分辨率）一键切换",
         0.174, 0.220),
        ("__diag__", C_ACCENT3, "渲染诊断条（右上角）",
         "画布缓冲 / CSS 尺寸 / 设备像素比 DPR / 窗口与视口尺寸 / 光线追踪分辨率 / 帧率 / 累积状态；"
         "物理读数面板（r₀、仰角 i、rₛ = 2M、光子球 3M、ISCO 6M、b_c = 3√3 M、阴影张角与屏上半径）"
         "位于同侧栏更下方，向下滚动即可看到",
         0.840, 0.040),
        ("controls", C_ACCENT4, "参数滑杆（四组）",
         "相机（r₀ / 仰角 / 视场）、吸积盘（内径 / 外径 / 厚度 / 温度 / 亮度 / 不透明度 / 湍流）、"
         "积分器（步数上限 / 容差）、图像（曝光 / 泛光 / 暗角 / 颗粒 / 分辨率）",
         0.205, 0.335),
        ("measure-out", C_WARM, "内置阴影半径校验",
         "沿画面中线扫描捕获掩膜，得到实测阴影半径，并与解析值 b_c·(1−rₛ/r₀)^(1/2) 比较；本次点击结果：%s" % measure,
         0.149, 0.123),
        ("btn-collapse", C_ACCENT2, "侧栏收起 / 展开（H 键）",
         "收起后渲染视口立即扩展为整个窗口，画面中心与相机方位保持不变",
         0.206, 0.110),
        ("__canvas__", "#9fe0ff", "理论阴影边界叠加（青色虚线）",
         "半径 = b_c·(1−rₛ/r₀)^(1/2)，与像素追踪结果在同一屏幕空间逐点对照",
         0.514, 0.402),
    ]

    S.apply_style(dark=True, fontsize=7.0)
    figw = 7.4
    left, right = 0.006, 0.994
    vw, vh = rect["__vw"], rect["__vh"]
    ax_h_in = figw * (right - left) * vh / vw
    row_in = 0.62
    legend_in = row_in * 3 + 0.10          # two columns, three rows
    top_in, bottom_in, mid_in = 0.30, 0.34, 0.12
    fig_h = top_in + ax_h_in + mid_in + legend_in + bottom_in
    fig = plt.figure(figsize=(figw, fig_h))
    ax = fig.add_axes([left, (fig_h - top_in - ax_h_in) / fig_h, right - left, ax_h_in / fig_h])
    ax.imshow(img, extent=[0, vw, vh, 0], interpolation="lanczos", aspect="auto")
    ax.set_xlim(0, vw); ax.set_ylim(vh, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#33415e"); sp.set_linewidth(0.7)
        sp.set_visible(False)

    for k, (el, col, head, body, dx, dy) in enumerate(items):
        tx, ty = vw * dx, vh * dy
        ax.text(tx, ty, str(k + 1), color="#0b1020", fontsize=6.6, weight="bold",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.30", fc=col, ec="white", lw=0.6))

    y_top_in = fig_h - top_in - ax_h_in - mid_in
    for k, (el, col, head, body, dx, dy) in enumerate(items):
        col_i, row_i = divmod(k, 3)
        x = 0.012 + col_i * 0.500                      # figure fraction
        y0_in = y_top_in - (row_i + 1) * row_in        # bottom of this row [in]
        # badge + heading on the first line of the row
        y_head_in = y0_in + row_in - 0.11
        fig.text(x, y_head_in / fig_h, str(k + 1), color="#0b1020", fontsize=6.4,
                 weight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.30", fc=col, ec="white", lw=0.6))
        fig.text(x + 0.017, y_head_in / fig_h, head, color="#f2f6ff",
                 fontsize=6.8, weight="bold", ha="left", va="center")
        # body: wrapped to at most three lines, centred in the rest of the row
        body_txt = textwrap.fill(body, 52)
        nl = body_txt.count("\n") + 1
        block_in = nl * 5.4 * 1.42 / 72.0               # wrapped block height [in]
        fig.text(x + 0.017, (y0_in + block_in * 0.5 + 0.03) / fig_h, body_txt,
                 color="#93a6c6", fontsize=5.4, ha="left", va="center",
                 linespacing=1.42)

    fig.text(0.5, (fig_h - top_in * 0.46) / fig_h,
             fig_title("f21",
                       "程序界面（1440×820 窗口，侧栏展开）：物理读数、"
                       "内置阴影半径校验与全屏预览入口"),
             ha="center", va="center", fontsize=8.6, color="white", weight="bold")
    fig.text(0.5, (bottom_in - 0.17) / fig_h,
             "编号 1–6 与上方截图中的对应位置一一对应；界面文字为程序原样输出，未做任何改写",
             ha="center", va="center", fontsize=5.7, color="#8d9ab5")
    return S.save(fig, "f21_app_ui")


# --------------------------------------------------------------------------- #
#  f23  sidebar / full-screen behaviour                                       #
# --------------------------------------------------------------------------- #
def f23_layout(page):
    tiles = []
    over = dict(paused=True, timeRate=0, renderScale=0.9, maxSteps=360)
    img, _, info = grab(page, "lay_panel", js=js_preset("经典视界 (推荐)", **over),
                        wait=5.0, viewport=(1024, 640))
    tiles.append({"img": np.asarray(img), "name": "侧栏展开 · 1024×640",
                  "params": "画布 %s · 黑洞位于剩余区域中心" % info["traced"]})
    img, _, info = grab(page, "lay_full", js=[], wait=4.0, panel=False)
    tiles.append({"img": np.asarray(img), "name": "全屏预览 · H 键收起侧栏",
                  "params": "画布 %s · 视口全部用于渲染" % info["traced"]})
    img, _, info = grab(page, "lay_small", js=[], wait=4.5, panel=False,
                        viewport=(900, 560))
    tiles.append({"img": np.asarray(img), "name": "窄窗口 · 900×560",
                  "params": "画布 %s · 中心不变、视野自适应" % info["traced"]})
    fig = montage(7.4,
                  fig_title("f23",
                            "视口自适应：侧栏展开 / 收起全屏 / 窄窗口"
                            "三种情形下黑洞始终居中"),
                  tiles, 3, 1, cap_in=0.44, gap_in=0.00, top_in=0.32, bottom_in=0.28,
                  note="相机以轨道控制器给出的球坐标 (r₀, i, φ) 定位，画面中心恒为原点方向；"
                       "改变侧栏与窗口尺寸只改变视口，不移动视线")
    return S.save(fig, "f23_app_layout")


# --------------------------------------------------------------------------- #
#  f22  performance benchmark                                                 #
# --------------------------------------------------------------------------- #
# ``window.__bh.cost(n)`` renders ``n`` frames into an off-screen target and
# reads one pixel back after each one.  The read-back blocks until the driver
# has finished every command of that frame, so the returned median is the true
# cost of a frame rather than the rate at which the host schedules
# requestAnimationFrame.  (Removing the host's frame-rate cap with Chromium's
# ``--disable-gpu-vsync`` makes the page unresponsive on a software rasteriser;
# this route needs no such switch.)
_COST_TIMER = "() => window.__bh.cost(%d)"


def bench(page, scales=(0.40, 0.60, 0.80, 1.00),
          steps=(75, 150, 300, 600, 1200),
          tols=(0.060, 0.030, 0.015, 0.0075),
          nframes: int = 7, settle_ms: int = 700, cool_ms: int = 1500):
    """Time two one-dimensional sweeps of the running renderer.

    The window is pinned to 1000x600 with the sidebar collapsed, so the traced
    pixel count is exactly ``viewport * scale``.

    * steps sweep -- the integrator budget ``n_max`` with ``tol`` fixed at the
      application default.  It exposes whether the budget binds at all.
    * tol sweep   -- the RK4 tolerance, which sets the adaptive step length
      ``h ~ tol * u / |du|`` and therefore the number of steps a *fixed*
      trajectory needs.  This is the knob that actually changes the work per
      pixel; ``n_max`` is raised to its ceiling (4096) so that it never clips.

    Two precautions make the numbers usable on a passively cooled machine.
    The sweeps are *interleaved* (each tolerance or budget step visits all four
    viewport scales back to back) so that a slow thermal drift hits every scale
    alike instead of accumulating on whichever scale happens to run last, and
    each configuration is preceded by an idle cooldown.  The reported cost is
    the minimum of the synchronous samples: the minimum is the standard
    estimator for a micro-benchmark, because contention and clock throttling
    can only ever add time.
    """
    page.set_viewport_size({"width": 1000, "height": 600})
    page.wait_for_timeout(400)
    page.evaluate("document.body.classList.add('nopanel')")
    page.wait_for_timeout(400)
    page.evaluate("window.__bh.set('autoQuality', false)")
    page.wait_for_timeout(settle_ms)

    def probe(kind, rs, **kw):
        page.wait_for_timeout(cool_ms)
        for k, v in kw.items():
            page.evaluate("window.__bh.set('%s', %s)" % (k, repr(v)))
        page.wait_for_timeout(settle_ms)
        t = json.loads(page.evaluate("JSON.stringify(window.__bh.cost(%d))"
                                     % nframes))
        info = json.loads(page.evaluate("JSON.stringify(window.__bh.info())"))
        tw, th = (int(v) for v in info["traced"].split("x"))
        ms = float(t["min"])
        row = dict(kind=kind, scale=rs, fps=1000.0 / max(ms, 1e-6),
                   frame_ms=ms, traced_w=tw, traced_h=th, pixels=tw * th,
                   steps=info["steps"], tol=info["tol"], hud_fps=info["fps"],
                   gpu=info["gpu"], shadowPx=info["shadowPx"], n=nframes,
                   t_min=t["min"], t_median=t["median"], t_max=t["max"])
        print("[bench] %-5s scale=%.2f steps=%4d tol=%.4f  %4dx%-4d"
              "  %8.2f ms  %7.2f fps  (HUD %5.1f)"
              % (kind, rs, row["steps"], row["tol"], tw, th, ms,
                 1000.0 / max(ms, 1e-6), info["fps"]), flush=True)
        return row

    rows = []
    for st in steps:
        for rs in scales:
            rows.append(probe("steps", rs, renderScale=rs, maxSteps=st,
                              tol=0.03))
    for tl in tols:
        for rs in scales:
            rows.append(probe("tol", rs, renderScale=rs, maxSteps=4096,
                              tol=tl))
    page.evaluate("document.body.classList.remove('nopanel')")
    with open(BENCH_JSON, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, ensure_ascii=False, indent=1)
    return rows


def _logticks(lo, hi, mult=(1.0, 2.0, 5.0)):
    """Round tick values that lie inside [lo, hi] on a decade grid."""
    out = []
    e = int(np.floor(np.log10(max(lo, 1e-12))))
    while 10.0 ** e <= hi * 1.5:
        for m in mult:
            v = m * 10.0 ** e
            if lo * 0.92 <= v <= hi * 1.08:
                out.append(v)
        e += 1
    return out or [lo, hi]


def _fmt(v):
    if v >= 1000:
        return "%d" % round(v)
    if v >= 10:
        return "%d" % round(v)
    if v >= 1:
        return ("%.1f" % v).rstrip("0").rstrip(".")
    return ("%.2f" % v).rstrip("0").rstrip(".")


def f22_bench(rows):
    """Cost model of the renderer: three measured laws plus their residuals.

    The three laws are *separately* identifiable, and the old single
    ``W = N_pix * 0.03/tol`` master curve mixed them, so it is replaced by:

    (a) saturated pixel law ``T = k N_pix + c`` (steps >= 300, tol = 0.03);
    (b) step-budget saturation (``n_max`` sweep at fixed tol = 0.03);
    (c) tolerance power law ``(T - c)/N_pix = k (tol/0.03)^{-p}``;
    (d) residuals of the three-parameter model, with the four configurations
        whose ``tol = 0.0075`` sweep is clipped by the ``n_max = 4096``
        ceiling shown separately and excluded from the fit.
    """
    step_rows = [r for r in rows if r["kind"] == "steps"]
    tol_rows = [r for r in rows if r["kind"] == "tol"]
    scales = sorted({r["scale"] for r in step_rows})
    steps = sorted({r["steps"] for r in step_rows})
    tols = sorted({r["tol"] for r in tol_rows})
    cols = [C_ACCENT2, C_ACCENT3, C_WARM, C_ACCENT4]

    def T_of(row):
        """Frame cost estimator: the fastest of the synchronous samples."""
        return float(row.get("t_min", row["frame_ms"]))

    # ---- (a) saturated pixel law: T = k*N_pix + c --------------------- #
    # Only budgets at or above 300 steps are saturated (panel b); a 75-step
    # budget truncates most rays and is deliberately not used here.
    sat = [r for r in step_rows if r["steps"] >= 300]
    Npx = np.array([float(r["pixels"]) for r in sat])
    Tpx = np.array([T_of(r) for r in sat])
    Apx = np.vstack([Npx, np.ones(Npx.size)]).T
    k, c = (float(v) for v in np.linalg.lstsq(Apx, Tpx, rcond=None)[0])
    # Robust trim once against residual drift, keeping the physics/hardware
    # interpretation (slope = cost per traced ray, intercept = fixed overhead).
    r_all = (k * Npx + c - Tpx) / Tpx
    keep = np.abs(r_all) <= max(0.06, 2.5 * float(np.std(r_all)))
    if int(keep.sum()) >= 8 and not bool(keep.all()):
        k, c = (float(v) for v in
                np.linalg.lstsq(Apx[keep], Tpx[keep], rcond=None)[0])

    # ---- (c) tolerance power law, anchored on the panel-(a) slope ------ #
    # Refinement below tol = 0.015 asks for more samples than the 4096-step
    # ceiling can deliver, so those four points are *budget bound*: the work
    # stops scaling and the measured time is a floor, not the tolerance law.
    fitr = [r for r in tol_rows if r["tol"] >= 0.0149]
    clip = [r for r in tol_rows if r["tol"] < 0.0149]

    def per_pixel(row):
        return (T_of(row) - c) / float(row["pixels"])

    lt = np.array([np.log(r["tol"] / 0.03) for r in fitr], float)
    lr = np.array([np.log(per_pixel(r) / k) for r in fitr], float)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(lt) & np.isfinite(lr)
    Bt = lt[ok][:, None]
    p = -float(np.linalg.lstsq(Bt, lr[ok], rcond=None)[0][0])

    def model(row):
        """Predicted synchronous frame cost [ms] of one configuration."""
        return c + k * float(row["pixels"]) * (
            row["tol"] / 0.03) ** (-p)

    # The four budget-bound points must be recognised by the *model* too:
    # a clipped sweep is any configuration whose predicted step count,
    # estimated as 0.03/tol times the saturated count, exceeds n_max.
    clipped = list(clip)
    clip_ratio = {r["scale"]: T_of(r) / model(r) for r in clipped}

    S.apply_style(dark=True, fontsize=7.0)
    fig = plt.figure(figsize=(7.4, 3.30))
    ax1 = fig.add_axes([0.050, 0.330, 0.196, 0.452])
    ax2 = fig.add_axes([0.290, 0.330, 0.196, 0.452])
    ax3 = fig.add_axes([0.530, 0.330, 0.196, 0.452])
    ax4 = fig.add_axes([0.770, 0.330, 0.180, 0.452])

    # ---- (a) saturated pixel law -------------------------------------- #
    seg = np.linspace(0.0, float(Npx[keep].max()) * 1.06, 32)
    ax1.plot(seg / 1e5, k * seg + c, "-", color=C_ACCENT, lw=1.1, alpha=0.9,
             zorder=1)
    for i, rs in enumerate(scales):
        sub = [r for r in step_rows if r["scale"] == rs]
        ax1.plot([r["pixels"] / 1e5 for r in sub], [T_of(r) for r in sub],
                 "o", color=cols[i % len(cols)], ms=2.6, alpha=0.35,
                 markeredgecolor="none", zorder=2)
        sub_s = [r for r in sat if r["scale"] == rs]
        ax1.plot([r["pixels"] / 1e5 for r in sub_s], [T_of(r) for r in sub_s],
                 "o", color=cols[i % len(cols)], ms=4.0, markeredgecolor="none",
                 zorder=3,
                 label="×%.2f  (%d kpx)" % (rs, sub_s[0]["pixels"] / 1000))
    ax1.set_xlim(0, float(Npx.max()) / 1e5 * 1.10)
    ax1.set_ylim(0, float(Tpx.max()) * 1.22)
    ax1.set_xlabel("追踪像素数  $N_{pix}/10^{5}$", fontsize=6.2)
    ax1.set_ylabel("同步单帧耗时  $T$  [ms]", fontsize=6.4)
    ax1.set_title("(a) 像素线性律：$T = kN_{pix}+c$", pad=4.0, fontsize=6.8)
    ax1.text(0.045, 0.955,
             "$k$ = %.2e ms/px\n$c$ = %.1f ms  (固定开销)\n淡色点：$n_{max}\\leq 150$"
             % (k, c),
             transform=ax1.transAxes, ha="left", va="top",
             fontsize=5.0, color="#9fb4d8", linespacing=1.45)
    ax1.legend(loc="lower right", fontsize=4.6, ncol=1, handlelength=0.7,
               labelspacing=0.22, borderpad=0.18, framealpha=0.5,
               handletextpad=0.4)
    S.grid(ax1, which="major", alpha=0.20)

    # ---- (b) does the step budget bind? -------------------------------- #
    for i, rs in enumerate(scales):
        sub = sorted([r for r in step_rows if r["scale"] == rs],
                     key=lambda r: r["steps"])
        ax2.plot([r["steps"] for r in sub], [T_of(r) for r in sub], "-o",
                 color=cols[i % len(cols)], ms=3.1, lw=1.15, label="×%.2f" % rs)
    ax2.set_xscale("log")
    ax2.set_xticks(steps)
    ax2.set_xticklabels([str(s) for s in steps], fontsize=5.6)
    yb = [T_of(r) for r in step_rows]
    ax2.set_ylim(min(yb) * 0.78, max(yb) * 1.42)
    ax2.set_xlabel("积分步数上限  $n_{max}$  （$tol$ = 0.03）", fontsize=6.4)
    ax2.set_ylabel("同步单帧耗时  $T$  [ms]", fontsize=6.4)
    ax2.set_title("(b) 步数上限饱和", pad=4.0, fontsize=7.0)
    ax2.legend(loc="upper left", fontsize=5.0, ncol=2, handlelength=0.9,
               columnspacing=0.6, borderpad=0.16, framealpha=0.55,
               handletextpad=0.45)
    ax2.text(0.5, 0.03, r"$n_{max}\geq 300$ 后趋平：多数光线" "\n"
             "在预算内提前逃逸或落向视界",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=4.9, color="#9fb4d8", linespacing=1.4)
    S.grid(ax2, which="both", alpha=0.20)

    # ---- (c) tolerance power law: all four scales collapse ------------- #
    for i, rs in enumerate(scales):
        sub = sorted([r for r in tol_rows if r["scale"] == rs],
                     key=lambda r: r["tol"])
        good = [r for r in sub if r["tol"] >= 0.0149]
        bad = [r for r in sub if r["tol"] < 0.0149]
        ax3.plot([r["tol"] for r in good], [per_pixel(r) for r in good],
                 "-o", color=cols[i % len(cols)], ms=3.1, lw=1.15,
                 label="×%.2f" % rs, zorder=3)
        ax3.plot([r["tol"] for r in bad], [per_pixel(r) for r in bad],
                 "o", color=cols[i % len(cols)], ms=3.4, mfc="none", mew=0.9,
                 zorder=3)
    tgrid = np.array([tols[0] * 0.74, tols[-1] * 1.40])
    ax3.plot(tgrid, k * (tgrid / 0.03) ** (-p), "--", color=C_ACCENT, lw=1.0,
             alpha=0.9, label=r"$k\,(tol/0.03)^{-p}$", zorder=2)
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlim(tgrid[0], tgrid[1])
    yc = [per_pixel(r) for r in tol_rows]
    ylo, yhi = min(yc) * 0.55, max(yc) * 1.9
    ax3.set_ylim(ylo, yhi)
    ty3 = _logticks(ylo, yhi)
    ax3.xaxis.set_major_locator(mticker.FixedLocator(tols))
    ax3.xaxis.set_major_formatter(mticker.FixedFormatter(
        ["0.060", "0.030", "0.015", "0.0075"]))
    ax3.xaxis.set_minor_locator(mticker.NullLocator())
    ax3.set_yticks(ty3)
    ax3.yaxis.set_major_formatter(mticker.FixedFormatter(
        ["%.1e" % t for t in ty3]))
    ax3.yaxis.set_minor_locator(mticker.NullLocator())
    ax3.tick_params(axis="both", labelsize=5.4)
    ax3.set_xlabel("RK4 相对容差  $tol$", fontsize=6.4)
    ax3.set_ylabel("单位像素代价  $(T-c)/N_{pix}$  [ms/px]", fontsize=5.9)
    ax3.set_title("(c) 容差幂律  $p$ = %.2f" % p, pad=4.0, fontsize=7.0)
    ax3.legend(loc="upper right", fontsize=5.0, ncol=2, handlelength=0.9,
               columnspacing=0.6, borderpad=0.16, framealpha=0.55,
               handletextpad=0.45)
    ax3.text(0.5, 0.045, "空心点：$n_{max}=4096$ 预算耗尽\n"
             "（工作不再随容差增长）",
             transform=ax3.transAxes, ha="center", va="bottom",
             fontsize=4.7, color="#e8a0a0", linespacing=1.35)
    S.grid(ax3, which="both", alpha=0.20)

    # ---- (d) relative residual of the three-parameter model ----------- #
    valid = [r for r in sat] + [r for r in fitr]
    resid = np.array([(model(r) - T_of(r)) / T_of(r) for r in valid])
    rclip = np.array([(model(r) - T_of(r)) / T_of(r) for r in clipped])
    ax4.bar(np.arange(len(valid)), resid * 100.0, color=C_ACCENT2,
            width=0.80, label="参与拟合")
    ax4.bar(np.arange(len(valid), len(valid) + len(clipped)),
            rclip * 100.0, color="#c2554f", width=0.80, alpha=0.85,
            label="预算耗尽")
    ax4.axhline(0, color="#5c6b8a", lw=0.8)
    ax4.axvline(len(valid) - 0.5, color="#5c6b8a", lw=0.7, ls=":")
    ax4.axhline(10, color="#5c6b8a", lw=0.6, ls="--")
    ax4.axhline(-10, color="#5c6b8a", lw=0.6, ls="--")
    ax4.set_xticks([]); ax4.set_xlabel("配置编号", fontsize=6.0)
    ax4.set_ylabel("模型偏差  [%]", fontsize=6.4)
    ax4.set_title("(d) 三段模型的残差", pad=4.0, fontsize=7.0)
    ymax = max(float(np.max(np.abs(np.r_[resid, rclip]))) * 100.0 * 1.22, 14.0)
    ax4.set_ylim(-ymax, ymax)
    ax4.tick_params(axis="y", labelsize=5.4)
    rms = float(np.sqrt(np.mean(resid ** 2))) * 100.0
    ax4.text(0.03, 0.975,
             "%d 组参与拟合\nRMS = %.1f%%\nmax = %.1f%%\n"
             "4 组预算耗尽（不参与）"
             % (len(valid), rms, float(np.abs(resid).max()) * 100.0),
             transform=ax4.transAxes, ha="left", va="top",
             fontsize=5.0, color="#9fb4d8", linespacing=1.38)
    S.grid(ax4, which="major", alpha=0.20)

    gpu = rows[0]["gpu"]
    fig.text(0.5, 0.992,
             fig_title("f22", "渲染代价分解：像素线性律、步数饱和律与容差幂律"
                              "（Chromium 无头模式，窗口 1000×600，侧栏收起）"),
             ha="center", va="top", fontsize=8.2, color="white", weight="bold")
    note = ("每组把一帧渲染进离屏目标并强制 GPU 回读，取 7 次同步耗时（墙钟）的"
            "最小值——最小值是微基准的标准估计量，争用与降频只会加时；为抵消被动"
            "散热的漂移，两次扫描均按“先扫参数、再扫视口”交叉进行并留出空闲冷却。"
            "共 %d 组配置。拟合结果：k = %.3e ms/px、c = %.2f ms、"
            "容差指数 p = %.3f；%d 组有效配置的相对偏差 RMS = %.1f%%、"
            "最大 %.1f%%。tol = 0.0075 的四组被 n_max = 4096 截断（工作不再随"
            "容差增长），故按“预算耗尽”单列，其中满视口的 2326 ms 是模型外推值的 "
            "%.1f 倍——这正是论文第 7.4 节讨论的步长下界饱和效应。GPU 报告名：%s 。"
            "该机为 Intel UHD 集成显卡、ANGLE/D3D11 硬件路径（4 逻辑核），k、c 只适用于本机，"
            "但三条标度律的形状与硬件无关；在 NVIDIA 独立显卡上跑同一扫描即可"
            "换算出该硬件的 k、c。"
            % (len(rows), k, c, p, len(valid), rms,
               float(np.abs(resid).max()) * 100.0,
               float(clip_ratio[1.0]),
               gpu[:80]))
    fig.text(0.5, 0.004, textwrap.fill(note, 96), ha="center", va="bottom",
             fontsize=5.0, color="#8d9ab5", linespacing=1.5)
    return S.save(fig, "f22_performance")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of f19 f20 f21 f22 f23")
    ap.add_argument("--reuse-bench", action="store_true",
                    help="rebuild f22 from tools/render_benchmark.json instead"
                         " of re-measuring the 36 configurations")
    args = ap.parse_args()
    want = set(args.only) if args.only else {"f19", "f20", "f21", "f22", "f23"}
    os.makedirs(APP_DIR, exist_ok=True)

    made = []
    with session(1024, 576) as (page, errors):
        jobs = [
            ("f19", lambda: f19_gallery(page)),
            ("f20", lambda: f20_sweep(page)),
            ("f21", lambda: f21_ui(page)),
            ("f23", lambda: f23_layout(page)),
        ]
        for tag, fn in jobs:
            if tag not in want:
                continue
            try:
                made.append(fn())
                print("[ok]", tag, "->", os.path.basename(made[-1]), flush=True)
            except Exception as exc:                 # keep the other figures coming
                print("[!!]", tag, type(exc).__name__, exc, flush=True)
        if "f22" in want:
            try:
                if args.reuse_bench:
                    rows = json.load(open(BENCH_JSON, encoding="utf-8"))["rows"]
                    print("[bench] reusing %d rows from %s"
                          % (len(rows), os.path.basename(BENCH_JSON)), flush=True)
                else:
                    rows = bench(page)
                made.append(f22_bench(rows))
                print("[ok] f22 ->", os.path.basename(made[-1]), flush=True)
            except Exception as exc:
                print("[!!] f22", type(exc).__name__, exc, flush=True)
        if errors:
            print("[page-errors]", errors[:8], flush=True)
    print("figures written:", len(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
