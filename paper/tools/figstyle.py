"""Shared publication style for the paper figures.

Every figure is written twice: a vector PDF for LaTeX and a PNG (200 dpi) for
the README / GitHub.  Chinese and Latin glyphs are both supported.
"""

from __future__ import annotations

import os
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# The Noto CJK faces are variable fonts, so the default weight lookup prints a
# "Failed to find font weight ..." line of pure noise on every glyph run.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
PNGDIR = os.path.join(FIGDIR, "png")

# --------------------------------------------------------------------------- #
#  Palette                                                                     #
# --------------------------------------------------------------------------- #
C_BG = "#070a12"
C_GRID = "#2a3550"
C_TEXT = "#1b1f2a"
C_ACCENT = "#c9743a"
C_ACCENT2 = "#2f7fb5"
C_ACCENT3 = "#4f9d69"
C_ACCENT4 = "#8d5fb0"
C_WARM = "#e0a03c"
C_DIM = "#7a8598"

SERIES = [C_ACCENT2, C_ACCENT, C_ACCENT3, C_ACCENT4, "#b5533f", "#4a4f63"]


def _pick_font(candidates):
    have = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in have:
            return c
    return None


def apply_style(dark: bool = False, fontsize: float = 9.0) -> None:
    """Configure matplotlib for a paper figure (light) or a space-themed one."""
    # A CJK-capable *serif* face drives BOTH running text and mathtext.
    # matplotlib >= 3.11 hands any string containing "$...$" to the mathtext
    # engine in full, and the segments *outside* the dollar signs are drawn with
    # ``font.family[0]`` **without** the usual font fallback chain (verified: a
    # list starting with Times New Roman turns every Chinese character of a
    # mixed label into a dummy box).  The CJK serif therefore has to come first;
    # its Latin glyphs are Source-Serif derived and sit well next to the
    # SimSun-bodied running text of the paper.  ``SimSun``/``NSimSun`` cannot be
    # used at all because matplotlib resolves their .ttc collection to a
    # Latin-only face, and ``SimHei`` lacks the superscripts used in labels.
    cjk_serif = _pick_font(["Noto Serif SC", "Source Han Serif SC", "Songti SC",
                            "STSong", "Microsoft YaHei", "SimHei"])
    latin = _pick_font(["Times New Roman", "Nimbus Roman", "DejaVu Serif"])
    fam = [f for f in (cjk_serif, latin, "DejaVu Sans") if f]
    mathfam = cjk_serif or "DejaVu Sans"

    plt.rcParams.update({
        "font.family": fam,
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1.0,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1.0,
        "ytick.labelsize": fontsize - 1.0,
        "legend.fontsize": fontsize - 1.0,
        "axes.unicode_minus": False,
        # ``custom`` + a CJK serif lets one string carry Chinese and formulas.
        "mathtext.fontset": "custom",
        "mathtext.rm": mathfam,
        "mathtext.it": mathfam + ":italic",
        "mathtext.bf": mathfam + ":bold",
        "mathtext.sf": mathfam,
        "mathtext.tt": mathfam,
        "mathtext.cal": mathfam,
        "mathtext.default": "regular",
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
    })

    if dark:
        plt.rcParams.update({
            "figure.facecolor": C_BG,
            "axes.facecolor": C_BG,
            "savefig.facecolor": C_BG,
            "axes.edgecolor": C_GRID,
            "axes.labelcolor": "#d8e0f0",
            "text.color": "#d8e0f0",
            "xtick.color": "#9fb0c8",
            "ytick.color": "#9fb0c8",
            "grid.color": C_GRID,
            "legend.labelcolor": "#d8e0f0",
        })
    else:
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#333844",
            "axes.labelcolor": C_TEXT,
            "text.color": C_TEXT,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "grid.color": "#d8dce4",
        })


def grid(ax, which="both", alpha=0.35):
    ax.grid(True, which=which, alpha=alpha, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def save(fig, name: str, png_only=False, pdf_dpi=None) -> str:
    """Write figures/<name>.pdf and figures/png/<name>.png.

    ``pdf_dpi`` only matters for figures that contain *rasterised* artists (see
    ``f09_doppler_map``): a dense ``pcolormesh`` stored as vector quads bloats the
    PDF by megabytes, so it is rasterised instead, and this sets the resolution of
    that embedding.  Vector-only figures ignore it.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(PNGDIR, exist_ok=True)
    png = os.path.join(PNGDIR, name + ".png")
    fig.savefig(png)
    if not png_only:
        kw = {} if pdf_dpi is None else {"dpi": pdf_dpi}
        fig.savefig(os.path.join(FIGDIR, name + ".pdf"), **kw)
    plt.close(fig)
    return png


def annotate(ax, text, xy, xytext, color=None, arrow=True, fontsize=None):
    """Small helper for annotation arrows."""
    kw = dict(textcoords="offset points", fontsize=fontsize or plt.rcParams["font.size"] - 1,
              ha="left", va="center")
    if color:
        kw["color"] = color
    if arrow:
        kw["arrowprops"] = dict(arrowstyle="->", lw=0.8, color=color or "#444")
    ax.annotate(text, xy=xy, xytext=xytext, **kw)
