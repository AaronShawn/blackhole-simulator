#!/usr/bin/env python3
"""V6: does the progressive accumulation actually average *different* samples?

Background
----------
The v1.0.0 shader declared a ``uJitterSeed`` uniform that ``main.js`` never
wrote, so every frame of the running mean used the *same* sub-pixel offset and
the accumulation could neither anti-alias the silhouette nor attenuate the
residual noise of the procedural star field.  v1.0.1 replaces the seed with a
monotone sample counter driving a Cranley-Patterson rotated R2 low-discrepancy
sequence, and moves the composite-pass dither phase out of the global frame
counter into an accumulation-epoch counter.

What went wrong the first time
------------------------------
The first version of this probe reported a frozen-mode plateau of 4.2e-3 and a
non-reproducible n=1 point of 3.3e-2.  Both were artefacts, and the clean
design below suppresses all three causes:

  * ``autoQuality`` was left on.  On the integrated-GPU test machine
    (Intel UHD via ANGLE/D3D11) the block averages drop below 24 fps for many
    seconds, so ``renderScale`` walked
    downwards, ``allocateTargets()`` resized every render target and the traced
    resolution changed *between* the measurements of one schedule;
  * the composite pass adds ``grain = 0.010`` of per-pixel dither whose phase
    was ``frameCount % 1024`` -- a global counter that ``resetAccumulation()``
    does not touch.  Two runs of the identical configuration therefore differed
    by a fresh dither pattern, which alone puts a floor of ~3e-3..4e-3 on 8-bit
    luma.  This release moves the phase onto the accumulation epoch;
  * the very first ``accumulateTo()`` after a stream of ``set()`` calls landed
    on a pipeline state that had not settled, which is why the old n=1 point
    was ~8x the plateau.

The probe therefore disables ``autoQuality``, ``grain`` and ``vignette``,
freezes the coordinate time, discards a warm-up render, asserts that the traced
resolution is constant across the whole schedule, and measures *repeated*
pairs at fixed ``n`` to establish the reproducibility floor of the probe
itself.

Blocks
------
  frozen  sample index pinned to 0 (the pre-1.0.1 behaviour) with the grain
          and vignette off -- every averaged frame is bit-identical, so the
          running mean must converge to that same frame for every ``n``.
  r2      R2 sample index, grain and vignette off -- the running mean averages
          genuinely different sub-pixel samples.
  post    R2 sample index with the shipped post-processing (grain 0.010,
          vignette 0.35) -- tests that the dither-phase fix made the fully
          post-processed image reproducible.

Metrics, all against the block's own ``n = N_REF`` reference image:

  rms   root-mean-square difference over the decimated grid  [0,1 luma]
  hf    mean squared difference of horizontally adjacent centre-row samples,
        a proxy for residual high-frequency noise

``window.__bh.accumulateTo(n)`` renders ``n`` frames back to back, reads the
tone-mapped LDR buffer back and returns a decimated grid plus the full centre
row, so the sample count of every reported image is exact (a screenshot would
keep accumulating while it is captured).

Run:  python paper/tools/validate_accum.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from make_render_figures import session  # noqa: E402

OUT = os.path.join(HERE, "validate_accum.json")
BASE = os.path.join(HERE, "validate_accum_base.json")

# Doubling schedule.  Sum(n) = 2*N_REF - 1 frames are rendered per block.
SCHEDULE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
N_REF = max(SCHEDULE)
# Frame counts at which the same measurement is repeated back to back.  The
# difference between the two copies is the reproducibility floor of the probe.
REPEAT = [1, 64]

VIEW = (560, 360)
SCALE = 0.5
MAX_STEPS = 120
TOL = 0.06
SIM_T = 14.0

BLOCKS = [
    {"id": "frozen", "frozen": True, "grain": 0.0, "vignette": 0.0},
    {"id": "r2", "frozen": False, "grain": 0.0, "vignette": 0.0},
    {"id": "post", "frozen": False, "grain": 0.010, "vignette": 0.35},
]


def _grid(res: dict) -> np.ndarray:
    return np.asarray(res["grid"], dtype=np.float64)


def _hf(row: np.ndarray) -> float:
    d = np.diff(row)
    return float(np.mean(d * d))


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def setup(page, block: dict) -> dict:
    """Put the renderer into the block's configuration and let it settle."""
    page.evaluate("on => window.__bh.freezeJitter(on)", block["frozen"])
    page.evaluate("v => window.__bh.set('grain', v)", block["grain"])
    page.evaluate("v => window.__bh.set('vignette', v)", block["vignette"])
    page.evaluate("v => window.__bh.set('paused', v)", True)
    page.evaluate("t => window.__bh.setTime(t)", SIM_T)
    # Discard two warm-up renders: the first accumulateTo() after a burst of
    # set() calls used to land on an unsettled pipeline state.
    for _ in range(2):
        page.evaluate("window.__bh.accumulateTo(1)")
    return page.evaluate("window.__bh.info()")


def measure(page, frozen: bool, block: dict) -> dict:
    warm = setup(page, block)
    rows: dict[int, dict] = {}
    infos: list[dict] = []
    for n in SCHEDULE:
        t0 = time.time()
        res = page.evaluate("n => window.__bh.accumulateTo(n)", n)
        ms = (time.time() - t0) * 1e3
        info = page.evaluate("window.__bh.info()")
        infos.append({"n": n, "traced": info["traced"], "scale": info["scale"],
                      "fps": info["fps"], "ms": ms})
        rows[n] = {"ms": ms, "grid": res["grid"], "row": res["row"],
                   "w": res["w"], "h": res["h"]}
        print("    [%-6s] n=%-5d %6.2f s  traced=%s"
              % (block["id"], n, (time.time() - t0), info["traced"]), flush=True)

    # Reproducibility pairs: measure the *same* n twice in a row.
    repeats = {}
    for n in REPEAT:
        a = page.evaluate("n => window.__bh.accumulateTo(n)", n)
        b = page.evaluate("n => window.__bh.accumulateTo(n)", n)
        ga, gb = _grid(a), _grid(b)
        ra, rb = np.asarray(a["row"]), np.asarray(b["row"])
        repeats[str(n)] = {
            "rms": _rms(ga, gb),
            "rms_row": _rms(ra, rb),
            "max": float(np.max(np.abs(ga - gb))),
            "row_a": ra.tolist(), "row_b": rb.tolist(),
        }
        print("    [%s] repeat n=%-5d rms=%.3e (row %.3e, max %.3e)"
              % (block["id"], n, repeats[str(n)]["rms"],
                 repeats[str(n)]["rms_row"], repeats[str(n)]["max"]), flush=True)

    ref = _grid(rows[N_REF])
    out = {"id": block["id"], "frozen": bool(frozen),
           "grain": block["grain"], "vignette": block["vignette"],
           "w": rows[N_REF]["w"], "h": rows[N_REF]["h"],
           "info_warmup": warm, "info_per_n": infos,
           "schedule": list(SCHEDULE), "rms": [], "hf": [], "ms": [],
           "hf_ref": _hf(np.asarray(rows[N_REF]["row"])),
           "row_n1": rows[1]["row"], "row_ref": rows[N_REF]["row"],
           "repeat": repeats}
    for n in SCHEDULE:
        g = _grid(rows[n])
        row = np.asarray(rows[n]["row"])
        out["rms"].append(_rms(g, ref))
        out["hf"].append(_hf(row))
        out["ms"].append(rows[n]["ms"])
    # The traced resolution must not move while a schedule is being measured.
    traced = {i["traced"] for i in infos}
    out["traced_constant"] = len(traced) == 1
    out["traced"] = sorted(traced)
    out["scale_constant"] = len({i["scale"] for i in infos}) == 1
    if not out["traced_constant"]:
        raise RuntimeError("traced resolution changed during the %s block: %s"
                           % (block["id"], sorted(traced)))
    return out


def main() -> int:
    if os.path.exists(OUT) and not os.path.exists(BASE):
        shutil.copyfile(OUT, BASE)
        print("[keep] pre-fix measurement archived at", BASE, flush=True)
    with session(*VIEW, dsf=1.0) as (page, errors):
        # Collapse the side panel: it is 336 CSS px wide and would otherwise
        # leave the render stage with a 224 x 360 aspect ratio.
        page.click("#btn-hide")
        page.wait_for_timeout(1500)     # let the debounced allocateTargets() run
        # Every confound identified above is switched off here.
        page.evaluate("window.__bh.set('autoQuality', false)")
        page.evaluate("window.__bh.set('maxSteps', %d)" % MAX_STEPS)
        page.evaluate("window.__bh.set('tol', %f)" % TOL)
        page.evaluate("window.__bh.set('renderScale', %f)" % SCALE)
        page.evaluate("window.__bh.set('accumulate', true)")
        page.evaluate("window.__bh.set('paused', true)")
        page.evaluate("window.__bh.set('shadowCircle', false)")
        page.evaluate("t => window.__bh.setTime(t)", SIM_T)
        info = page.evaluate("window.__bh.info()")
        print("[info]", json.dumps(info), flush=True)
        print("[gpu ]", info["gpu"], "traced", info["traced"], flush=True)
        data = {"viewport": list(VIEW), "scale": SCALE, "n_ref": N_REF,
                "maxSteps": MAX_STEPS, "tol": TOL, "simTime": SIM_T,
                "panel": "hidden", "autoQuality": False,
                "shadowCircle": False, "warmup_discarded": 2,
                "repeat_schedule": REPEAT, "gpu": info["gpu"],
                "traced": info["traced"], "blocks": {}}
        for block in BLOCKS:
            print("[run] block=%s grain=%s vignette=%s frozen=%s"
                  % (block["id"], block["grain"], block["vignette"],
                     block["frozen"]), flush=True)
            data["blocks"][block["id"]] = measure(page, block["frozen"], block)
        if errors:
            data["page_errors"] = errors[:20]
            print("[page]", *errors[:8], sep="\n  ", flush=True)
    data["r2"] = data["blocks"]["r2"]        # legacy-compatible aliases
    data["frozen"] = data["blocks"]["frozen"]
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print("[wrote]", OUT, flush=True)

    # Console summary ---------------------------------------------------- #
    print("\n  n       rms frozen   rms R2     rms post   rms*R2*sqrt(n)"
          "   rms*post*sqrt(n)")
    fz = data["blocks"]["frozen"]["rms"]
    rz = data["blocks"]["r2"]["rms"]
    pz = data["blocks"]["post"]["rms"]
    for i, n in enumerate(SCHEDULE):
        print("  %-7d %10.5f  %10.5f  %10.5f  %13.5f  %14.5f"
              % (n, fz[i], rz[i], pz[i], rz[i] * math.sqrt(n),
                 pz[i] * math.sqrt(n)))
    for bid in ("frozen", "r2", "post"):
        rp = data["blocks"][bid]["repeat"]
        print("  repeat floor [%-6s] n=1 %.3e   n=64 %.3e"
              % (bid, rp["1"]["rms"], rp["64"]["rms"]))
        print("    traced constant: %s %s"
              % (data["blocks"][bid]["traced_constant"],
                 data["blocks"][bid]["traced"]))
    print("\n  hf frozen n=1 %8.5f  n=2048 %8.5f"
          % (data["blocks"]["frozen"]["hf"][0],
             data["blocks"]["frozen"]["hf"][-1]))
    print("  hf R2     n=1 %8.5f  n=2048 %8.5f"
          % (data["blocks"]["r2"]["hf"][0], data["blocks"]["r2"]["hf"][-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
