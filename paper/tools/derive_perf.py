"""Derive the performance-model constants quoted in the paper.

Reads ``render_benchmark.json`` (produced by
``make_render_figures.py --only f22 --from-json``'s recorder, i.e. the
``bench()`` routine driving ``window.__bh.cost(n)`` inside the running
renderer) and emits ``perf_derived.json`` with:

* (a) the saturated pixel law  ``T = k N_pix + c``  (steps >= 300, tol = 0.03)
      fitted with the same single robust trim used by ``f22_bench``;
* (c) the tolerance power law  ``(T - c)/N_pix = k (tol/0.03)^-p``;
* (d) the residual vector of the three-parameter model, with the four
      ``tol = 0.0075`` configurations that are clipped by the ``n_max = 4096``
      ceiling reported separately and excluded from the fit.

Run from the repository root::

    python paper/tools/derive_perf.py
"""

from __future__ import annotations

import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "render_benchmark.json")
OUT = os.path.join(HERE, "perf_derived.json")


def T_of(row):
    """Frame cost of one benchmark row: the fastest synchronous sample."""
    return float(row.get("t_min", row["frame_ms"]))


def fit_pixel_law(step_rows):
    """T = k N_pix + c on the saturated budget rows, with one robust trim."""
    sat = [r for r in step_rows if r["steps"] >= 300]
    Npx = np.array([float(r["pixels"]) for r in sat])
    Tpx = np.array([T_of(r) for r in sat])
    A = np.vstack([Npx, np.ones(Npx.size)]).T
    k, c = (float(v) for v in np.linalg.lstsq(A, Tpx, rcond=None)[0])
    r_all = (k * Npx + c - Tpx) / Tpx
    keep = np.abs(r_all) <= max(0.06, 2.5 * float(np.std(r_all)))
    trimmed = bool(not keep.all() and keep.sum() >= 8)
    if trimmed:
        k, c = (float(v) for v in np.linalg.lstsq(A[keep], Tpx[keep], rcond=None)[0])
    pred = k * Npx + c
    rel = (pred - Tpx) / Tpx
    return {
        "k_ms_per_px": k,
        "c_ms": c,
        "n_points": int(Npx.size),
        "n_kept": int(keep.sum()),
        "trimmed": trimmed,
        "rms_ms": float(np.sqrt(np.mean((pred - Tpx) ** 2))),
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
        "max_abs_rel": float(np.max(np.abs(rel))),
        "points": [
            {"pixels": int(p), "T_ms": float(t)} for p, t in zip(Npx, Tpx)
        ],
    }


def fit_tol_law(tol_rows, k, c):
    """(T - c)/N_pix = k (tol/0.03)^-p; returns p and per-scale ratios."""
    use = [r for r in tol_rows if not (r["tol"] < 0.015 and r["steps"] >= 4096)]
    x = np.log(np.array([0.03 / r["tol"] for r in use], float))
    y = np.log(
        np.array(
            [(T_of(r) - c) / float(r["pixels"]) for r in use], float
        )
        / k
    )
    A = np.vstack([x, np.ones(x.size)]).T
    slope, intercept = (float(v) for v in np.linalg.lstsq(A, y, rcond=None)[0])
    resid = A @ np.array([slope, intercept]) - y
    clipped = [r for r in tol_rows if (r["tol"] < 0.015 and r["steps"] >= 4096)]
    return {
        "p": -slope,
        "log_k_ratio": intercept,
        "n_points": int(x.size),
        "rms_log": float(np.sqrt(np.mean(resid**2))),
        "clipped": [
            {
                "scale": r["scale"],
                "tol": r["tol"],
                "T_ms": T_of(r),
                "steps": r["steps"],
            }
            for r in clipped
        ],
    }


def main() -> int:
    with open(SRC, "r", encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    step_rows = [r for r in rows if r["kind"] == "steps"]
    tol_rows = [r for r in rows if r["kind"] == "tol"]

    pix = fit_pixel_law(step_rows)
    tol = fit_tol_law(tol_rows, pix["k_ms_per_px"], pix["c_ms"])

    # (b) step-budget saturation: does the budget above 300 steps buy anything?
    sat = []
    for scale in sorted({r["scale"] for r in step_rows}):
        grp = [r for r in step_rows if r["scale"] == scale]
        grp.sort(key=lambda r: r["steps"])
        base = next((r for r in grp if r["steps"] == 300), None)
        for r in grp:
            if base is None:
                continue
            sat.append(
                {
                    "scale": scale,
                    "steps": r["steps"],
                    "T_ms": T_of(r),
                    "ratio_vs_300": T_of(r) / T_of(base),
                }
            )

    out = {
        "source": os.path.basename(SRC),
        "gpu": rows[0].get("gpu", ""),
        "pixel_law": pix,
        "tol_law": tol,
        "budget_saturation": sat,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print("k = %.4e ms/px   c = %.2f ms   points = %d (kept %d, trimmed=%s)"
          % (pix["k_ms_per_px"], pix["c_ms"], pix["n_points"], pix["n_kept"],
             pix["trimmed"]))
    print("pixel law rms = %.2f ms  (rel rms %.2f%%, max %.2f%%)"
          % (pix["rms_ms"], 100 * pix["rms_rel"], 100 * pix["max_abs_rel"]))
    print("tol exponent p = %.3f  over %d points (log rms %.3f)"
          % (tol["p"], tol["n_points"], tol["rms_log"]))
    print("clipped tol=0.0075 rows:", [c["scale"] for c in tol["clipped"]])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
