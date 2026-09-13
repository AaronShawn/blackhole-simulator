"""Isolate the one anomalous point of the f22 tolerance sweep.

``tols`` sweep of ``tools/make_render_figures.py`` reported
``tol=0.0075, scale=1.00`` at 2326 ms, about 1.7x above the trend of the other
three scales at the same tolerance and about 3.3x above the ``tol=0.015``
point of the same scale (every other factor-of-two in the sweep buys ~1.8x).
This script re-measures that configuration *first*, in a freshly launched
browser, with a longer cooldown, so that a thermal artefact of having run 32
configurations before it can be ruled out; it then walks the two knobs that
could produce a genuine superlinear jump:

  * ``maxSteps``  -- the loop budget.  If the tracer is budget-bound the cost
    is flat above the knee and falls when the budget is cut.
  * ``stepScale`` -- multiplies the adaptive step length *after* the 0.002
    radian floor is applied, so halving the step count halves the cost only if
    the floor, rather than the tolerance, is setting the step length.

Only the minimum of ``window.__bh.cost(n)`` is reported: contention and clock
throttling can only add time, so the minimum is the robust estimator (see the
paper, section on the performance model).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from playwright.sync_api import sync_playwright                   # noqa: E402
from launcher import resource_root, start_server                  # noqa: E402

OUT = os.path.join(HERE, "diag_taucap.json")
# Same launch flags as ``make_render_figures.py``: the frame-rate-cap switches
# are deliberately absent, they make a software rasteriser spin.
CHROME_ARGS = ["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
               "--use-angle=d3d11", "--enable-gpu-rasterization"]
NFRAMES = 9
COOL_MS = 5000
SETTLE_MS = 900


def main() -> int:
    srv = start_server(os.path.join(resource_root(), "web"))
    url = "http://127.0.0.1:%d/index.html" % srv.server_address[1]
    rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True,
                                     args=CHROME_ARGS)
        page = browser.new_page(viewport={"width": 1000, "height": 600})
        page.goto(url, wait_until="load")
        page.wait_for_function("() => window.__bh !== undefined",
                               timeout=120000)
        page.wait_for_function("() => window.__bh.info().fps > 0",
                               timeout=120000)
        page.evaluate("document.body.classList.add('nopanel')")
        page.evaluate("window.__bh.set('autoQuality', false)")
        page.evaluate("window.__bh.set('paused', true)")
        page.evaluate("window.__bh.set('timeRate', 0)")
        page.wait_for_timeout(1000)
        info = json.loads(page.evaluate("JSON.stringify(window.__bh.info())"))
        print("[diag] gpu =", info["gpu"], flush=True)

        def probe(tag, **kw):
            page.wait_for_timeout(COOL_MS)
            for k, v in kw.items():
                page.evaluate("window.__bh.set('%s', %s)" % (k, repr(v)))
            page.wait_for_timeout(SETTLE_MS)
            t = json.loads(page.evaluate("JSON.stringify(window.__bh.cost(%d))"
                                         % NFRAMES))
            inf = json.loads(page.evaluate(
                "JSON.stringify(window.__bh.info())"))
            row = dict(kw)
            row.update(tag=tag, t_min=t["min"], t_median=t["median"],
                       t_max=t["max"], traced=inf["traced"],
                       steps=inf["steps"], tol=inf["tol"], gpu=inf["gpu"])
            rows.append(row)
            print("[diag] %-22s scale=%.2f steps=%4d tol=%.4f stepScale=%.2f"
                  "  min=%8.1f  med=%8.1f" %
                  (tag, kw.get("renderScale"), inf["steps"], inf["tol"],
                   kw.get("stepScale", 1.0), t["min"], t["median"]),
                  flush=True)
            return row

        # 1. the suspected point, measured first on a cold machine
        probe("suspect-cold", renderScale=1.00, maxSteps=4096, tol=0.0075)
        # 2. the same scale at the next tolerance up, for the in-run ratio
        probe("ref-tol015", renderScale=1.00, maxSteps=4096, tol=0.015)
        probe("suspect-repeat", renderScale=1.00, maxSteps=4096, tol=0.0075)
        # 3. budget sweep: does anything change when the loop budget moves?
        for nm in (512, 1024, 2048, 4096):
            probe("budget-%d" % nm, renderScale=1.00, maxSteps=nm,
                  tol=0.0075)
        # 4. step-length sweep: cost should track 1/stepScale if the clamp
        #    floor, and not the tolerance, is what sets the step length.
        for ss in (1.5, 2.0, 3.0):
            probe("stepScale-%.1f" % ss, renderScale=1.00, maxSteps=4096,
                  tol=0.0075, stepScale=ss)
        # 5. and the same two knobs at the neighbouring tolerance, so the
        #    comparison is available without a second run
        for ss in (2.0,):
            probe("tol015-stepScale-%.1f" % ss, renderScale=1.00,
                  maxSteps=4096, tol=0.015, stepScale=ss)
        browser.close()
    srv.shutdown()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"gpu": rows[0]["gpu"] if rows else "",
                   "n_frames": NFRAMES, "cool_ms": COOL_MS,
                   "rows": rows}, fh, ensure_ascii=False, indent=1)

    base = [r for r in rows if r["tag"] == "suspect-repeat"]
    b = float(np.ravel(base[0]["t_min"])[0]) if base else float("nan")
    for r in rows:
        print("  %-22s %8.1f ms   (x%.2f vs repeat)"
              % (r["tag"], r["t_min"], r["t_min"] / b))
    print("[diag] written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
