#!/usr/bin/env python3
"""Drive the in-app sub-pixel shadow probe and record what it reports.

The simulator ships a live "measure shadow radius" button that runs
``window.__bh.measure()`` (web/src/main.js).  This script drives exactly that
code path in a headless browser at several viewports, so the numbers quoted in
the paper for the *engineering* layer of experiment V3 are reproducible with a
single command and are not typed in by hand:

    python paper/tools/measure_probe.py

It writes ``paper/tools/probe_measured.json`` which
``paper/tools/make_tables.py`` turns into ``tables/t17_subpixel_probe.tex``.

The physics of the probe is documented in the paper; in short, 24 stratified
sub-pixel samples taken from the same Cranley--Patterson sequence that drives
the progressive accumulator turn the binary capture mask into a per-pixel
coverage fraction, and the two edge pixels of the centre row are inverted to
sub-pixel accuracy.  The coverage quantisation leaves a one-sigma band of
1/(2*sqrt(2)*24) = 0.0147 px on each edge.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from launcher import start_server, resource_root          # noqa: E402
from playwright.sync_api import sync_playwright           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)

# (width, height, label) -- the default window of the desktop build first,
# then the two viewports used elsewhere in the paper.
VIEWPORTS = [
    (1440, 820, "docked"),
    (1600, 900, "default"),
    (1000, 600, "paper"),
]


def main() -> int:
    httpd = start_server(os.path.join(resource_root(), "web"))
    url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    rows = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                channel="msedge", headless=True,
                args=["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
                      "--use-angle=d3d11", "--enable-gpu-rasterization"],
            )
            for (w, h, label) in VIEWPORTS:
                page = browser.new_page(viewport={"width": w, "height": h})
                page.on("pageerror", lambda e: print("[pageerror] %s" % e))
                page.goto(url, wait_until="load")
                page.wait_for_function("() => window.__bh !== undefined", timeout=60000)
                # The probe is a pure shader computation: the HUD only adds
                # cost, never accuracy.
                page.evaluate("document.body.classList.add('nopanel')")
                page.wait_for_timeout(4000)
                res = page.evaluate("() => window.__bh.measure()")
                rows.append({"label": label, "width": w, "height": h, **res})
                print("[probe] %-8s %4dx%-4d measured %.8f  analytic %.8f  "
                      "rel %+.4f%%  sigma %.5f px"
                      % (label, w, h, res["measured"], res["analytic"],
                         res["rel"], res["sigmaPx"]))
                page.close()
            browser.close()
    finally:
        httpd.shutdown()

    out = os.path.join(HERE, "probe_measured.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({
            "note": "Produced by paper/tools/measure_probe.py; "
                    "each row is one window.__bh.measure() call in the shipped app.",
            "samples": rows[0]["samples"],
            "sigmaPx": rows[0]["sigmaPx"],
            "rows": rows,
        }, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("[probe] wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
