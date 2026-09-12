#!/usr/bin/env python3
"""Headless verification: render the simulator with the system Edge (Chromium),
capture console/shader errors, report GPU + frame timing, save screenshots.

Usage:  python tools/verify.py [--shots DIR] [--width 1600] [--height 900]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from launcher import start_server, resource_root  # noqa: E402

from playwright.sync_api import sync_playwright  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="work/screens")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--wait", type=float, default=6.0)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    web_dir = os.path.join(resource_root(), "web")
    httpd = start_server(web_dir)
    url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    os.makedirs(args.shots, exist_ok=True)

    errors, logs = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="msedge",
            headless=not args.headed,
            args=[
                "--ignore-gpu-blocklist",
                "--enable-unsafe-swiftshader",
                "--use-angle=d3d11",
                "--enable-gpu-rasterization",
            ],
        )
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("console", lambda m: logs.append("%s: %s" % (m.type, m.text)))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url, wait_until="load")

        page.wait_for_function("() => window.__bh !== undefined", timeout=60000)
        info0 = page.evaluate("window.__bh.info()")
        print("[webgl] gpu =", info0["gpu"])
        print("[webgl] webgl2 =", info0["webgl2"], " hdr =", info0["hdr"])

        # let the jittered accumulation converge
        page.evaluate("window.__bh.set('paused', true)")
        page.wait_for_timeout(int(args.wait * 1000))
        info = page.evaluate("window.__bh.info()")
        print("[frame] fps=%.2f  accum=%s  shadowPx=%.2f  r0=%.2f M"
              % (info["fps"], info["accCount"], info["shadowPx"], info["dist"]))

        shots = [
            ("01-default", {}),
            ("02-close-ring", {"preset": "近观光子环"}),
            ("03-topdown", {"preset": "俯视盘面"}),
            ("04-lensing-only", {"preset": "纯引力透镜 (无盘)"}),
            ("05-shadow-circle", {"preset": "经典视界 (推荐)", "shadow": 1}),
        ]
        for name, spec in shots:
            if "preset" in spec:
                page.evaluate("window.__bh.applyPreset(%r)" % spec["preset"])
            if spec.get("shadow"):
                page.evaluate("window.__bh.set('shadowCircle', true)")
            page.evaluate("window.__bh.controls.update()")
            page.wait_for_timeout(int(args.wait * 1000))
            path = os.path.join(args.shots, name + ".png")
            page.screenshot(path=path)
            print("[shot]", path)

        # numeric self-check: analytic shadow radius vs. the numerically traced
        # photon-capture boundary, at several observer distances
        page.evaluate("window.__bh.set('shadowCircle', false)")
        page.evaluate("window.__bh.set('autoQuality', 0)")
        page.evaluate("window.__bh.set('renderScale', 1.0)")
        page.evaluate("window.__bh.set('paused', true)")
        rows = page.evaluate(
            """() => {
              const b = window.__bh, out = [];
              for (const d of [8, 12, 20, 26, 40, 80, 150]) {
                b.camera.position.setLength(d);
                b.controls.update();
                const m = b.measure();
                out.push([d, m.measured, m.analytic, m.rel]);
              }
              b.applyPreset('经典视界 (推荐)');
              return out;
            }"""
        )
        print("\n[r0/M]  measured_px   analytic_px   rel.err")
        for d, meas, ana, rel in rows:
            print("  %6.1f  %10.2f  %11.2f  %+7.2f%%" % (d, meas, ana, rel))
        browser.close()

    httpd.shutdown()

    errs = [e for e in errors if e.strip()]
    bad = [l for l in logs
           if ("error" in l.lower() or "GL_" in l or "shader" in l.lower())
           and "favicon" not in l.lower()]
    print("\n=== console (last 20) ===")
    for line in logs[-20:]:
        print(" ", line)
    if bad:
        print("\n!!! suspicious console lines:", len(bad))
        for line in bad[:20]:
            print("  ", line)
    if errs:
        print("\n!!! page errors:")
        for e in errs[:10]:
            print("  ", e)
    print("\nresult:", "FAIL" if (errs or bad) else "OK")
    return 1 if (errs or bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
