#!/usr/bin/env python3
"""Flexible render probe: load the simulator, apply JS/preset tweaks, save a PNG.

  python tools/shot.py --preset "俯视盘面" --js "window.__bh.set('diskBright',0.2)" --out x.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from launcher import start_server, resource_root  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset")
    ap.add_argument("--js", action="append", default=[])
    ap.add_argument("--out", default="work/screens/shot.png")
    ap.add_argument("--wait", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--nohud", action="store_true")
    ap.add_argument("--eval", action="append", default=[],
                    help="JS expression to evaluate and print after the wait")
    args = ap.parse_args()

    httpd = start_server(os.path.join(resource_root(), "web"))
    url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="msedge", headless=True,
            args=["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
                  "--use-angle=d3d11", "--enable-gpu-rasterization"],
        )
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.on("console", lambda m: print("[console:%s] %s" % (m.type, m.text))
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: print("[pageerror] %s" % e))
        page.goto(url, wait_until="load")
        page.wait_for_function("() => window.__bh !== undefined", timeout=60000)
        if args.nohud:
            page.evaluate("document.body.classList.add('nopanel')")
        if args.preset:
            page.evaluate("window.__bh.applyPreset(%r)" % args.preset)
        for js in args.js:
            page.evaluate(js)
        page.wait_for_timeout(int(args.wait * 1000))
        page.screenshot(path=args.out)
        print("[info]", page.evaluate("JSON.stringify(window.__bh.info())"))
        for ex in args.eval:
            print("[eval]", ex, "=>", page.evaluate("JSON.stringify(%s)" % ex))
        print("[shot]", args.out)
        browser.close()
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
