#!/usr/bin/env python3
"""Check the panel collapse / expand behaviour and the centred framing."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from launcher import start_server, resource_root  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def centre(page):
    return page.evaluate(
        """() => {
          const c = document.getElementById('gl');
          const r = c.getBoundingClientRect();
          return {css: [Math.round(r.width), Math.round(r.height)],
                  buf: [c.width, c.height],
                  left: Math.round(r.left), top: Math.round(r.top),
                  shadowPx: window.__bh.info().shadowPx,
                  fps: +window.__bh.info().fps.toFixed(1)};
        }"""
    )


def main() -> int:
    out = "work/screens/ui"
    os.makedirs(out, exist_ok=True)
    httpd = start_server(os.path.join(resource_root(), "web"))
    url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="msedge", headless=True,
            args=["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
                  "--use-angle=d3d11", "--enable-gpu-rasterization"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("pageerror", lambda e: print("[pageerror]", e))
        page.goto(url, wait_until="load")
        page.wait_for_function("() => window.__bh !== undefined", timeout=60000)
        page.evaluate("window.__bh.set('paused', true)")
        page.wait_for_timeout(4000)
        print("panel open :", centre(page))
        page.screenshot(path=os.path.join(out, "01-panel-open.png"))

        page.click("#btn-collapse")
        page.wait_for_timeout(4000)
        print("collapsed  :", centre(page))
        page.screenshot(path=os.path.join(out, "02-collapsed.png"))

        page.click("#btn-expand")
        page.wait_for_timeout(3000)
        print("re-expanded:", centre(page))
        page.screenshot(path=os.path.join(out, "03-expanded.png"))

        # close-up of the scrollbar area
        page.screenshot(path=os.path.join(out, "04-scrollbar.png"),
                        clip={"x": 300, "y": 0, "width": 120, "height": 500})
        browser.close()
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
