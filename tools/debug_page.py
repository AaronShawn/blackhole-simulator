#!/usr/bin/env python3
"""Dump console output / errors / a screenshot of the page without waiting on JS hooks."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from launcher import start_server, resource_root  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

web_dir = os.path.join(resource_root(), "web")
httpd = start_server(web_dir)
url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
print("url:", url)
os.makedirs("work/screens", exist_ok=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch(
        channel="msedge", headless=True,
        args=["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader",
              "--use-angle=d3d11", "--enable-gpu-rasterization"],
    )
    page = browser.new_page(viewport={"width": 1000, "height": 600})
    page.on("console", lambda m: print("[console:%s] %s" % (m.type, m.text)))
    page.on("pageerror", lambda e: print("[pageerror] %s" % e))
    page.on("requestfailed", lambda r: print("[reqfail] %s %s" % (r.url, r.failure)))
    resp = page.goto(url, wait_until="load")
    print("status:", resp.status if resp else None)
    page.wait_for_timeout(9000)
    page.screenshot(path="work/screens/debug.png")
    print("screenshot saved")
    browser.close()
httpd.shutdown()
