#!/usr/bin/env python3
"""Calibrate pywebview/WebView2 window sizing under Windows DPI scaling.

Creates a few throw-away windows with different requested sizes and prints the
resulting client rect (physical px), the work area and the page viewport.
"""
import ctypes
import threading
import time
from ctypes import wintypes


def make_aware():
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except Exception:
        return "unaware"


def work_area():
    r = wintypes.RECT()
    ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(r), 0)
    return r.right - r.left, r.bottom - r.top


def rect_of(title, client=True):
    u = ctypes.windll.user32
    h = u.FindWindowW(None, title)
    if not h:
        return None
    r = wintypes.RECT()
    if client:
        u.GetClientRect(h, ctypes.byref(r))
    else:
        u.GetWindowRect(h, ctypes.byref(r))
    return (r.right - r.left, r.bottom - r.top) if client else (r.right - r.left, r.bottom - r.top)


def run_case(webview, req_w, req_h):
    title = "DPI probe %dx%d" % (req_w, req_h)
    win = webview.create_window(title, html="<body style='background:#123'>probe</body>",
                                width=req_w, height=req_h)

    def probe():
        time.sleep(5)
        js = "JSON.stringify([window.innerWidth, window.innerHeight, window.devicePixelRatio])"
        try:
            page = win.evaluate_js(js)
        except Exception as exc:
            page = "err:%s" % exc
        print("req=%-11s outer=%-13s client=%-13s page=%s"
              % ("%dx%d" % (req_w, req_h), rect_of(title, False), rect_of(title, True), page),
              flush=True)
        try:
            win.destroy()
        except Exception:
            pass

    threading.Thread(target=probe, daemon=True).start()
    webview.start(gui="edgechromium", debug=False, private_mode=True)


def main() -> int:
    print("dpi awareness:", make_aware(), " dpi:", ctypes.windll.user32.GetDpiForSystem(),
          " work area:", work_area(), flush=True)
    import webview
    for w, h in [(1600, 900), (1200, 700), (900, 560)]:
        run_case(webview, w, h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
