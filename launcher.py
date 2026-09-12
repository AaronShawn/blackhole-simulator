#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launcher.py - portable desktop entry point.

Flow:
  1. start a local-only HTTP server on 127.0.0.1 serving the web/ folder
     (http:// instead of file:// so that ES modules and CORS behave).
  2. open a native WebView2 (Edge Chromium) window through pywebview;
     WebView2 uses D3D11/ANGLE hardware acceleration, so an NVIDIA GPU is
     driven directly.
  3. if the WebView2 runtime is missing, fall back to the default browser.

Dependency: pywebview (bundled by PyInstaller, nothing to install for users).
"""

import os
import socket
import sys
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


APP_TITLE = "Black Hole Simulator - Schwarzschild Geodesic Ray Tracer"


def make_dpi_aware() -> str:
    """Declare per-monitor DPI awareness *before* any window exists.

    Without this the process runs DPI-virtualised: pywebview hands WebView2 a
    logical client size while the compositor renders at the physical size, so
    the page gets a viewport larger than the window and the render appears
    zoomed / off-centre on 125-200% displays.
    """
    try:
        import ctypes
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return "per-monitor-v2"
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PROCESS_PER_MONITOR_DPI_AWARE
            return "per-monitor"
        except Exception:
            pass
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            return "system"
        except Exception:
            pass
    except Exception:
        pass
    return "unaware"


def resource_root() -> str:
    """Directory that contains web/ (works for PyInstaller onefile and onedir)."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return base


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        SimpleHTTPRequestHandler.end_headers(self)


def start_server(web_dir: str) -> ThreadingHTTPServer:
    handler = partial(QuietHandler, directory=web_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", free_port()), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def user_data_dir() -> str:
    """Keep the WebView2 profile next to the exe (portable); fall back to temp."""
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(exe_dir, "userdata")
    try:
        os.makedirs(candidate, exist_ok=True)
        probe = os.path.join(candidate, ".writable")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return candidate
    except Exception:
        import tempfile
        return os.path.join(tempfile.gettempdir(), "blackhole_webview2")


def screen_work_area():
    """(width, height) of the primary monitor work area in the process' pixels."""
    try:
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.right - rect.left, rect.bottom - rect.top
    except Exception:
        pass
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


PROBE_JS = """JSON.stringify({
  dpr: window.devicePixelRatio,
  inner: [window.innerWidth, window.innerHeight],
  outer: [window.outerWidth, window.outerHeight],
  doc: [document.documentElement.clientWidth, document.documentElement.clientHeight],
  canvasCss: [document.getElementById('gl').clientWidth, document.getElementById('gl').clientHeight],
  canvasBuf: [document.getElementById('gl').width, document.getElementById('gl').height],
  diag: document.getElementById('diag') ? document.getElementById('diag').innerText : '',
  fps: (window.__bh && window.__bh.info) ? window.__bh.info().fps : null,
  errors: window.__BH_ERR || null
})"""


def push_viewport_once(window, scale: float) -> bool:
    """Tell the page its *real* visible viewport (CSS px).

    pywebview can hand WebView2 a control larger than the window (it ends up
    sized to the monitor work area on high-DPI displays), which makes the page
    lay out for a viewport bigger than what the user can see, so the render is
    cropped / off-centre.  The page uses this value to pin its layout box.
    """
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        hwnd = u.FindWindowW(None, APP_TITLE)
        if not hwnd:
            return False
        rect = wintypes.RECT()
        if not u.GetClientRect(hwnd, ctypes.byref(rect)):
            return False
        w = (rect.right - rect.left) / scale
        h = (rect.bottom - rect.top) / scale
        if w < 80 or h < 80:
            return False
        window.evaluate_js(
            "window.__BH_HOST_VIEWPORT={w:%d,h:%d,dpr:%f};"
            "window.dispatchEvent(new Event('bh-viewport'));" % (round(w), round(h), scale)
        )
        return True
    except Exception:
        return False


def viewport_pusher(window, scale: float) -> None:
    time.sleep(2.5)
    for _ in range(240):          # ~6 minutes, then stop caring
        push_viewport_once(window, scale)
        time.sleep(1.5)


def probe_worker(window) -> None:
    time.sleep(10)
    try:
        print("[probe]", window.evaluate_js(PROBE_JS))
    except Exception as exc:
        print("[probe] failed:", exc)
    try:
        window.destroy()
    except Exception:
        pass


def main() -> int:
    args = {a.lower() for a in sys.argv[1:]}
    dpi = make_dpi_aware()
    root = resource_root()
    web_dir = os.path.join(root, "web")
    if not os.path.isdir(web_dir):
        print("[!] web directory not found:", web_dir, file=sys.stderr)
        return 2

    httpd = start_server(web_dir)
    url = "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    print("[*] local server:", url)
    print("[*] dpi awareness:", dpi)
    # pywebview/WinForms interpret create_window(width, height) as *logical*
    # units and scale them by the monitor DPI itself, so never pre-scale here.
    import ctypes
    try:
        scale = max(1.0, ctypes.windll.user32.GetDpiForSystem() / 96.0)
    except Exception:
        scale = 1.0
    sw, sh = screen_work_area()
    logical_w, logical_h = sw / scale, sh / scale
    win_w = int(min(1500, logical_w * 0.88))
    win_h = int(min(860, logical_h * 0.88))
    print("[*] dpi scale %.2f | work area %dx%d px (%.0fx%.0f logical) | window %dx%d logical"
          % (scale, sw, sh, logical_w, logical_h, win_w, win_h))

    # Let WebView2 ignore the GPU blocklist and prefer the discrete GPU.
    os.environ.setdefault(
        "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
        "--ignore-gpu-blocklist --enable-gpu-rasterization --enable-zero-copy",
    )

    serve_only = "--server-only" in args or "--browser" in args
    fullscreen = "--fullscreen" in args

    if not serve_only:
        try:
            import webview  # type: ignore

            window = webview.create_window(
                APP_TITLE,
                url,
                width=win_w,
                height=win_h,
                min_size=(880, 560),
                background_color="#05070c",
                fullscreen=fullscreen,
                text_select=True,
            )
            try:
                window.move(48, 36)          # keep the frame inside the work area
            except Exception:
                pass
            if "--probe" in args:
                threading.Thread(target=probe_worker, args=(window,), daemon=True).start()
            else:
                threading.Thread(target=viewport_pusher, args=(window, scale), daemon=True).start()
            webview.start(
                gui="edgechromium",
                debug=False,
                private_mode=False,
                storage_path=user_data_dir(),
            )
            return 0
        except Exception as exc:
            print("[!] native window failed (%s); opening the default browser." % exc,
                  file=sys.stderr)

    webbrowser.open(url)
    print("[*] opened in browser. Press Ctrl+C to stop the server.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
