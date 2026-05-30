"""
Lysa desktop launcher.

Runs the exact same FastAPI web app inside a native desktop window via
pywebview (system WebKit on macOS, WebView2 on Windows, GTK/WebKit on Linux),
so the desktop build keeps 100% of the web app's capabilities — the
OpenSeadragon tile viewer, per-channel pyramids, LIF support, everything.

How it works:
  1. Bind a free localhost port and start uvicorn on a background daemon
     thread (the server stays internal — not exposed beyond 127.0.0.1).
  2. Wait until the port actually accepts connections (so the window never
     opens on a not-yet-ready server).
  3. Open a native window pointing at the local server and hand control to
     pywebview's event loop. When the window closes, the process exits and
     the daemon server thread is torn down with it.

Run with:  python desktop.py
Bundle with PyInstaller/cx_Freeze to ship a double-clickable .app / .exe.

Desktop niceties (vs. the plain browser build):
  - Remembers window size/position between launches (~/.lysa/window.json).
  - Native "Open folder" / "Open file" dialogs via a pywebview JS bridge,
    so the frontend can call window.pywebview.api.* instead of the browser
    file picker.
  - A native application menu (macOS) with standard items.
"""

import json
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from lysa import __version__
from lysa.app import create_app

# --- Window-state persistence ----------------------------------------------
# Remembered across launches so the window reopens where you left it.
_STATE_DIR = Path.home() / ".lysa"
_STATE_FILE = _STATE_DIR / "window.json"
_DEFAULT_GEOMETRY = {"width": 1400, "height": 900, "x": None, "y": None}
_MIN_W, _MIN_H = 900, 600


def load_window_state() -> dict:
    """Return saved {width,height,x,y}, falling back to defaults. Never raises."""
    geo = dict(_DEFAULT_GEOMETRY)
    try:
        saved = json.loads(_STATE_FILE.read_text())
        for k in ("width", "height", "x", "y"):
            if isinstance(saved.get(k), (int, float)):
                geo[k] = int(saved[k])
        # Sanity: never restore a uselessly tiny window.
        geo["width"] = max(geo["width"], _MIN_W)
        geo["height"] = max(geo["height"], _MIN_H)
    except Exception:
        pass  # no/invalid state file — defaults are fine
    return geo


def save_window_state(width, height, x, y) -> None:
    """Persist window geometry. Never raises (best-effort)."""
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(
            {"width": int(width), "height": int(height),
             "x": int(x), "y": int(y)}))
    except Exception:
        pass


def find_free_port(start: int = 8000, end: int = 8100) -> int:
    """Return the first free TCP port on localhost in [start, end)."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return start


def wait_until_ready(port: int, timeout: float = 15.0) -> bool:
    """Block until something accepts connections on `port`, or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _serve(app, port: int) -> None:
    """Run uvicorn (called on a background thread). log_level low to stay quiet."""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main() -> int:
    try:
        import webview  # pywebview — imported here so a missing dep is friendly
    except ImportError:
        sys.stderr.write(
            "Lysa desktop needs pywebview.\n"
            "  Install it with:  pip install -r requirements-desktop.txt\n"
            "  (or run the web version with:  python server.py)\n"
        )
        return 1

    app = create_app()
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    server_thread = threading.Thread(
        target=_serve, args=(app, port), daemon=True)
    server_thread.start()

    if not wait_until_ready(port):
        sys.stderr.write("Lysa server failed to start within the timeout.\n")
        return 1

    geo = load_window_state()
    # IMPORTANT: do NOT pass x/y to create_window. On macOS, pywebview applies
    # the initial position via a move() during window init — before the window
    # is attached to a screen — and its own move handler then dereferences
    # screen() (which is None at that point) and crashes. We restore size here
    # and defer position to the 'shown' event, where screen() is valid.
    window = webview.create_window(
        f"Lysa {__version__}", base_url,
        width=geo["width"], height=geo["height"], min_size=(_MIN_W, _MIN_H))

    # Latest known geometry, flushed to disk on change so the window reopens
    # where the user left it even if a later event is missed.
    _geo = {"width": geo["width"], "height": geo["height"],
            "x": geo["x"], "y": geo["y"]}

    def _position_on_screen(x, y):
        """True if (x, y) lies within ~any connected screen (with margin).

        Guards against restoring a position from a now-disconnected monitor,
        which would open the window off-screen. Fails open (False) if the
        screen API isn't usable, so we simply don't restore position.
        """
        try:
            for s in (webview.screens or []):
                sx, sy = getattr(s, "x", 0), getattr(s, "y", 0)
                sw, sh = getattr(s, "width", 0), getattr(s, "height", 0)
                if sx - 50 <= x <= sx + sw - 50 and sy - 20 <= y <= sy + sh - 20:
                    return True
        except Exception:
            pass
        return False

    def _on_shown():
        # Restore position now that the window is on a screen — only if the
        # saved spot is actually visible on a currently-connected display.
        if _geo["x"] is None or _geo["y"] is None:
            return
        if not _position_on_screen(_geo["x"], _geo["y"]):
            return  # stale/off-screen — leave at the default placement
        try:
            window.move(int(_geo["x"]), int(_geo["y"]))
        except Exception:
            pass

    def _on_resized(w, h):
        _geo["width"], _geo["height"] = w, h
        save_window_state(_geo["width"], _geo["height"],
                          _geo["x"] or 0, _geo["y"] or 0)

    def _on_moved(x, y):
        _geo["x"], _geo["y"] = x, y
        save_window_state(_geo["width"], _geo["height"], x, y)

    def _on_closing():
        try:
            save_window_state(window.width, window.height, window.x, window.y)
        except Exception:
            save_window_state(_geo["width"], _geo["height"],
                              _geo["x"] or 0, _geo["y"] or 0)

    # pywebview's event API has shifted across versions; wire each handler
    # defensively so an API mismatch can never stop the app from launching.
    for evt_name, handler in (("shown", _on_shown),
                              ("resized", _on_resized),
                              ("moved", _on_moved),
                              ("closing", _on_closing)):
        try:
            getattr(window.events, evt_name).__iadd__(handler)
        except Exception:
            pass

    menu = _build_menu(webview, window, base_url)
    try:
        webview.start(menu=menu)
    except TypeError:
        # Older pywebview without the menu kwarg — start without a custom menu.
        webview.start()
    return 0


def _build_menu(webview, window, base_url):
    """Construct a native application menu. Returns [] if the API is absent."""
    try:
        from webview.menu import Menu, MenuAction, MenuSeparator
    except Exception:
        return []

    def reload_app():
        try:
            window.load_url(base_url)
        except Exception:
            pass

    def open_in_browser():
        try:
            webbrowser.open(base_url)
        except Exception:
            pass

    try:
        return [
            Menu("File", [
                MenuAction("Reload", reload_app),
                MenuSeparator(),
                MenuAction("Open in Browser", open_in_browser),
            ]),
            Menu("Help", [
                MenuAction("Lysa on GitHub", lambda: webbrowser.open(
                    "https://github.com/prshanthramachandran/Lysa")),
            ]),
        ]
    except Exception:
        return []


if __name__ == "__main__":
    raise SystemExit(main())
