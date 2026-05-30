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
"""

import socket
import sys
import threading
import time

import uvicorn

from lysa import __version__
from lysa.app import create_app


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

    server_thread = threading.Thread(
        target=_serve, args=(app, port), daemon=True)
    server_thread.start()

    if not wait_until_ready(port):
        sys.stderr.write("Lysa server failed to start within the timeout.\n")
        return 1

    webview.create_window(
        f"Lysa {__version__}",
        f"http://127.0.0.1:{port}",
        width=1400,
        height=900,
        min_size=(900, 600),
    )
    # Blocks until the window is closed; the daemon server thread exits with us.
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
