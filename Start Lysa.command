#!/bin/bash
# Lysa — double-click launcher for macOS.
#
# Double-click this file in Finder. macOS opens Terminal.app and runs
# the script. Lysa starts on http://localhost:8050 and your default
# browser opens to it after a 3-second delay. Press Ctrl+C in the
# terminal window to stop the server.

# cd to the folder this script lives in (so relative paths work no
# matter where the user opened it from).
cd "$(dirname "$0")"

clear
echo ""
echo "================================================"
echo "   Lysa — Microscopy Image Viewer"
echo "================================================"
echo ""

# --- Python check ---
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not found on this Mac."
    echo ""
    echo "Install it via Homebrew:"
    echo "    brew install python3"
    echo ""
    echo "Or download from https://www.python.org/downloads/"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

# --- Dependency install ---
# Skip the pip step if the main package is already importable. Saves
# ~10-15 seconds on every subsequent launch.
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "First-time setup: installing dependencies..."
    echo "(This only runs once and can take a minute.)"
    echo ""
    python3 -m pip install -r requirements.txt --quiet \
        || python3 -m pip install -r requirements.txt --quiet --break-system-packages
fi

echo "Starting Lysa server on http://localhost:8050"
echo "Your browser will open in a moment."
echo ""
echo "To stop Lysa: press Ctrl+C in this window, then close it."
echo ""

# Open the browser after the server has had a moment to come up.
( sleep 2.5 && open "http://localhost:8050" ) &

# Run the server in foreground; this blocks until Ctrl+C.
python3 server.py

# Reach here when the server exits (Ctrl+C or crash). Keep the window
# open so the user can read any error messages instead of having the
# terminal vanish.
echo ""
echo "Lysa has stopped."
read -p "Press Enter to close this window..."
