#!/bin/bash
# Double-clickable macOS launcher for the Lysa DESKTOP window (pywebview).
# Falls back to creating the venv and installing desktop deps on first run.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "First run — setting up Python environment..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q
fi

# Ensure desktop deps (pywebview) are present.
if ! .venv/bin/python -c "import webview" >/dev/null 2>&1; then
    echo "Installing desktop dependencies (pywebview)..."
    .venv/bin/pip install -r requirements-desktop.txt -q
fi

exec .venv/bin/python desktop.py
