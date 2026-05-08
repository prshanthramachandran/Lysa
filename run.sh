#!/bin/bash
# Lysa - Microscopy Image Viewer & Analysis Tool
# Run this script to start the application

echo "================================================"
echo "  Lysa - Microscopy Image Viewer & Analysis"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt -q --break-system-packages 2>/dev/null

echo ""
echo "Starting Lysa server..."
echo "Open your browser to: http://localhost:8050"
echo "Press Ctrl+C to stop."
echo ""

python3 server.py
