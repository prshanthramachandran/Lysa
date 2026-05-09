@echo off
REM Lysa — double-click launcher for Windows.
REM
REM Double-click this file in Explorer. A console window opens, the
REM server starts, and your default browser opens to localhost:8050.
REM Press Ctrl+C in the console to stop.

cd /d "%~dp0"

echo.
echo ================================================
echo    Lysa - Microscopy Image Viewer
echo ================================================
echo.

REM --- Python check ---
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo Install Python 3 from https://www.python.org/downloads/
    echo and make sure to tick "Add to PATH" during install.
    echo.
    pause
    exit /b 1
)

REM --- Dependency install ---
REM Skip pip if FastAPI is already importable.
python -c "import fastapi, uvicorn" >nul 2>nul
if errorlevel 1 (
    echo First-time setup: installing dependencies...
    echo This only runs once and can take a minute.
    echo.
    python -m pip install -r requirements.txt --quiet
)

echo Starting Lysa server on http://localhost:8050
echo Your browser will open in a moment.
echo.
echo To stop Lysa: press Ctrl+C in this window, then close it.
echo.

REM Schedule the browser to open after a couple of seconds.
start "" /min cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8050"

python server.py

echo.
echo Lysa has stopped.
pause
