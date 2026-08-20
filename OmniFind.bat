@echo off
REM Double-click launcher for the OmniFind desktop app.
REM
REM Uses the venv interpreter directly rather than activating the environment:
REM activation only edits PATH for an interactive shell, and a double-clicked
REM .bat has no shell to inherit it. `start ""` with pythonw.exe keeps the
REM console window from appearing behind the app at all.
cd /d "%~dp0backend"

if not exist ".venv\Scripts\pythonw.exe" (
    echo Could not find .venv in omnifind\backend.
    echo Run the setup steps in backend\README.md first.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" desktop.py
