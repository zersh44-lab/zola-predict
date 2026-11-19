@echo off
REM Zola Predict - One-Click Dashboard Launcher for Windows 11
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Running setup...
    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
)

echo [*] Launching Zola Predict Local Web Dashboard...
".venv\Scripts\python.exe" main.py dashboard
pause
