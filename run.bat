@echo off
REM Zola Predict - Windows 11 CLI Launcher
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Running setup...
    powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
)

".venv\Scripts\python.exe" main.py %*
