# Zola Predict - Windows 11 Automated Setup Script

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "       ZOLA PREDICT - Windows 11 Setup Script          " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# 1. Detect Python / uv
$pythonCmd = $null
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "[+] uv package manager detected." -ForegroundColor Green
    $pythonCmd = "uv"
} else {
    Write-Host "[*] Checking standard Python..." -ForegroundColor Yellow
}

# 2. Check or create .env from .env.example
if (-not (Test-Path ".env")) {
    Write-Host "[*] Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "[+] .env template created. Add your FOOTBALL_API_KEY to .env" -ForegroundColor Green
} else {
    Write-Host "[+] .env already exists." -ForegroundColor Green
}

# 3. Virtual environment setup
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[*] Creating Python virtual environment..." -ForegroundColor Yellow
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv .venv
    } else {
        python -m venv .venv
    }
} else {
    Write-Host "[+] Virtual environment exists." -ForegroundColor Green
}

# Install or refresh dependencies on every setup run, including newly added providers.
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt
} else {
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

# 4. Initialize Database
Write-Host "[*] Initializing local SQLite database..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe main.py init-db

# 5. Check environment status
Write-Host "[*] Checking environment configuration status..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe main.py check-env

Write-Host "`nSetup completed! Run '.\.venv\Scripts\python.exe main.py --help' to see available commands." -ForegroundColor Cyan
