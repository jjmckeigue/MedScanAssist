param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating .venv..."
    py -3 -m venv .venv
}

if (-not $SkipInstall) {
    Write-Host "Installing backend dependencies..."
    & $venvPython -m pip install -r "backend/requirements-dev.txt"
}

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..."
    Copy-Item ".env.example" ".env"
}

Write-Host "Starting backend API at http://127.0.0.1:8000 ..."
& $venvPython -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
