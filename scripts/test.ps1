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
    Write-Host "Installing backend test dependencies..."
    & $venvPython -m pip install -r "backend/requirements-dev.txt"
}

Write-Host "Running backend smoke tests..."
& $venvPython -m pytest backend/tests -q
