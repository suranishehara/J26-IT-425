$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements-gpu.txt
    if ($LASTEXITCODE -ne 0) { throw 'GPU package installation failed.' }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
    & '.\.venv\Scripts\python.exe' scripts\verify_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }
} finally {
    Pop-Location
}
