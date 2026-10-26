$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResearchPython = Join-Path $env:USERPROFILE '.local\bin\python3.12.exe'
if (-not (Test-Path -LiteralPath $ResearchPython)) {
    throw "Python 3.12.11 tidak ditemukan: $ResearchPython"
}
& $ResearchPython -c "import sys; print(sys.version); print(sys.executable)"
& $ResearchPython -m pip install --break-system-packages -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Instalasi dependensi gagal.' }
& $ResearchPython -m ipykernel install --user --name seismic-python312 --display-name 'Python 3.12.11 (Seismic Research)'
if ($LASTEXITCODE -ne 0) { throw 'Registrasi kernel gagal.' }
& $ResearchPython (Join-Path $ProjectRoot 'scripts\verify_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'Pemeriksaan environment gagal.' }
