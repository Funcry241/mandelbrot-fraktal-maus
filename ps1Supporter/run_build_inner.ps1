# Dateiname: run_build_inner.ps1
# Maus-Kommentar: Baut Executable und kopiert sie still nach dist. Keine Geräusche, nur Fehler.
# MC|ft=PS|role=innerBuild|minlog=yes

param([string]$Configuration = "RelWithDebInfo")
$ErrorActionPreference = 'Stop'

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$buildDir    = Join-Path $projectRoot 'build'
$distDir     = Join-Path $projectRoot 'dist'

cmake --build $buildDir --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Error "[INNER] Build failed."
    exit 1
}

$exeRelPath = "$Configuration\mandelbrot_otterdream.exe"
$exeAltPath = "mandelbrot_otterdream.exe"
$exePath = Join-Path $buildDir $exeRelPath
if (-not (Test-Path $exePath)) {
    $exePath = Join-Path $buildDir $exeAltPath
}
if (-not (Test-Path $exePath)) {
    Write-Error "[INNER] Executable not found."
    exit 1
}

if (-not (Test-Path $distDir)) {
    New-Item -ItemType Directory -Path $distDir | Out-Null
}
Copy-Item $exePath -Destination $distDir -Force
exit 0
