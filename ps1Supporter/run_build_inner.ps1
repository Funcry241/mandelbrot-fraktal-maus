# Dateiname: run_build_inner.ps1
# Maus-Kommentar: Baut nur den inneren Teil via CMake & kopiert Executable nach /dist.
# MC|ft=PS|role=innerBuild|env=any|core=no

<#+
  MausID: krypto-42
  Dieses Skript baut das Projekt (ohne Neu-Konfiguration) aus dem bestehenden Build-Ordner
  und kopiert die erzeugte EXE ins /dist-Verzeichnis – bereit zum Ausführen.
#>

param(
    [string]$Configuration = "RelWithDebInfo"
)

$ErrorActionPreference = 'Stop'
Write-Host "`n-- [MAUS-INNERBUILD] startet --`n"

# Pfade setzen
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$buildDir    = Join-Path $projectRoot 'build'
$distDir     = Join-Path $projectRoot 'dist'

Write-Host "[INNER] Projekt-Root:      $projectRoot"
Write-Host "[INNER] Build-Verzeichnis: $buildDir"
Write-Host "[INNER] Dist-Verzeichnis:  $distDir`n"

# 1) CMake-Build starten
Write-Host "[INNER] Starte CMake-Build (Konfiguration: '$Configuration')..."
cmake --build $buildDir --config $Configuration --parallel

# 2) Executable ermitteln und kopieren
$exePath = Join-Path $buildDir "$Configuration\mandelbrot_otterdream.exe"
if (-not (Test-Path $exePath)) {
    $exePath = Join-Path $buildDir "mandelbrot_otterdream.exe"
}

if (Test-Path $exePath) {
    if (-not (Test-Path $distDir)) {
        New-Item -ItemType Directory -Path $distDir | Out-Null
    }
    Copy-Item $exePath -Destination $distDir -Force
    Write-Host "[INNER] Executable kopiert: $exePath -> $distDir"
} else {
    Write-Warning "[INNER] Executable nicht gefunden! Erwartet unter: $exePath"
}

Write-Host "`n-- [MAUS-INNERBUILD] abgeschlossen --`n"
exit 0
