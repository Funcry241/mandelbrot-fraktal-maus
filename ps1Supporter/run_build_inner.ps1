# Dateiname: run_build_inner.ps1
# Maus-Kommentar: MC|ft=PS|role=innerBuild|env=any|core=no

<#+
  MausID: κρυπτό-42
  (Nur für die Maus: Dieses Skript baut nur den „inneren“ Teil via CMake und kopiert das Ergebnis ins dist-Verzeichnis.)
#>

param(
    [string]$Configuration = "RelWithDebInfo"
)

$ErrorActionPreference = 'Stop'

# 0) Skript- und Projektpfad ermitteln
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$buildDir    = Join-Path $projectRoot 'build'
$distDir     = Join-Path $projectRoot 'dist'

Write-Host "[INNER] Projekt-Root: $projectRoot"
Write-Host "[INNER] Build-Verzeichnis: $buildDir"
Write-Host "[INNER] Dist-Verzeichnis: $distDir"

# 1) Inneren Build Schritt ausführen
Write-Host "[INNER] Starte CMake-Build"
cmake --build $buildDir --config $Configuration --parallel

# 2) Exe kopieren
$exePath = Join-Path $buildDir "$Configuration\mandelbrot_otterdream.exe"
if (-not (Test-Path $exePath)) {
    $exePath = Join-Path $buildDir 'mandelbrot_otterdream.exe'
}
if (Test-Path $exePath) {
    Copy-Item $exePath -Destination $distDir -Force
    Write-Host "[INNER] Kopiere: $exePath → $distDir"
} else {
    Write-Warning "[INNER] Exe nicht gefunden: $exePath"
}

Write-Host "[INNER] Innerer Build-Schritt abgeschlossen."
