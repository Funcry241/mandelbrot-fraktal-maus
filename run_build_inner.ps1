param()

# Setze Arbeitsverzeichnis auf Projekt-Root
$root = $PSScriptRoot
Set-Location $root

$build = Join-Path $root "build"
$dist  = Join-Path $root "dist"

# Clean bestehende Ordner
Remove-Item -Recurse -Force $build, $dist -ErrorAction SilentlyContinue

# Konfiguration & Build
cmake --preset windows-msvc
cmake --build --preset build

# Dist vorbereiten
New-Item -ItemType Directory -Force -Path $dist | Out-Null

# Exe und DLLs kopieren
Copy-Item (Join-Path $build "mandelbrot_otterdream.exe") $dist -ErrorAction Stop
$bin = Join-Path $root "vcpkg/installed/x64-windows/bin"
Copy-Item (Join-Path $bin "glew32.dll") $dist -ErrorAction SilentlyContinue
Copy-Item (Join-Path $bin "glfw3.dll") $dist -ErrorAction SilentlyContinue
Copy-Item "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/cudart64_120.dll" $dist -ErrorAction SilentlyContinue

Write-Host "`n✅ Build abgeschlossen und Dateien kopiert nach $dist" -ForegroundColor Green