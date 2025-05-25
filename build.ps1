# build.ps1 – Automatisierter Build für OtterDream Mandelbrot Renderer

# Stoppe Skript bei jedem Fehler
$ErrorActionPreference = 'Stop'

Write-Host "Starte Build..."

# --- 1) Clean alte Verzeichnisse und Logfile ---
$pathsToClean = @("build", "dist", "mandelbrot_otterdream_log.txt")
foreach ($path in $pathsToClean) {
    if (Test-Path $path) {
        Remove-Item -Recurse -Force $path
        Write-Host "Entfernt: $path"
    }
}

# --- 2) MSVC-Umgebung laden via vswhere ---
# Finde vswhere.exe
$vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe nicht gefunden unter $vswhere. Bitte installieren Sie Visual Studio Installer."
    exit 1
}

# Suche neueste VS-Installation mit C++-Tools
$vsInstallPath = & "$vswhere" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstallPath) {
    Write-Error "Keine VS-Installation mit C++-Tools gefunden."
    exit 1
}

# Pfad zu vcvars64.bat
$vcvars = Join-Path $vsInstallPath 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    Write-Error "vcvars64.bat nicht gefunden unter $vcvars"
    exit 1
}
Write-Host "Verwende VS-Installation: $vsInstallPath"
Write-Host "Lade MSVC-Umgebung: $vcvars"

# Lade Umgebungsvariablen
& cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
Write-Host "MSVC-Umgebung geladen."

# --- 3) vcpkg-Toolchain ermitteln ---
try {
    $vcpkgExe = (Get-Command vcpkg.exe -ErrorAction Stop).Source
    $vcpkgRoot = Split-Path $vcpkgExe -Parent
} catch {
    Write-Error "vcpkg.exe nicht im PATH gefunden."
    exit 1
}
$toolchainFile = Join-Path $vcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
if (-not (Test-Path $toolchainFile)) {
    Write-Error "vcpkg-Toolchain-File nicht gefunden unter $toolchainFile"
    exit 1
}
Write-Host "Verwende vcpkg-Toolchain: $toolchainFile"

# --- 4) Build-Verzeichnis anlegen ---
New-Item -ItemType Directory -Path build | Out-Null

# --- 5) CMake-Konfiguration & Build ---
# Zwingend Ninja als Generator nutzen
$generator = 'Ninja'
Write-Host "Verwende CMake-Generator: $generator"

cmake -S . -B build -G "$generator" `
      -DCMAKE_TOOLCHAIN_FILE="$toolchainFile" `
      -DVCPKG_TARGET_TRIPLET="x64-windows" `
      -DCMAKE_BUILD_TYPE="RelWithDebInfo"

cmake --build build --config RelWithDebInfo --parallel

# --- 6) Dist-Verzeichnis frisch anlegen ---
Remove-Item dist -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path dist | Out-Null

# --- 7) EXE und DLLs nach dist kopieren ---
$exePath = "build\Release\mandelbrot_otterdream.exe"
if (-not (Test-Path $exePath)) { $exePath = "build\mandelbrot_otterdream.exe" }
if (Test-Path $exePath) {
    Copy-Item $exePath -Destination dist -Force
    Write-Host "EXE wurde nach dist\ kopiert."

    # vcpkg-DLLs
    $dlls = @('glfw3.dll','glew32.dll')
    foreach ($dll in $dlls) {
        $found = Get-ChildItem "$vcpkgRoot\installed\x64-windows\bin" -Filter $dll -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            Copy-Item $found.FullName -Destination dist -Force
            Write-Host "$dll kopiert."
        } else {
            Write-Warning "$dll nicht gefunden im vcpkg bin-Verzeichnis."
        }
    }

    # CUDA Runtime DLLs
    try {
        $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
        $cudaBin = Join-Path (Split-Path $nvcc -Parent) 'bin'
        Get-ChildItem -Path $cudaBin -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue |
          ForEach-Object {
            Copy-Item $_.FullName -Destination dist -Force
            Write-Host "$_ kopiert."
          }
    } catch {
        Write-Warning "nvcc.exe nicht gefunden: CUDA-DLLs nicht kopiert."
    }

} else {
    Write-Warning "Kein Executable in build\ gefunden."
}

Write-Host "Build abgeschlossen."
exit 0
