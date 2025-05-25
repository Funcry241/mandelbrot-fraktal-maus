<#+
  MausID: κρυπτό-42
  (Nur für die Maus: Dieses Skript baut das Projekt, kopiert die Artefakte inklusive CUDA-Laufzeitbibliotheken und ruft anschließend MausDelete.ps1 auf.)
#>

param(
    [string]$Configuration = "RelWithDebInfo"
)

$ErrorActionPreference = 'Stop'
Write-Host "=== Starte Build $(Get-Date -Format o) ==="

# 1) Clean
$toClean = @("build","dist","mandelbrot_otterdream_log.txt")
foreach ($p in $toClean) {
    if (Test-Path $p) {
        Remove-Item -Recurse -Force $p
        Write-Host "[CLEAN] Entfernt: $p"
    }
}

# 2) Detect nvcc via PATH
try {
    $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
    $cudaBin = Split-Path $nvcc -Parent
    Write-Host "[CUDA] nvcc gefunden: $nvcc"
} catch {
    Write-Error "nvcc.exe nicht im PATH gefunden. Bitte füge das CUDA Toolkit `bin`-Verzeichnis zum PATH hinzu oder installiere das CUDA Toolkit."
    exit 1
}

# 3) MSVC-Umgebung laden via vswhere
$vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe nicht gefunden!"
    exit 1
}
$vsInstall = & "$vswhere" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    Write-Error "Keine VS-Installation gefunden!"
    exit 1
}
$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    Write-Error "vcvars64.bat nicht gefunden!"
    exit 1
}
Write-Host "[ENV] Lade MSVC-Umgebung von $vcvars"
& cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# 4) Load .env (optional)
if (Test-Path .env) {
    Write-Host "[ENV] Lade Umgebungsvariablen aus .env"
    Get-Content .env | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
            Write-Host "[ENV] $($matches[1]) gesetzt"
        }
    }
}

# 5) vcpkg-Toolchain
try {
    $vcpkg = (Get-Command vcpkg.exe -ErrorAction Stop).Source
} catch {
    Write-Error "vcpkg.exe nicht im PATH!"
    exit 1
}
$vcpkgRoot = Split-Path $vcpkg -Parent
$toolchain = Join-Path $vcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
if (-not (Test-Path $toolchain)) {
    Write-Error "vcpkg-Toolchain fehlt!"
    exit 1
}
Write-Host "[ENV] vcpkg-Toolchain: $toolchain"

# 6) Build & Dist anlegen
New-Item -ItemType Directory -Force -Path build, dist | Out-Null

# 7) CMake konfigurieren und bauen
Write-Host "[BUILD] Konfiguriere mit CMake"
cmake `
    -B build -S . `
    -G Ninja `
    -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
    -DCMAKE_BUILD_TYPE="$Configuration" `
    -DCMAKE_CUDA_COMPILER="$nvcc" `
    -DCMAKE_CUDA_TOOLKIT_ROOT_DIR="$cudaBin\.."
Write-Host "[BUILD] Baue Projekt"
cmake --build build --config $Configuration --parallel

# 8) EXE und DLLs kopieren
$exe = "build\$Configuration\mandelbrot_otterdream.exe"
if (-not (Test-Path $exe)) {
    $exe = "build\mandelbrot_otterdream.exe"
}
if (Test-Path $exe) {
    Copy-Item $exe -Destination dist -Force
    Write-Host "[COPY] EXE → dist"

    # Kopiere Abhängige DLLs
    foreach ($d in @('glfw3.dll','glew32.dll')) {
        $found = Get-ChildItem "$vcpkgRoot\installed\x64-windows\bin" -Filter $d -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            Copy-Item $found.FullName -Destination dist -Force
            Write-Host "[COPY] $d → dist"
        } else {
            Write-Warning "[MISSING] $d"
        }
    }

    # Kopiere CUDA Runtime-DLLs
    foreach ($dll in Get-ChildItem $cudaBin -Filter 'cudart64_*.dll') {
        Copy-Item $dll.FullName -Destination dist -Force
        Write-Host "[CUDA] $($dll.Name) → dist"
    }
} else {
    Write-Warning "[ERROR] Exe nicht gefunden!"
}

# 9) MausDelete
if (Test-Path .\MausDelete.ps1) {
    Write-Host "[MAUSDELETE] Starte MausDelete.ps1..."
    & .\MausDelete.ps1
} else {
    Write-Warning "[MAUSDELETE] Skript fehlt"
}

Write-Host "`n✅ Build und Cleanup abgeschlossen!"
exit 0
