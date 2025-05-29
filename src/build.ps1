<#
  MausID: κρυπτό-42
  # OSZILLATIONSSCHUTZ:
  # Änderungen an Build-Parametern (Toolchain, Generator, CMake-Flags etc.)
  # nur zwischen <START OSZILLATIONS­SCHUTZ> und <ENDE OSZILLATIONS­SCHUTZ> durchführen!
#>

if ($false) {
  # <START OSZILLATIONS­SCHUTZ>
  # Generator               = Ninja
  # CMake-Toolchain-File    = vcpkg/scripts/buildsystems/vcpkg.cmake
  # VCPKG_TARGET_TRIPLET    = x64-windows
  # CMAKE_BUILD_TYPE        = RelWithDebInfo
  # CUDA-Compiler           = nvcc.exe (aus PATH)
  # <ENDE OSZILLATIONS­SCHUTZ>
}

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

# 2) MSVC-Umgebung laden via vswhere
$vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { Write-Error "vswhere.exe nicht gefunden!"; exit 1 }
$vsInstall = & "$vswhere" -latest -products * `
               -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
               -property installationPath
if (-not $vsInstall) { Write-Error "Keine VS-Installation gefunden!"; exit 1 }
$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { Write-Error "vcvars64.bat nicht gefunden!"; exit 1 }
Write-Host "[ENV] Lade MSVC-Umgebung von $vcvars"
& cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# 3) vcpkg-Toolchain
try { $vcpkg = (Get-Command vcpkg.exe -ErrorAction Stop).Source } catch {
    Write-Error "vcpkg.exe nicht im PATH!"; exit 1
}
$vcpkgRoot = Split-Path $vcpkg -Parent
$toolchain = Join-Path $vcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
if (-not (Test-Path $toolchain)) { Write-Error "vcpkg-Toolchain fehlt!"; exit 1 }
Write-Host "[ENV] vcpkg-Toolchain: $toolchain"

# 4) Build & Dist anlegen
New-Item -ItemType Directory -Force -Path build, dist | Out-Null

# 5) CMake konfigurieren und bauen
Write-Host "[CMAKE] Konfiguriere ($Configuration)"
cmake -S . -B build -G Ninja `
      -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
      -DVCPKG_TARGET_TRIPLET="x64-windows" `
      -DCMAKE_BUILD_TYPE="$Configuration"
Write-Host "[CMAKE] Baue Projekt"
cmake --build build --config $Configuration --parallel

# 6) EXE und DLLs kopieren
$exe = "build\$Configuration\mandelbrot_otterdream.exe"
if (-not (Test-Path $exe)) { $exe = "build\mandelbrot_otterdream.exe" }
if (Test-Path $exe) {
    Copy-Item $exe -Destination dist -Force; Write-Host "[COPY] EXE → dist"
    foreach ($d in @('glfw3.dll','glew32.dll')) {
        $found = Get-ChildItem "$vcpkgRoot\installed\x64-windows\bin" `
                  -Filter $d -ErrorAction SilentlyContinue |
                 Select-Object -First 1
        if ($found) {
            Copy-Item $found.FullName -Destination dist -Force
            Write-Host "[COPY] $d → dist"
        } else {
            Write-Warning "[MISSING] $d"
        }
    }
    try {
        $nvcc = (Get-Command nvcc.exe).Source
        $cudaBin = Split-Path $nvcc -Parent
        Get-ChildItem $cudaBin -Filter 'cudart64_*.dll' | ForEach-Object {
            Copy-Item $_.FullName -Destination dist -Force
            Write-Host "[COPY] $($_.Name) → dist"
        }
    } catch {
        Write-Warning "[CUDA] Runtime-DLLs nicht kopiert"
    }
} else {
    Write-Warning "[ERROR] Exe nicht gefunden!"
}

# 7) MausDelete
if (Test-Path .\MausDelete.ps1) {
    Write-Host "[MAUSDELETE] Starte MausDelete.ps1..."
    & .\MausDelete.ps1
} else {
    Write-Warning "[MAUSDELETE] Skript fehlt"
}

Write-Host "`n✅ Build und Cleanup abgeschlossen!"
exit 0
