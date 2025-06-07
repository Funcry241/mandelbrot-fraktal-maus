<#+
  MausID: κρυπτό-42
  # Meta: Optimiertes Build-Script mit SSH, CMake und vcpkg für Mandelbrot-Otterdream.
#>

param(
    [string]$Configuration = "RelWithDebInfo"
)

$ErrorActionPreference = 'Stop'

Write-Host "=== 🚀 Starting Build $(Get-Date -Format o) ==="

# 0) SSH-Agent Setup
if ((Get-Service ssh-agent -ErrorAction SilentlyContinue).Status -ne 'Running') {
    Start-Service ssh-agent
    Write-Host "[SSH] ssh-agent started."
} else {
    Write-Host "[SSH] ssh-agent already running."
}

if (-not (ssh-add -l 2>&1) -match "SHA256") {
    $keyPath = "$Env:USERPROFILE\.ssh\id_ed25519"
    if (Test-Path $keyPath) {
        ssh-add $keyPath | Out-Null
        Write-Host "[SSH] SSH key loaded."
    } else {
        Write-Error "[SSH] No SSH key found at $keyPath."
        exit 1
    }
} else {
    Write-Host "[SSH] SSH key already loaded."
}

# 1) Clean old builds
foreach ($p in "build", "dist", "mandelbrot_otterdream_log.txt") {
    if (Test-Path $p) {
        Remove-Item $p -Recurse -Force
        Write-Host "[CLEAN] Removed: $p"
    }
}

# 2) Supporter check
$supporterDir = "ps1Supporter"
if (-not (Test-Path $supporterDir)) {
    Write-Error "[SUPPORT] Missing supporter folder: $supporterDir"
    exit 1
}

# 3) Find nvcc
try {
    $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
    $cudaBin = Split-Path $nvcc -Parent
    Write-Host "[CUDA] nvcc found: $nvcc"
} catch {
    Write-Error "[CUDA] nvcc not found in PATH."
    exit 1
}

# 4) Find MSVC via vswhere
$vswhere = "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = & "$vswhere" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    Write-Error "[MSVC] No valid Visual Studio installation."
    exit 1
}
$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
Write-Host "[ENV] Loading MSVC env from $vcvars"
& cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^(\w+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# 5) Load optional .env
if (Test-Path .env) {
    Write-Host "[ENV] Loading .env file"
    Get-Content .env | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
            Write-Host "[ENV] $($matches[1]) set."
        }
    }
}

# 6) vcpkg Setup
try {
    $vcpkg = (Get-Command vcpkg.exe -ErrorAction Stop).Source
} catch {
    Write-Error "[VCPKG] vcpkg not found in PATH."
    exit 1
}
$vcpkgRoot = Split-Path $vcpkg -Parent
$toolchain = "$vcpkgRoot\scripts\buildsystems\vcpkg.cmake"
Write-Host "[ENV] vcpkg Toolchain: $toolchain"

# 7) Create build folders
New-Item -ItemType Directory -Force -Path build, dist | Out-Null

# 8) CMake Configure & Build
Write-Host "[BUILD] Configuring CMake..."
$env:Path += ";C:\ProgramData\chocolatey\bin"
cmake -S . -B build -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$PSScriptRoot/vcpkg/scripts/buildsystems/vcpkg.cmake" `
    "-DCMAKE_BUILD_TYPE=$Configuration" `
    "-DCMAKE_CUDA_COMPILER=$nvcc" `
    "-DCMAKE_CUDA_TOOLKIT_ROOT_DIR=$cudaBin\.." `
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

Write-Host "[BUILD] Building project..."
cmake --build build --config $Configuration --parallel

# 9) Copy binaries
$exe = "build\$Configuration\mandelbrot_otterdream.exe"
if (-not (Test-Path $exe)) { $exe = "build\mandelbrot_otterdream.exe" }
if (Test-Path $exe) {
    Copy-Item $exe -Destination dist -Force
    Write-Host "[COPY] mandelbrot_otterdream.exe → dist" -ForegroundColor Green
} else {
    Write-Error "[COPY] Executable missing after build."
    exit 1
}

foreach ($dll in 'glfw3.dll','glew32.dll') {
    $src = Get-ChildItem "$vcpkgRoot\installed\x64-windows\bin" -Filter $dll -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($src) {
        Copy-Item $src.FullName -Destination dist -Force
        Write-Host "[COPY] $dll → dist" -ForegroundColor Green
    } else {
        Write-Error "[COPY] $dll missing!"
        exit 1
    }
}

$cudaDlls = Get-ChildItem $cudaBin -Filter 'cudart64_*.dll'
if ($cudaDlls) {
    foreach ($dll in $cudaDlls) {
        Copy-Item $dll.FullName -Destination dist -Force
        Write-Host "[CUDA] $($dll.Name) → dist" -ForegroundColor Green
    }
} else {
    Write-Error "[CUDA] cudart64_*.dll missing!"
    exit 1
}

# 10) Run supporter scripts
foreach ($script in 'run_build_inner.ps1','MausDelete.ps1','MausGitAutoCommit.ps1') {
    $path = Join-Path $supporterDir $script
    if (Test-Path $path) {
        Write-Host "[SUPPORT] Running $script"
        & $path
    } else {
        Write-Error "[SUPPORT] Missing supporter script: $script"
        exit 1
    }
}

# 🧡 Fun Fact
Write-Host "`nFun Fact: Otters hold hands while sleeping so they don't drift apart. 🦦" -ForegroundColor Cyan
Write-Host "`nBuild completed successfully!" -ForegroundColor Green
exit 0
