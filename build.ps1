# ASCII-only build script for OtterDream Mandelbrot Project
# Purpose: Clean build with vcpkg, CMake, Ninja, CUDA, MSVC
# Note: No Emoji or Unicode, safe for PowerShell/Ninja parsing

param(
    [string]$Configuration = "RelWithDebInfo"
)

$ErrorActionPreference = 'Stop'
Write-Host "`n=== Build started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===`n"

# SSH agent setup
if ((Get-Service ssh-agent -ErrorAction SilentlyContinue).Status -ne 'Running') {
    Start-Service ssh-agent
    Write-Host "[SSH] ssh-agent started."
}
if (-not (ssh-add -l 2>&1) -match "SHA256") {
    $keyPath = "$Env:USERPROFILE\.ssh\id_ed25519"
    if (Test-Path $keyPath) {
        ssh-add $keyPath | Out-Null
        Write-Host "[SSH] Key loaded."
    } else {
        Write-Error "[SSH] Key not found: $keyPath"
        exit 1
    }
} else {
    Write-Host "[SSH] Key already active."
}

# Cleanup old artifacts
foreach ($p in "build", "dist", "mandelbrot_otterdream_log.txt") {
    if (Test-Path $p) {
        Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "[CLEAN] Removed: $p"
    }
}

# Supporter script check
$supporterDir = "ps1Supporter"
if (-not (Test-Path $supporterDir)) {
    Write-Error "[SUPPORT] Missing folder: $supporterDir"
    exit 1
}

# CUDA setup
try {
    $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
    $cudaBin = Split-Path $nvcc -Parent
    Write-Host "[CUDA] nvcc found: $nvcc"
} catch {
    Write-Error "[CUDA] nvcc not found."
    exit 1
}

# MSVC setup
$vswhere = "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = & "$vswhere" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    Write-Error "[MSVC] Visual Studio not found."
    exit 1
}
$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^([\w]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# Load .env
if (Test-Path .env) {
    Write-Host "[ENV] Loading .env..."
    Get-Content .env | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
            Write-Host "[ENV] $($matches[1]) set."
        }
    }
}

# vcpkg setup
try {
    $vcpkg = (Get-Command vcpkg.exe -ErrorAction Stop).Source
    $vcpkgRoot = Split-Path $vcpkg -Parent
    $toolchain = "$vcpkgRoot\scripts\buildsystems\vcpkg.cmake"
    Write-Host "[VCPKG] Toolchain: $toolchain"
} catch {
    Write-Error "[VCPKG] Not found."
    exit 1
}

# Create directories
New-Item -ItemType Directory -Force -Path build, dist | Out-Null

# CMake configure (clean call)
Write-Host "[INFO] CMake version: $(cmake --version | Select-String -Pattern 'cmake version')"
Write-Host "[BUILD] Configuring project..."
$cudaArch = "-DCMAKE_CUDA_ARCHITECTURES=80;86;89;90"
$cmakeArgs = @(
    "-S", ".", "-B", "build", "-G", "Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_CUDA_COMPILER=$nvcc",
    "-DCMAKE_CUDA_TOOLKIT_ROOT_DIR=$($cudaBin)\..",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    $cudaArch
)
cmake @cmakeArgs

# CMake build
Write-Host "[BUILD] Starting build..."
cmake --build build --config $Configuration --parallel

# Copy executable
$exe = "build\$Configuration\mandelbrot_otterdream.exe"
if (-not (Test-Path $exe)) { $exe = "build\mandelbrot_otterdream.exe" }
if (Test-Path $exe) {
    Copy-Item $exe -Destination dist -Force
    Write-Host "[COPY] Executable to dist"
} else {
    Write-Error "[BUILD] Build probably failed. Check output above!"
    exit 1
}

# Copy GLEW/GLFW DLLs
$dllSearchRoots = Get-ChildItem "$PSScriptRoot\vcpkg_installed" -Recurse -Directory | Where-Object { $_.Name -eq "bin" }
foreach ($dll in 'glfw3.dll','glew32.dll') {
    $src = $dllSearchRoots | ForEach-Object {
        Get-ChildItem $_.FullName -Filter $dll -ErrorAction SilentlyContinue
    } | Select-Object -First 1
    if ($src) {
        Copy-Item $src.FullName -Destination dist -Force
        Write-Host "[COPY] $dll to dist"
    } else {
        Write-Error "[COPY] $dll missing!"
        exit 1
    }
}

# Copy CUDA runtime DLLs
$cudaDlls = Get-ChildItem $cudaBin -Filter 'cudart64_*.dll'
if ($cudaDlls) {
    foreach ($dll in $cudaDlls) {
        Copy-Item $dll.FullName -Destination dist -Force
        Write-Host "[CUDA] $($dll.Name) to dist"
    }
} else {
    Write-Error "[CUDA] cudart64_*.dll missing!"
    exit 1
}

# Run supporter scripts
foreach ($script in 'run_build_inner.ps1','MausDelete.ps1','MausGitAutoCommit.ps1') {
    $path = Join-Path $supporterDir $script
    if (Test-Path $path) {
        Write-Host "[SUPPORT] Executing $script..."
        & $path
    } else {
        Write-Error "[SUPPORT] Missing: $script"
        exit 1
    }
}

Write-Host "`nBuild completed successfully."
exit 0
