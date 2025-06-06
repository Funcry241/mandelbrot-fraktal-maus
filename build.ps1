<#+
  MausID: κρυπτό-42
  # Meta: Zentrale Fehlerbehandlung + SSH Agent Setup + kleine UX-Extras.
#>

param(
    [string]$Configuration = "RelWithDebInfo"
)

$ErrorActionPreference = 'Stop'

Write-Host "=== 🚀 Starting Build $(Get-Date -Format o) ==="

# 0) SSH-Agent Setup
Write-Host "[SSH] Checking if ssh-agent service is running..."

if (-not (Get-Service ssh-agent -ErrorAction SilentlyContinue)) {
    Write-Error "[SSH] ssh-agent service not found! Please install OpenSSH Client via 'Optional Features'."
    exit 1
}

$sshAgent = Get-Service ssh-agent
if ($sshAgent.Status -ne 'Running') {
    Start-Service ssh-agent
    Write-Host "[SSH] ssh-agent service started."
} else {
    Write-Host "[SSH] ssh-agent already running."
}

$sshKeys = ssh-add -l 2>&1
if ($sshKeys -match "The agent has no identities") {
    Write-Host "[SSH] No SSH keys found. Adding default key..."
    $keyPath = "$Env:USERPROFILE\.ssh\id_ed25519"
    if (Test-Path $keyPath) {
        ssh-add $keyPath | Out-Null
        Write-Host "[SSH] SSH key loaded successfully."
    } else {
        Write-Error "[SSH] SSH key not found at $keyPath"
        exit 1
    }
} else {
    Write-Host "[SSH] SSH key is already loaded."
}

# 1) Clean previous builds
$toClean = @("build","dist","mandelbrot_otterdream_log.txt")
foreach ($p in $toClean) {
    if (Test-Path $p) {
        Remove-Item -Recurse -Force $p
        Write-Host "[CLEAN] Removed: $p"
    }
}

# 2) Validate supporter directory
$supporterDir = "ps1Supporter"
if (-not (Test-Path $supporterDir)) {
    Write-Error "[SUPPORT] Directory '$supporterDir' not found. Please ensure all scripts are inside."
    exit 1
}

# 3) Detect nvcc
try {
    $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
    $cudaBin = Split-Path $nvcc -Parent
    Write-Host "[CUDA] nvcc found: $nvcc"
} catch {
    Write-Error "[CUDA] nvcc.exe not found. Please install CUDA Toolkit or fix your PATH."
    exit 1
}

# 4) MSVC Environment Setup via vswhere
$vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    Write-Error "[MSVC] vswhere.exe not found!"
    exit 1
}
$vsInstall = & "$vswhere" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    Write-Error "[MSVC] No valid Visual Studio installation found!"
    exit 1
}
$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    Write-Error "[MSVC] vcvars64.bat not found!"
    exit 1
}
Write-Host "[ENV] Loading MSVC environment from $vcvars"
& cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# 5) Load .env (optional)
if (Test-Path .env) {
    Write-Host "[ENV] Loading environment variables from .env"
    Get-Content .env | ForEach-Object {
        if ($_ -match '^(.*?)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
            Write-Host "[ENV] $($matches[1]) set"
        }
    }
}

# 6) vcpkg Toolchain
try {
    $vcpkg = (Get-Command vcpkg.exe -ErrorAction Stop).Source
} catch {
    Write-Error "[VCPKG] vcpkg.exe not found in PATH!"
    exit 1
}
$vcpkgRoot = Split-Path $vcpkg -Parent
$toolchain = Join-Path $vcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
if (-not (Test-Path $toolchain)) {
    Write-Error "[VCPKG] Toolchain file not found!"
    exit 1
}
Write-Host "[ENV] vcpkg Toolchain: $toolchain"

# 7) Create build & dist folders
New-Item -ItemType Directory -Force -Path build, dist | Out-Null

# 8) CMake Configure & Build
Write-Host "[BUILD] Configuring with CMake..."
$env:Path += ";C:\ProgramData\chocolatey\bin"
cmake `
    -B build -S . `
    -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$PSScriptRoot/vcpkg/scripts/buildsystems/vcpkg.cmake" `
    "-DCMAKE_BUILD_TYPE=$Configuration" `
    "-DCMAKE_CUDA_COMPILER=$nvcc" `
    "-DCMAKE_CUDA_TOOLKIT_ROOT_DIR=$cudaBin\.." `
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

Write-Host "[BUILD] Building project..."
cmake --build build --config $Configuration --parallel

# 9) Copy binaries
$exe = "build\$Configuration\mandelbrot_otterdream.exe"
if (-not (Test-Path $exe)) {
    $exe = "build\mandelbrot_otterdream.exe"
}
if (Test-Path $exe) {
    Copy-Item $exe -Destination dist -Force
    Write-Host "[COPY] mandelbrot_otterdream.exe → dist" -ForegroundColor Green
} else {
    Write-Host "[COPY] mandelbrot_otterdream.exe missing!" -ForegroundColor Yellow
    exit 1
}

foreach ($d in @('glfw3.dll','glew32.dll')) {
    $found = Get-ChildItem "$vcpkgRoot\installed\x64-windows\bin" -Filter $d -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        Copy-Item $found.FullName -Destination dist -Force
        Write-Host "[COPY] $d → dist" -ForegroundColor Green
    } else {
        Write-Host "[COPY] $d missing!" -ForegroundColor Yellow
        exit 1
    }
}

$cudaOk = $false
foreach ($dll in Get-ChildItem $cudaBin -Filter 'cudart64_*.dll') {
    if ($dll) {
        Copy-Item $dll.FullName -Destination dist -Force
        Write-Host "[CUDA] $($dll.Name) → dist" -ForegroundColor Green
        $cudaOk = $true
    }
}
if (-not $cudaOk) {
    Write-Host "[CUDA] cudart64_*.dll missing!" -ForegroundColor Yellow
    exit 1
}

# 10) Run supporter scripts
$scriptsToRun = @(
    'run_build_inner.ps1',
    'MausDelete.ps1',
    'MausGitAutoCommit.ps1'
)
foreach ($script in $scriptsToRun) {
    $path = Join-Path $supporterDir $script
    if (Test-Path $path) {
        Write-Host "[SUPPORT] Running $script"
        & $path
    } else {
        Write-Warning "[SUPPORT] '$script' not found in $supporterDir"
        exit 1
    }
}

# 🧡 Bonus Extra 🧡
Write-Host "`n Fun Fact: Otters hold hands while sleeping so they don't drift apart. 🦦" -ForegroundColor Cyan

Write-Host "`n Build and copy completed successfully!" -ForegroundColor Green
exit 0
