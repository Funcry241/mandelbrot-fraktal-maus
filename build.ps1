<#
  MausID: kryptó-42
  # Meta-Kommentar: Updated Build-Script
  # Features:
  # - Clean ENGLISH console output
  # - Auto SSH-agent startup and key loading
  # - Small extra: Build duration timer!
#>

param(
    [string]$Configuration = "RelWithDebInfo"
)

$ErrorActionPreference = 'Stop'
$buildStartTime = Get-Date

Write-Host "=== Build started at $($buildStartTime.ToString("o")) ==="

# 0) Ensure SSH agent is running and key is loaded
try {
    $agentStatus = Get-Service -Name ssh-agent -ErrorAction Stop
    if ($agentStatus.Status -ne 'Running') {
        Write-Host "[SSH] Starting ssh-agent..."
        Start-Service ssh-agent
    } else {
        Write-Host "[SSH] ssh-agent already running."
    }
} catch {
    Write-Error "ssh-agent service not found! Please ensure OpenSSH Client is installed."
    exit 1
}

$sshKeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
$existingKeys = ssh-add -l 2>$null
if (-not $existingKeys -or ($existingKeys -notmatch [Regex]::Escape($sshKeyPath))) {
    Write-Host "[SSH] Adding SSH key..."
    ssh-add $sshKeyPath
} else {
    Write-Host "[SSH] SSH key already added."
}

# 1) Clean old build artifacts
$toClean = @("build","dist","mandelbrot_otterdream_log.txt")
foreach ($p in $toClean) {
    if (Test-Path $p) {
        Remove-Item -Recurse -Force $p
        Write-Host "[CLEAN] Removed: $p"
    }
}

# 2) Check supporter directory
$supporterDir = "ps1Supporter"
if (-not (Test-Path $supporterDir)) {
    Write-Error "[SUPPORT] Directory '$supporterDir' not found. Ensure support scripts exist."
    exit 1
}

# 3) Detect nvcc
try {
    $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
    $cudaBin = Split-Path $nvcc -Parent
    Write-Host "[CUDA] nvcc found: $nvcc"
} catch {
    Write-Error "nvcc.exe not found in PATH. Please install the CUDA Toolkit or update PATH."
    exit 1
}

# 4) Load MSVC Environment via vswhere
$vswhere = Join-Path ${Env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe not found!"
    exit 1
}
$vsInstall = & "$vswhere" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
    Write-Error "No valid Visual Studio installation found!"
    exit 1
}
$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    Write-Error "vcvars64.bat not found!"
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
            Write-Host "[ENV] Set $($matches[1])"
        }
    }
}

# 6) vcpkg Toolchain
try {
    $vcpkg = (Get-Command vcpkg.exe -ErrorAction Stop).Source
} catch {
    Write-Error "vcpkg.exe not found in PATH!"
    exit 1
}
$vcpkgRoot = Split-Path $vcpkg -Parent
$toolchain = Join-Path $vcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
if (-not (Test-Path $toolchain)) {
    Write-Error "vcpkg toolchain not found!"
    exit 1
}
Write-Host "[ENV] vcpkg toolchain: $toolchain"

# 7) Create build & dist directories
New-Item -ItemType Directory -Force -Path build, dist | Out-Null

# 8) CMake configure and build
Write-Host "[BUILD] Configuring with CMake"
$env:Path += ";C:\\ProgramData\\chocolatey\\bin"
cmake -B build -S . -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    "-DCMAKE_BUILD_TYPE=$Configuration" `
    "-DCMAKE_CUDA_COMPILER=$nvcc" `
    "-DCMAKE_CUDA_TOOLKIT_ROOT_DIR=$cudaBin\.." `
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

Write-Host "[BUILD] Building project"
cmake --build build --config $Configuration --parallel

# 9) Copy EXE and DLLs
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

# End and duration output
$buildEndTime = Get-Date
$duration = New-TimeSpan -Start $buildStartTime -End $buildEndTime
Write-Host "`n✅ Build and copy completed successfully! Duration: $($duration.ToString())" -ForegroundColor Green
exit 0
