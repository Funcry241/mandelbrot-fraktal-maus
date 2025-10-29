##### Otter: Spinner heartbeat via timer (no PS7 deps), fast-dev knobs (OTTER_FAST): native CUDA arch + optional sccache; install-step skip when fast; no global build-dir change per 2.2.  (Option B: keep single inner tool tag; drop initial duplicate)
##### Schneefuchs: Keep PS5.1-safe: only -f formatting, no pipeline chain ops, no ?.?
##### Maus: Deterministic logs; stable progress even in silent phases; rebase self-heal handled in support script; timings log consolidated (single line); empty-timings safe.
##### Datei: build.ps1

param(
    [ValidateSet('Debug','Release','RelWithDebInfo','MinSizeRel')]
    [string]$Configuration = 'RelWithDebInfo',
    [switch]$Clean
)

# strict & quiet
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'
Set-StrictMode -Version Latest
$PSNativeCommandUseErrorActionPreference = $false

# import support module
$SupportModulePath = Join-Path $PSScriptRoot 'ps1Supporter\Otter.Build.Support.psm1'
if (-not (Test-Path $SupportModulePath)) { throw 'Support module missing: {0}' -f $SupportModulePath }
Import-Module $SupportModulePath -Force

# timings container (script-scope!)
$script:TIMINGS = @()
function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][ScriptBlock]$Do
    )
    Write-Log -Level INFO -Msg ('[STEP] {0,-24} ...' -f $Name)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Do
        $sw.Stop()
        $script:TIMINGS += [PSCustomObject]@{ Name = $Name; Ms = [double]$sw.Elapsed.TotalMilliseconds }
        Write-Log -Level OK -Msg ('{0,-24} done in {1:N1} ms' -f $Name,$sw.Elapsed.TotalMilliseconds)
    } catch {
        $sw.Stop()
        $script:TIMINGS += [PSCustomObject]@{ Name = ($Name + ' (fail)'); Ms = [double]$sw.Elapsed.TotalMilliseconds }
        Write-Log -Level ERR -Msg ('{0,-24} failed after {1:N1} ms' -f $Name,$sw.Elapsed.TotalMilliseconds)
        throw
    }
}

# roots
$ProjectRoot    = Split-Path -Parent $PSCommandPath
$buildDir       = Join-Path $ProjectRoot 'build'
$distDir        = Join-Path $ProjectRoot 'dist'
$localVcpkgRoot = Join-Path $ProjectRoot 'vcpkg'
$toolchain      = Join-Path $localVcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
$installedDir   = Join-Path $ProjectRoot 'vcpkg_installed'
$binaryCache    = Join-Path $ProjectRoot 'vcpkg_cache'

# tool state (script-scope so PSSA sieht Verwendungen in Blocks)
$script:VsInstall  = $null
$script:VcVarsBat  = $null
$script:Nvcc       = $null
$script:CudaBin    = $null
$script:CMakeExe   = $null
$script:CMakeDir   = $null
$script:DumpbinExe = $null
$script:ClExe      = $null
$script:Generator  = 'Ninja'
$script:GenArgs    = @()
$NinjaFound        = $false
$script:NinjaDir   = $null
$Exe               = $null
$MapFile           = $null
$PdbFile           = $null
$script:SccacheExe = $null

# fast-dev flag (without 2.2 global dir change)
$UseFast = $false
if ($Env:OTTER_FAST -and ($Env:OTTER_FAST -eq '1')) { $UseFast = $true }

Write-Host ''
Write-Log -Level INFO -Msg ('=== Build started: {0} ===' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Write-Host ''

Invoke-Step 'Metrics init' { Initialize-BuildMetricsDir -ProjectRoot $ProjectRoot }

Invoke-Step 'Clean policy' {
    if ($Clean) {
        if (Test-Path $buildDir) { Write-Log -Level INFO -Msg ('[CLEAN] Removing: {0}' -f $buildDir); Remove-Item -Recurse -Force $buildDir }
        if (Test-Path $distDir)  { Write-Log -Level INFO -Msg ('[CLEAN] Removing: {0}' -f $distDir);  Remove-Item -Recurse -Force $distDir  }
    } else {
        if (Test-Path $distDir)  { Write-Log -Level INFO -Msg '[CLEAN] Re-creating dist for deterministic output'; Remove-Item -Recurse -Force $distDir }
    }
}

Invoke-Step 'SSH agent' {
    try {
        $svc = Get-Service ssh-agent -ErrorAction SilentlyContinue
        if ($svc -and $svc.Status -ne 'Running') { Start-Service ssh-agent; Write-Log -Level INFO -Msg '[SSH] ssh-agent started.' }
        $hasKey = (ssh-add -l 2>&1) -match 'SHA256'
        if (-not $hasKey) {
            $keyPath = "$Env:USERPROFILE\.ssh\id_ed25519"
            if (Test-Path $keyPath) { ssh-add $keyPath | Out-Null; Write-Log -Level INFO -Msg '[SSH] Key loaded.' }
            else { throw ('[SSH] Key not found: {0}' -f $keyPath) }
        } else { Write-Log -Level INFO -Msg '[SSH] Key already active.' }
    } catch {
        Write-Log -Level WARN -Msg ('[SSH] Skipping ssh-agent setup (not critical). Details: {0}' -f $_.Exception.Message)
    }
}

Invoke-Step 'Supporter check' {
    $supporterDir = Join-Path $ProjectRoot 'ps1Supporter'
    if (-not (Test-Path $supporterDir)) { Write-Log -Level WARN -Msg ('[SUPPORT] Missing folder: {0} (skipping)' -f $supporterDir) }
}

Invoke-Step 'CUDA nvcc' {
    $script:Nvcc = Resolve-Nvcc
    if (-not $script:Nvcc) { throw '[CUDA] nvcc.exe not found. Install CUDA Toolkit 13.0+ or set CUDA_PATH to its root.' }
    $script:CudaBin = Split-Path $script:Nvcc -Parent
    Write-Log -Level INFO -Msg ('[CUDA] nvcc: {0}' -f $script:Nvcc)
    $verOut = & $script:Nvcc --version 2>$null | Out-String
    $m = [regex]::Match($verOut, 'release\s+(\d+)\.(\d+)')
    if ($m.Success) {
        $cudaMajor = [int]$m.Groups[1].Value; $cudaMinor = [int]$m.Groups[2].Value
        Write-Log -Level INFO -Msg ('[CUDA] Detected CUDA release: {0}.{1}' -f $cudaMajor,$cudaMinor)
        if ($cudaMajor -lt 13) { throw ('[CUDA] CUDA 13.0+ required. Detected {0}.{1}' -f $cudaMajor,$cudaMinor) }
    } else { Write-Log -Level WARN -Msg '[CUDA] Version parse skipped.' }
}

Invoke-Step 'MSVC toolchain' {
    $vswhere = "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw '[MSVC] vswhere.exe not found.' }
    $script:VsInstall = & "$vswhere" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $script:VsInstall) { throw '[MSVC] Visual Studio with C++ toolset not found.' }
    $script:VcVarsBat = Join-Path $script:VsInstall 'VC\Auxiliary\Build\vcvars64.bat'
}

Invoke-Step '.env load' { Read-DotEnv (Join-Path $ProjectRoot '.env') }

Invoke-Step 'vcpkg bootstrap' {
    if (-not (Test-Path $localVcpkgRoot)) { throw ('[VCPKG] Project-local vcpkg missing: {0}' -f $localVcpkgRoot) }
    $localVcpkgExe = Join-Path $localVcpkgRoot 'vcpkg.exe'
    if (-not (Test-Path $localVcpkgExe)) {
        $bootstrap = Join-Path $localVcpkgRoot 'bootstrap-vcpkg.bat'
        if (-not (Test-Path $bootstrap)) { throw ('[VCPKG] bootstrap-vcpkg.bat missing in {0}' -f $localVcpkgRoot) }
        Write-Log -Level INFO -Msg '[BOOT] Bootstrapping vcpkg.exe ...'
        & $bootstrap -disableMetrics | Write-Host
    }
    [Environment]::SetEnvironmentVariable('VCPKG_ROOT',                $localVcpkgRoot, 'Process')
    [Environment]::SetEnvironmentVariable('VCPKG_DEFAULT_TRIPLET',     'x64-windows',   'Process')
    [Environment]::SetEnvironmentVariable('VCPKG_DEFAULT_BINARY_CACHE',$binaryCache,    'Process')
    [Environment]::SetEnvironmentVariable('VCPKG_FEATURE_FLAGS',       'manifests,versions,registries,binarycaching,quiet', 'Process')
    [Environment]::SetEnvironmentVariable('VCPKG_KEEP_ENV_VARS',       'VCPKG_FEATURE_FLAGS;VCPKG_DEFAULT_TRIPLET;VCPKG_DEFAULT_BINARY_CACHE;CUDA_PATH;CUDA_HOME;PATH;INCLUDE;LIB', 'Process')
    Write-Log -Level INFO -Msg ('[VCPKG] VCPKG_ROOT    = {0}' -f $localVcpkgRoot)
    Write-Log -Level INFO -Msg ('[VCPKG] Toolchain     = {0}' -f $toolchain)
    Write-Log -Level INFO -Msg ('[VCPKG] Installed dir = {0}' -f $installedDir)
    Write-Log -Level INFO -Msg ('[VCPKG] Binary cache  = {0}' -f $binaryCache)
}

Invoke-Step 'Generator pick' {
    try {
        $ninjaExe = (Get-Command ninja.exe -ErrorAction Stop).Source
        $script:NinjaDir = Split-Path $ninjaExe -Parent
        $NinjaFound = $true
        Write-Log -Level INFO -Msg ('[TOOLS] Ninja: {0}' -f $ninjaExe)
    } catch {
        $ninCandidate = Get-ChildItem -Path (Join-Path $localVcpkgRoot 'downloads\tools\ninja') -Filter ninja.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($ninCandidate) {
            $script:NinjaDir = $ninCandidate.Directory.FullName
            $NinjaFound = $true
            Write-Log -Level INFO -Msg ('[TOOLS] Using Ninja from vcpkg: {0}' -f $ninCandidate.FullName)
        }
    }
    if (-not $NinjaFound) {
        $script:Generator = 'Visual Studio 17 2022'
        $script:GenArgs   = @('-A','x64')
        Write-Log -Level WARN -Msg ('[TOOLS] Ninja not found -> fallback to {0}' -f $script:Generator)
    }
}

Invoke-Step 'Rust runner' { Install-RustRunner -ProjectRoot $ProjectRoot }

Invoke-Step 'Load metrics' {
    Import-BuildMetrics -Config $Configuration -Generator $script:Generator
    $cfgMs = Get-ExpectedMs 'CMake configure'
    $bldMs = Get-ExpectedMs 'CMake build'
    if ($cfgMs) { Write-Log -Level INFO -Msg ('[TIME] Baseline configure ~{0:N1} s' -f ([math]::Round($cfgMs/1000.0,1))) }
    if ($bldMs) { Write-Log -Level INFO -Msg ('[TIME] Baseline build     ~{0:N1} s' -f ([math]::Round($bldMs/1000.0,1))) }
}

$origPath = $env:Path
Invoke-Step 'PATH priming' {
    $gitDir = Resolve-GitDir
    if ($gitDir) { Write-Log -Level INFO -Msg ('[GIT] Found at: {0}' -f $gitDir) } else { Write-Log -Level WARN -Msg '[GIT] Not found yet; will try later.' }

    $script:CMakeExe = Resolve-CMake
    if (-not $script:CMakeExe) { throw '[TOOLS] cmake.exe not found. Install CMake or add to PATH.' }
    $script:CMakeDir = Split-Path $script:CMakeExe -Parent
    Write-Log -Level INFO -Msg ('[TOOLS] CMake: {0}' -f $script:CMakeExe)

    # minimal PATH, then vcvars
    $env:Path = @("$Env:SystemRoot\System32","$Env:SystemRoot","$Env:SystemRoot\System32\Wbem","$Env:SystemRoot\System32\WindowsPowerShell\v1.0","$Env:SystemRoot\System32\OpenSSH",$script:CudaBin) -join ';'
    $vcOk = Import-VcEnvSafe $script:VcVarsBat
    if (-not $vcOk) {
        Write-Log -Level WARN -Msg '[MSVC] Proceeding without imported vcvars.'
        if ($script:Generator -eq 'Ninja') {
            Write-Log -Level WARN -Msg '[GEN] Switching generator to ''Visual Studio 17 2022'' because vcvars is missing.'
            $script:Generator = 'Visual Studio 17 2022'
            $script:GenArgs   = @('-A','x64')
        }
    }

    Add-ToPath $script:CMakeDir
    if ($NinjaFound -and $script:NinjaDir) { Add-ToPath $script:NinjaDir }
    if ($script:CudaBin) { Add-ToPath $script:CudaBin }
    if ($gitDir)         { Add-ToPath $gitDir         }

    $script:DumpbinExe = Resolve-Dumpbin
    if ($script:DumpbinExe) { Write-Log -Level INFO -Msg ('[TOOLS] dumpbin: {0}' -f $script:DumpbinExe) } else { Write-Log -Level WARN -Msg '[TOOLS] dumpbin not found (symbol dumps may be skipped).' }

    $script:ClExe = Resolve-Cl
    if ($script:ClExe) { Write-Log -Level INFO -Msg ('[TOOLS] cl: {0}' -f $script:ClExe) } else { Write-Log -Level WARN -Msg '[TOOLS] cl.exe not found on PATH/VS — CMake autodetect.' }

    # optional sccache for dev speed (only when OTTER_FAST=1)
    try {
        if ($UseFast) {
            $sc = $null
            try { $sc = (Get-Command sccache.exe -ErrorAction Stop).Source } catch {}
            if ($sc) { $script:SccacheExe = $sc; Write-Log -Level INFO -Msg ('[TOOLS] sccache: {0}' -f $script:SccacheExe) }
            else { Write-Log -Level WARN -Msg '[TOOLS] sccache not found; continuing without compiler cache.' }
        }
    } catch {}

    Write-Log -Level INFO -Msg ('[PATH] Ready (len={0})' -f $env:Path.Length)
}

Invoke-Step 'Create dirs' { New-Item -ItemType Directory -Force -Path $buildDir, $distDir, $installedDir, $binaryCache | Out-Null }

Invoke-Step 'Cache guard' {
    $cache = Join-Path $buildDir 'CMakeCache.txt'
    if (Test-Path $cache) {
        $cacheText = Get-Content $cache -Raw

        $cachedSrc  = ([regex]'CMAKE_HOME_DIRECTORY:INTERNAL=([^\r\n]+)').Match($cacheText).Groups[1].Value
        if (-not $cachedSrc) { $cachedSrc = $null }

        $cachedGen  = ([regex]'CMAKE_GENERATOR:INTERNAL=([^\r\n]+)').Match($cacheText).Groups[1].Value
        if (-not $cachedGen) { $cachedGen = $null }

        $cachedNvcc = ([regex]'CMAKE_CUDA_COMPILER:FILEPATH=([^\r\n]+)').Match($cacheText).Groups[1].Value
        if (-not $cachedNvcc) { $cachedNvcc = $null }

        $needNuke = $false

        try {
            $srcNow = (Resolve-Path $ProjectRoot).Path.ToLowerInvariant()
            if ($cachedSrc) {
                $srcOld = $cachedSrc
                try { $srcOld = (Resolve-Path $cachedSrc).Path } catch {}
                if ($srcOld.ToLowerInvariant() -ne $srcNow) {
                    Write-Log -Level WARN -Msg ('[CMAKE] Cache source mismatch: {0} -> {1}' -f $srcOld,$srcNow)
                    $needNuke = $true
                }
            }
        } catch { $needNuke = $true }

        if ($cachedGen  -and ($cachedGen  -ne $script:Generator)) {
            Write-Log -Level WARN -Msg ('[CMAKE] Cache generator mismatch: {0} -> {1}' -f $cachedGen,$script:Generator)
            $needNuke = $true
        }
        if ($cachedNvcc -and ($cachedNvcc -ne $script:Nvcc)) {
            Write-Log -Level WARN -Msg ('[CMAKE] Cache nvcc mismatch: {0} -> {1}' -f $cachedNvcc,$script:Nvcc)
            $needNuke = $true
        }

        if ($needNuke) {
            Write-Log -Level INFO -Msg '[CLEAN] Removing stale build dir due to cache mismatch'
            Remove-Item -Recurse -Force $buildDir
            New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
        }
    }
}

Invoke-Step 'vcpkg preinstall (safe)' {
    $vcpkgExe = Join-Path $localVcpkgRoot 'vcpkg.exe'
    if (-not (Test-Path $vcpkgExe)) { throw '[VCPKG] vcpkg.exe missing after bootstrap.' }
    $ok = Invoke-VcpkgInstallSafe -ProjectRoot $ProjectRoot -Triplet 'x64-windows' -VcpkgExe $vcpkgExe -InstalledDir $installedDir
    if (-not $ok) { throw 'vcpkg preinstall failed' }
}

Write-Rule 'CONFIGURE'
Invoke-Step 'CMake configure' {
    $cmLineObj = (& $script:CMakeExe --version | Select-String -Pattern 'cmake version' | Select-Object -First 1)
    $cmVerText = if ($cmLineObj) { $cmLineObj.ToString() } else { (& $script:CMakeExe --version | Select-Object -First 1) }
    Write-Log -Level INFO -Msg ('[INFO] CMake version: {0}' -f $cmVerText)
    Write-Log -Level INFO -Msg '[BUILD] Configuring project...'

    # CUDA arch policy: Release -> multi-arch; else -> native (or forced native when OTTER_FAST=1)
    $cudaArch = $null
    if ($Configuration -eq 'Release' -and (-not $UseFast)) {
        $cudaArch = '-DCMAKE_CUDA_ARCHITECTURES=80;86;89;90'
    } else {
        $cudaArch = '-DCMAKE_CUDA_ARCHITECTURES=native'
    }

    $tripletArg = '-DVCPKG_TARGET_TRIPLET=x64-windows'
    $mapFlags   = '/INCREMENTAL:NO /OPT:REF /OPT:ICF /MAP:mandelbrot_otterdream.map /MAPINFO:EXPORTS /VERBOSE:REF /VERBOSE:LIB'
    # Correct genex: MultiThreaded[Debug]DLL depending on config
    $msvcRt     = 'MultiThreaded$<$<CONFIG:Debug>:Debug>DLL'

    $cmakeArgs = @('-S', $ProjectRoot, '-B', $buildDir, '-G', $script:Generator) + $script:GenArgs + @(
        ('-DCMAKE_TOOLCHAIN_FILE={0}' -f $toolchain),
        ('-DVCPKG_INSTALLED_DIR={0}' -f $installedDir),
        $tripletArg,
        ('-DCMAKE_BUILD_TYPE={0}' -f $Configuration),
        ('-DCMAKE_CUDA_COMPILER={0}' -f $script:Nvcc),
        '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DCMAKE_CXX_STANDARD=23',
        '-DCMAKE_CUDA_STANDARD=20',
        '-DCMAKE_MESSAGE_LOG_LEVEL=NOTICE',
        ('-DCMAKE_MSVC_RUNTIME_LIBRARY={0}' -f $msvcRt),
        ('-DCMAKE_EXE_LINKER_FLAGS={0}' -f $mapFlags),
        ('-DCMAKE_SHARED_LINKER_FLAGS={0}' -f $mapFlags),
        ('-DCMAKE_MODULE_LINKER_FLAGS={0}' -f $mapFlags),
        $cudaArch
    )

    if ($script:ClExe) {
        $cmakeArgs += @(
            ('-DCMAKE_C_COMPILER={0}'   -f $script:ClExe),
            ('-DCMAKE_CXX_COMPILER={0}' -f $script:ClExe)
        )
    } else {
        Write-Log -Level WARN -Msg '[CMAKE] C/CXX compiler not set explicitly (cl.exe not found) — CMake autodetect.'
    }

    if ($UseFast -and $script:SccacheExe) {
        $cmakeArgs += @(
            ('-DCMAKE_CXX_COMPILER_LAUNCHER={0}'  -f $script:SccacheExe),
            ('-DCMAKE_CUDA_COMPILER_LAUNCHER={0}' -f $script:SccacheExe)
        )
        Write-Log -Level INFO -Msg '[FAST] Enabling sccache for C++ and CUDA.'
    }

    $cfgLog = Join-Path $buildDir 'configure.log'
    $expectedCfg = Get-ExpectedMs 'CMake configure'
    Start-BuildProgress -Activity 'CMake configure' -ExpectedMs $expectedCfg

    # spinner heartbeat while subprocess is silent
    $hbTimer = New-Object System.Timers.Timer
    $hbTimer.Interval = 120
    $hbTimer.AutoReset = $true
    $null = Register-ObjectEvent -InputObject $hbTimer -EventName Elapsed -SourceIdentifier 'OtterHB_Config' -Action { Update-BuildProgress -Percent -1 -Status 'heartbeat' }
    $hbTimer.Start()

    try {
        $onCfgLine = {
            param($line)
            $txt = $line
            if ($txt -match '^\[(INFO|WARN|ERR)\]\s+\[[A-Z]+\]\s*(.*)$') { $txt = $matches[2] }
            $target = -1; $st = 'Parsing output'
            if     ($txt -match '^\s*--\s+The (C|C\+\+) compiler identification') { $target = 5;  $st = 'Detect compiler' }
            elseif ($txt -match '^\s*--\s+(Detecting|Check for working|Performing Test)') { $target = 15; $st = 'Compiler checks' }
            elseif ($txt -match '^\s*--\s+Found ') { $target = 40; $st='Finding packages' }
            elseif ($txt -match '^\s*--\s+Configuring done') { $target = 80; $st = 'Configuring done' }
            elseif ($txt -match '^\s*--\s+Generating done')  { $target = 95; $st = 'Generating done' }
            elseif ($txt -match 'Build files have been written to:') { $target = 100; $st = 'Files written' }
            if ($target -ge 0) { Update-BuildProgress -Percent $target -Status $st } else { Update-BuildProgress -Percent -1 -Status $st }
        }

        $code = Invoke-WithOtterProc `
            -SourceTag 'CMAKE' `
            -Exe $script:CMakeExe `
            -ArgumentList $cmakeArgs `
            -WorkingDirectory $ProjectRoot `
            -LogPath $cfgLog `
            -OnLine $onCfgLine

        if ($code -ne 0) {
            Write-Log -Level ERR -Msg ('[CMAKE] configure exited with code {0}' -f $code)
            Write-Log -Level ERR -Msg ('[CMAKE] ---- last 200 lines of: {0} ----' -f $cfgLog)
            if (Test-Path $cfgLog) { Get-Content $cfgLog -Tail 200 | ForEach-Object { Write-Log -Level INFO -Msg ('[CMAKE] {0}' -f $_) } }
            Write-Log -Level ERR -Msg '[CMAKE] ---- end of log tail ----'
            foreach ($lf in @('CMakeFiles\CMakeError.log','CMakeFiles\CMakeOutput.log')) {
                $p = Join-Path $buildDir $lf
                if (Test-Path $p) {
                    Write-Log -Level ERR -Msg ('[CMAKE] ---- tail of: {0} ----' -f $p)
                    Get-Content $p -Tail 120 | ForEach-Object { Write-Log -Level INFO -Msg ('[CMAKE] {0}' -f $_) }
                } else {
                    Write-Log -Level ERR -Msg ('[CMAKE] Log file not found: {0}' -f $p)
                }
            }
            throw ('CMake configure exited with code {0}' -f $code)
        }
    } finally {
        try { $hbTimer.Stop() } catch {}
        try { Unregister-Event -SourceIdentifier 'OtterHB_Config' -ErrorAction SilentlyContinue } catch {}
        try { $hbTimer.Dispose() } catch {}
        Stop-BuildProgress
    }
}

# Export compile_commands.json (best effort)
Invoke-Step 'Export compile_commands' {
    $cc = Join-Path $buildDir 'compile_commands.json'
    if (-not (Test-Path $cc)) {
        if ($NinjaFound -and (Test-Path $script:NinjaDir)) {
            $ccDir = Join-Path $buildDir 'compdb'
            New-Item -ItemType Directory -Force -Path $ccDir | Out-Null
            $ccArgs = @(
                '-S', $ProjectRoot, '-B', $ccDir, '-G', 'Ninja',
                ('-DCMAKE_TOOLCHAIN_FILE={0}' -f $toolchain),
                ('-DVCPKG_INSTALLED_DIR={0}' -f $installedDir),
                '-DVCPKG_TARGET_TRIPLET=x64-windows',
                ('-DCMAKE_BUILD_TYPE={0}' -f $Configuration),
                ('-DCMAKE_CUDA_COMPILER={0}' -f $script:Nvcc),
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
            )
            $code = Invoke-WithOtterProc -SourceTag 'CMAKE' -Exe $script:CMakeExe -ArgumentList $ccArgs -WorkingDirectory $ProjectRoot -LogPath (Join-Path $ccDir 'cc_configure.log')
            if ($code -eq 0 -and (Test-Path (Join-Path $ccDir 'compile_commands.json'))) {
                Copy-Item (Join-Path $ccDir 'compile_commands.json') $cc -Force
                Write-Log -Level INFO -Msg '[COMPDB] compile_commands.json exported via Ninja.'
                try { Remove-Item $ccDir -Recurse -Force } catch {}
            } else {
                Write-Log -Level WARN -Msg '[COMPDB] Ninja configure failed, writing empty compile_commands.json as fallback.'
                '[]' | Set-Content -Path $cc -Encoding UTF8
            }
        } else {
            Write-Log -Level WARN -Msg '[COMPDB] Ninja not available; writing empty compile_commands.json for VSCode.'
            '[]' | Set-Content -Path $cc -Encoding UTF8
        }
    } else {
        Write-Log -Level INFO -Msg '[COMPDB] compile_commands.json already present.'
    }
}

Invoke-Step 'Patch GLEW targets' {
    $glewTargets = Join-Path $installedDir 'x64-windows\share\glew\glew-targets.cmake'
    if (Test-Path $glewTargets) {
        (Get-Content $glewTargets) | Where-Object { $_ -notmatch 'glew32d\.lib' } | Set-Content $glewTargets
        Write-Log -Level INFO -Msg '[PATCH] Removed invalid reference to glew32d.lib'
    }
}

Write-Rule 'BUILD'
Invoke-Step 'CMake build' {
    Write-Log -Level INFO -Msg '[BUILD] Starting build...'
    $extraBuildArgs = @()
    if ($script:Generator -eq 'Ninja') { $extraBuildArgs = @('--','-d','stats') }
    $buildLog = Join-Path $buildDir 'full_build.log'
    $expectedBuild = Get-ExpectedMs 'CMake build'
    Start-BuildProgress -Activity ("Build via {0}" -f $script:Generator) -ExpectedMs $expectedBuild

    # spinner heartbeat for the build phase
    $hbTimer2 = New-Object System.Timers.Timer
    $hbTimer2.Interval = 120
    $hbTimer2.AutoReset = $true
    $null = Register-ObjectEvent -InputObject $hbTimer2 -EventName Elapsed -SourceIdentifier 'OtterHB_Build' -Action { Update-BuildProgress -Percent -1 -Status 'heartbeat' }
    $hbTimer2.Start()

    try {
        $buildArgs = @('--build', $buildDir, '--config', $Configuration, '--parallel') + $extraBuildArgs
        $onBuildLine = {
            param($line)
            $txt = $line
            if ($txt -match '^\[(INFO|WARN|ERR)\]\s+\[[A-Z]+\]\s*(.*)$') { $txt = $matches[2] }
            $m = [regex]::Match($txt, '^\[\s*(\d+)\s*/\s*(\d+)\s*\]')
            if ($m.Success) {
                $cur = [int]$m.Groups[1].Value
                $tot = [int]$m.Groups[2].Value
                if ($tot -gt 0) {
                    $pct = [int][Math]::Floor(($cur * 100.0) / $tot)
                    Update-BuildProgress -Percent $pct -Status ('{0}/{1}' -f $cur,$tot)
                    return
                }
            }
            if     ($txt -match 'Linking|ninja:|Build files have been written') { Update-BuildProgress -Percent -1 -Status 'Linking / finalizing'; return }
            elseif ($txt -match 'Building|Compiling|\.cu\.obj|\.cpp\.obj|\.obj$') { Update-BuildProgress -Percent -1 -Status 'Compiling'; return }
            Update-BuildProgress -Percent -1 -Status 'Building'
        }
        $code = Invoke-WithOtterProc `
            -SourceTag 'NINJA' `
            -Exe $script:CMakeExe `
            -ArgumentList $buildArgs `
            -WorkingDirectory $ProjectRoot `
            -LogPath $buildLog `
            -OnLine $onBuildLine
        if ($code -ne 0) { throw ('Build exited with code {0}' -f $code) }
    } finally {
        try { $hbTimer2.Stop() } catch {}
        try { Unregister-Event -SourceIdentifier 'OtterHB_Build' -ErrorAction SilentlyContinue } catch {}
        try { $hbTimer2.Dispose() } catch {}
        Stop-BuildProgress
    }
}

Invoke-Step 'Collect products' {
    $Exe = Join-Path $buildDir ("{0}\mandelbrot_otterdream.exe" -f $Configuration)
    if (-not (Test-Path $Exe)) { $Exe = Join-Path $buildDir 'mandelbrot_otterdream.exe' }
    if (-not (Test-Path $Exe)) { throw '[BUILD] Executable not found.' }

    if (-not (Test-Path $distDir)) { New-Item -ItemType Directory -Path $distDir -Force | Out-Null }
    Copy-Item $Exe -Destination $distDir -Force
    Write-Log -Level INFO -Msg '[COPY] Executable to dist'

    $map = Get-ChildItem -Path $buildDir -Recurse -Filter 'mandelbrot_otterdream.map' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($map) { $MapFile = $map; Copy-Item $map.FullName -Destination $distDir -Force; Write-Log -Level INFO -Msg '[COPY] Map to dist' } else { Write-Log -Level WARN -Msg '[MAP] Map file not found (continuing).' }

    $pdb = Get-ChildItem -Path $buildDir -Recurse -Filter 'mandelbrot_otterdream*.pdb' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pdb) { $PdbFile = $pdb; Copy-Item $pdb.FullName -Destination $distDir -Force; Write-Log -Level INFO -Msg '[COPY] PDB to dist' }

    try {
        $exeItem = Get-Item $Exe -ErrorAction Stop
        Write-Log -Level INFO -Msg ('[ARTIFACT] EXE  {0}  ({1})' -f $exeItem.Name, (Format-Size $exeItem.Length))
        try { $sha = (Get-FileHash $exeItem.FullName -Algorithm SHA256).Hash; Write-Log -Level INFO -Msg ('[ARTIFACT] EXE  SHA256 {0}' -f $sha) } catch {}
        if ($PdbFile) {
            $pdbItem = Get-Item $PdbFile.FullName -ErrorAction SilentlyContinue
            if ($pdbItem) { Write-Log -Level INFO -Msg ('[ARTIFACT] PDB  {0}  ({1})' -f $pdbItem.Name, (Format-Size $pdbItem.Length)) }
        }
        if ($MapFile) {
            $mapItem = Get-Item $MapFile.FullName -ErrorAction SilentlyContinue
            if ($mapItem) { Write-Log -Level INFO -Msg ('[ARTIFACT] MAP  {0}  ({1})' -f $mapItem.Name, (Format-Size $mapItem.Length)) }
        }
    } catch {}
}

Invoke-Step 'Runtime DLLs' {
    $dllSearchRoots = Get-ChildItem $installedDir -Recurse -Directory | Where-Object { $_.Name -eq 'bin' }
    foreach ($dll in 'glfw3.dll','glew32.dll') {
        $src = $dllSearchRoots | ForEach-Object { Get-ChildItem $_.FullName -Filter $dll -ErrorAction SilentlyContinue } | Select-Object -First 1
        if ($src) { Copy-Item $src.FullName -Destination $distDir -Force; Write-Log -Level INFO -Msg ('[COPY] {0} to dist' -f $dll) }
        else { throw ('[VERIFY] Missing runtime DLL: {0}' -f $dll) }
    }
    $needsCudart = $false; $dllName = $null
    try {
        $depsText = $null
        if ($script:DumpbinExe) { $depsText = (& $script:DumpbinExe /DEPENDENTS $Exe 2>$null | Out-String) }
        else { try { $db = Get-Command dumpbin.exe -ErrorAction Stop; $depsText = (& $db.Source /DEPENDENTS $Exe 2>$null | Out-String) } catch {} }
        $m = [regex]::Match($depsText, 'cudart64_\d+\.dll', 'IgnoreCase')
        if ($m.Success) { $needsCudart = $true; $dllName = $m.Value }
    } catch {}
    if ($needsCudart -and $dllName) {
        $candidates = @()
        if ($script:CudaBin) { $candidates += (Join-Path $script:CudaBin $dllName) }
        if ($Env:CUDA_PATH)  { $candidates += (Join-Path $Env:CUDA_PATH ("bin\{0}" -f $dllName)) }
        $cand = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($cand) { Copy-Item $cand -Destination $distDir -Force; Write-Log -Level INFO -Msg ('[COPY] {0} to dist' -f (Split-Path $cand -Leaf)) }
        else { Write-Log -Level WARN -Msg ('[CUDA] {0} not found near CUDA bin; relying on system PATH if needed.' -f $dllName) }
    } else { Write-Log -Level INFO -Msg '[CUDA] Static CUDA runtime detected; no DLL copy.' }
}

Invoke-Step 'Supporter hooks' {
    $supporterDir = Join-Path $ProjectRoot 'ps1Supporter'
    if (-not (Test-Path $supporterDir)) {
        Write-Log -Level WARN -Msg ('[SUPPORT] Folder {0} missing - skipping hooks' -f $supporterDir)
    } else {
        $pathBak = $origPath
        $gitDir2 = Resolve-GitDir
        if ($gitDir2) { Write-Log -Level INFO -Msg ('[GIT] Using: {0}' -f $gitDir2) }

        $additions = @()
        if ($script:CMakeDir) { $additions += $script:CMakeDir }
        if ($NinjaFound -and $script:NinjaDir) { $additions += $script:NinjaDir }
        if ($gitDir2) { $additions += $gitDir2 }

        $env:Path = ($additions + $pathBak) -join ';'
        $env:CMAKE_EXE = $script:CMakeExe
        if ($NinjaFound -and $script:NinjaDir) {
            $env:NINJA_EXE = (Join-Path -Path $script:NinjaDir -ChildPath 'ninja.exe')
        } else {
            Remove-Item Env:\NINJA_EXE -ErrorAction SilentlyContinue
        }

        $gitOk = $false
        try { & git --version | Out-Null; $gitOk = $true } catch { Write-Log -Level WARN -Msg '[GIT] git.exe not found; will skip auto-commit.' }

        $scripts = @('MausDelete.ps1','MausGitAutoCommit.ps1')
        foreach ($scriptName in $scripts) {
            if (($scriptName -eq 'MausGitAutoCommit.ps1') -and (-not $gitOk)) { Write-Log -Level WARN -Msg ('[SUPPORT] Skipping {0} (git not available).' -f $scriptName); continue }
            $path = Join-Path $supporterDir $scriptName
            if (-not (Test-Path $path)) { Write-Log -Level WARN -Msg ('[SUPPORT] Missing: {0} (skipping)' -f $scriptName); continue }

            Write-Log -Level INFO -Msg ('[SUPPORT] Executing {0}...' -f $scriptName)
            $eap = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                $global:LASTEXITCODE = 0
                $out = & $path 2>&1
                if ($out) {
                    $out | ForEach-Object {
                        $line = ($_ | Out-String).TrimEnd()
                        if ($line) {
                            $sel = ($line | Select-ToolOutput)
                            if ($sel) { Write-Log -Level INFO -Msg ('[SUPPORT] {0}' -f $sel) }
                        }
                    }
                }
                if ($LASTEXITCODE -ne 0) { Write-Log -Level WARN -Msg ('[SUPPORT] {0} exited with code {1} -- ignored.' -f $scriptName,$LASTEXITCODE); $global:LASTEXITCODE = 0 }
            } catch {
                Write-Log -Level WARN -Msg ('[SUPPORT] {0} raised: {1} -- ignored.' -f $scriptName,$_.Exception.Message)
            } finally { $ErrorActionPreference = $eap }
        }
        $env:Path = $pathBak
    }
}

Invoke-Step 'Verify and install' {
    if ($UseFast) {
        Write-Log -Level INFO -Msg '[INSTALL] Skipped (OTTER_FAST=1)'
    } else {
        $req = 'glew32.dll','glfw3.dll'
        $missing = $req | Where-Object { -not (Test-Path (Join-Path $distDir $_)) }
        if ($missing -and $missing.Count -gt 0) {
            $list = ($missing -join ', ')
            throw ('[VERIFY] Missing runtime DLLs: {0}' -f $list)
        }
        Write-Log -Level INFO -Msg '[INSTALL] Installing to ./dist'
        & $script:CMakeExe --install $buildDir --prefix $distDir
    }
}

# summary / timings (consolidated + empty-safe)
try {
    $buildLog = Join-Path $buildDir 'full_build.log'
    if (Test-Path $buildLog) {
        $warns = (Select-String -Path $buildLog -Pattern 'warning' | Measure-Object).Count
        $errs  = (Select-String -Path $buildLog -Pattern 'error'   | Measure-Object).Count
        Write-Log -Level INFO -Msg ('[BUILD] Diagnostics: {0} warnings, {1} errors' -f $warns,$errs)
    }
} catch {}

Write-Host ''
Write-Log -Level OK -Msg '=== Build completed successfully. ==='
Write-Host ''
Write-SuccessBox 'Build completed successfully'
Write-Host ''

$pad = ($script:TIMINGS | ForEach-Object { $_.Name.Length } | Measure-Object -Maximum).Maximum
if (-not $pad) { $pad = 10 }
$script:TIMINGS | Sort-Object { $_.Ms } -Descending | ForEach-Object {
    $name = $_.Name.PadRight([Math]::Min([Math]::Max($pad,24),48))
    ('{0}  {1,8:N1} ms' -f $name, $_.Ms)
} | ForEach-Object { Write-Log -Level INFO -Msg ('[TIME] {0}' -f $_) }

# console-only total (no dist exports; timings persist in .build_metrics)
try {
    $total = 0.0
    if ($script:TIMINGS -and $script:TIMINGS.Count -gt 0) {
        $sum = ($script:TIMINGS | Measure-Object -Property Ms -Sum).Sum
        if ($null -ne $sum) { $total = [Math]::Round([double]$sum,1) }
    }
    Write-Log -Level INFO -Msg ('[TIME] Total: {0:N1} ms  ({1:N3} s)  [baseline: .build_metrics]' -f $total, ($total/1000.0))
} catch {}

# history append only when timings exist
if ($script:TIMINGS -and $script:TIMINGS.Count -gt 0) {
    Add-RunToHistory -Timings $script:TIMINGS -Config $Configuration -Generator $script:Generator
}
Update-BaselineFromHistory -Config $Configuration -Generator $script:Generator

exit 0
