##### Otter: Clean one-tag console ([PS] for script, [RUST] for external tools), native fallback paths, softer spinner color, compact vcpkg/CMake filtering.
##### Schneefuchs: PS5.1-safe; removed extra prefixes; robust null checks; deduped leading tool tags; heartbeat keeps rendering during silences.
##### Maus: Deterministic output & metrics; no “[NINJA] [NINJA]”; progress ETA; baseline learning preserved.
##### Datei: ps1Supporter/Otter.Build.Support.psm1

# -------- module state --------------------------------------------------------
$script:UseColor              = ($Host -and $Host.UI -and $Host.UI.RawUI)
$script:ProgressEnabled       = $false
$script:UseInlineProgress     = $false
$script:IsVSCodeHost          = $false
$script:ProgressActivity      = ''
$script:ProgressStart         = Get-Date
$script:LastPercentShown      = -1
$script:LastStatusText        = ''
$script:SpinChars             = @('|','/','-','\')
$script:SpinIx                = 0
$script:ThrottleMs            = 120
$script:LastRenderAt          = $null
$script:ExpectedMs            = $null
$script:_ProgressPrefBak      = $null
$script:_needsRerenderAfterLog= $false
$script:MetricsDir            = $null
$script:MetricsHistory        = $null
$script:MetricsBaseline       = $null
$script:ExpectedMsByStep      = @{}
$script:OtterProc             = $null   # path to rust runner; null => fallback
$script:_FilterState          = @{ dropPkg=$false; dropAdvice=$false }
$script:InlineColor           = 'DarkCyan'

# -------- console/progress ----------------------------------------------------
function Test-InteractiveConsole {
    try {
        $isVSCode = (($Host.Name -match 'Visual Studio Code') -or ($env:TERM_PROGRAM -eq 'vscode') -or $env:VSCODE_PID -or $env:VSCODE_GIT_IPC_HANDLE)
        if ($env:CI -or $env:GITHUB_ACTIONS) { return $false }
        if (-not $Host -or -not $Host.UI -or -not $Host.UI.RawUI) { return $false }
        if ([Console]::IsOutputRedirected -and -not $isVSCode) { return $false }
        return $true
    } catch { return $false }
}
function Start-BuildProgress([Parameter(Mandatory)][string]$Activity,[double]$ExpectedMs=$null){
    $script:IsVSCodeHost      = (($Host.Name -match 'Visual Studio Code') -or ($env:TERM_PROGRAM -eq 'vscode') -or $env:VSCODE_PID -or $env:VSCODE_GIT_IPC_HANDLE)
    $script:ProgressEnabled   = $true
    $script:UseInlineProgress = $true
    $script:ProgressActivity  = $Activity
    $script:ProgressStart     = Get-Date
    $script:LastPercentShown  = -1
    $script:LastStatusText    = 'Starting...'
    $script:SpinIx            = 0
    $script:LastRenderAt      = (Get-Date).AddYears(-1)   # force first render
    $script:ExpectedMs        = $ExpectedMs
    if ($script:ProgressEnabled) {
        $script:_ProgressPrefBak   = $global:ProgressPreference
        $global:ProgressPreference = 'Continue'
        Show-InlineProgress -Percent -1 -Status $script:LastStatusText
    }
}
function Clear-InlineProgressForLog {
    if ($script:ProgressEnabled -and $script:UseInlineProgress) {
        try { $ws = $Host.UI.RawUI.WindowSize.Width } catch { $ws = 100 }
        $ws = [Math]::Max(30,[int]$ws)
        Write-Host ("`r" + (' ' * ($ws - 1)) + "`r") -NoNewline
        $script:_needsRerenderAfterLog = $true
    }
}
function Test-RenderThrottle([int]$MinDeltaMs=$script:ThrottleMs){
    if (-not $script:LastRenderAt) { $script:LastRenderAt = Get-Date; return $true }
    $delta = ((Get-Date) - $script:LastRenderAt).TotalMilliseconds
    if ($delta -ge $MinDeltaMs) { $script:LastRenderAt = Get-Date; return $true }
    return $false
}
function Show-InlineProgress([int]$Percent,[string]$Status){
    if (-not $script:ProgressEnabled) { return }

    if ($Percent -lt 0 -and $script:ExpectedMs -and $script:ExpectedMs -gt 0) {
        $elapsedMs = ((Get-Date) - $script:ProgressStart).TotalMilliseconds
        $Percent = [Math]::Min(99, [int][Math]::Floor(($elapsedMs * 100.0) / [Math]::Max($script:ExpectedMs,1)))
        if ($script:LastPercentShown -ge 0) { $Percent = [Math]::Max($Percent, $script:LastPercentShown) }
    }

    if (-not (Test-RenderThrottle)) { return }
    if ($Percent -ge 0) { $script:LastPercentShown = [Math]::Max($script:LastPercentShown, [Math]::Min(100,$Percent)) }

    $elapsed = [int]((Get-Date) - $script:ProgressStart).TotalSeconds
    $etaText = ''
    if ($script:LastPercentShown -ge 1 -and $script:LastPercentShown -lt 100) {
        $eta = [int][Math]::Ceiling(($elapsed * (100 - $script:LastPercentShown)) / [Math]::Max($script:LastPercentShown,1))
        if ($eta -ge 0) { $etaText = ("  ~{0}s left" -f $eta) }
    }

    try { $ws = $Host.UI.RawUI.WindowSize.Width } catch { $ws = 100 }
    $barWidth = [Math]::Min(60, [Math]::Max(20, $ws - 24))
    $statusSafe = ($Status -as [string]); if (-not $statusSafe) { $statusSafe = '' }

    if ($Percent -lt 0 -or $script:LastPercentShown -lt 0) {
        $script:SpinIx = ($script:SpinIx + 1) % $script:SpinChars.Count
        $text = ("[PS] [{0}] {1}  {2}s" -f $script:SpinChars[$script:SpinIx], $statusSafe, $elapsed)
        if ($script:UseColor) { Write-Host ("`r" + $text) -NoNewline -ForegroundColor $script:InlineColor } else { Write-Host ("`r" + $text) -NoNewline }
        return
    }

    $pc     = [Math]::Max(0,[Math]::Min(100,$script:LastPercentShown))
    $filled = [int][Math]::Floor(($pc/100) * $barWidth)
    $bar    = ('#' * $filled) + ('.' * ($barWidth - $filled))
    $text   = ("[PS] [{0}] {1,3}%  {2}  {3}s{4}" -f $bar, $pc, $statusSafe, $elapsed, $etaText)
    if ($script:UseColor) { Write-Host ("`r" + $text) -NoNewline -ForegroundColor $script:InlineColor } else { Write-Host ("`r" + $text) -NoNewline }
}
function Update-BuildProgress([int]$Percent,[string]$Status=''){
    if (-not $script:ProgressEnabled) { return }
    if ($Status) { $script:LastStatusText = $Status }
    if ($Percent -ge 0) { Show-InlineProgress -Percent $Percent -Status $script:LastStatusText }
    else { Show-InlineProgress -Percent -1 -Status $script:LastStatusText }
}
function Stop-BuildProgress {
    if ($script:ProgressEnabled) {
        if ($script:UseInlineProgress) { Write-Host "" } else { }
        if ($null -ne $script:_ProgressPrefBak) { $global:ProgressPreference = $script:_ProgressPrefBak }
    }
    $script:ProgressEnabled  = $false
    $script:ProgressActivity = ''
}

# -------- logging -------------------------------------------------------------
function Write-InfoPretty {
    param(
        [string]$Msg,
        [switch]$NoPrefix
    )
    if (-not $script:UseColor) {
        if ($NoPrefix) { Write-Host $Msg }
        else { Write-Host ("[PS] " + $Msg) }
        return
    }
    $s = $Msg
    while ($s -match '^\s*(\[[^\]]+\])\s*(.*)$') {
        $tag  = $matches[1]; $rest = $matches[2]
        if (-not $NoPrefix) { Write-Host "[PS] " -NoNewline -ForegroundColor Gray }
        Write-Host $tag -NoNewline -ForegroundColor Gray
        if ($rest -and ($rest -notmatch '^\[')) {
            Write-Host ' ' -NoNewline
            $regex = '(?<num>(\b\d+(\.\d+)?\s?(ms|s|GB|MB|KB|B|%)\b)|(\b\d{2,5}x\d{2,5}\b))'
            $pos = 0
            foreach ($m in [System.Text.RegularExpressions.Regex]::Matches($rest,$regex)) {
                $preLen = $m.Index - $pos
                if ($preLen -gt 0) { Write-Host ($rest.Substring($pos,$preLen)) -NoNewline -ForegroundColor Gray }
                if ($m.Groups['num'].Success) { Write-Host $m.Value -NoNewline -ForegroundColor White } else { Write-Host $m.Value -NoNewline -ForegroundColor Gray }
                $pos = $m.Index + $m.Length
            }
            if ($pos -lt $rest.Length) { Write-Host ($rest.Substring($pos)) -NoNewline -ForegroundColor Gray }
            Write-Host ''
            return
        }
        Write-Host ' ' -NoNewline
        $s = $rest
    }
    if ($NoPrefix) { Write-Host $s -ForegroundColor Gray }
    else { Write-Host ("[PS] " + $s) -ForegroundColor Gray }
}
function Write-Rule([string]$label=''){
    Clear-InlineProgressForLog
    $width = 60; $line  = '-' * $width
    if ([string]::IsNullOrWhiteSpace($label)) { Write-Host ("[PS] " + $line) -ForegroundColor Gray }
    else {
        $prefix = "-- $label "; $pad = [Math]::Max(0,$width-$prefix.Length)
        Write-Host ("[PS] " + $prefix + ('-' * $pad)) -ForegroundColor Gray
    }
    if ($script:_needsRerenderAfterLog -and $script:UseInlineProgress) { $script:_needsRerenderAfterLog = $false; Show-InlineProgress -Percent $script:LastPercentShown -Status $script:LastStatusText }
}
function Write-SuccessBox([string]$text){
    Clear-InlineProgressForLog
    $t="  $text  "; $top='+'+('-'*$t.Length)+'+'; $mid='|'+$t+'|'
    Write-Host ("[PS] " + $top) -ForegroundColor Green
    Write-Host ("[PS] " + $mid) -ForegroundColor Gray
    Write-Host ("[PS] " + $top) -ForegroundColor Green
    if ($script:_needsRerenderAfterLog -and $script:UseInlineProgress) { $script:_needsRerenderAfterLog = $false; Show-InlineProgress -Percent $script:LastPercentShown -Status $script:LastStatusText }
}
function Write-Log {
    param([ValidateSet('STEP','INFO','OK','WARN','ERR')][string]$Level='INFO',[string]$Msg)
    Clear-InlineProgressForLog
    if (-not $script:UseColor) {
        Write-Host ("[PS] [$Level] " + $Msg)
    } else {
        switch ($Level) {
            'STEP' { Write-Host "[PS] [STEP] " -NoNewline -ForegroundColor Cyan;   Write-Host $Msg -ForegroundColor Gray }
            'INFO' { Write-Host "[PS] [INFO] " -NoNewline -ForegroundColor Gray;   Write-InfoPretty -Msg $Msg -NoPrefix }
            'OK'   { Write-Host "[PS] [OK] "   -NoNewline -ForegroundColor Green;  Write-Host $Msg -ForegroundColor Green }
            'WARN' { Write-Host "[PS] [WARN] " -NoNewline -ForegroundColor Yellow; Write-Host $Msg }
            'ERR'  { Write-Host "[PS] [ERR] "  -NoNewline -ForegroundColor Red;    Write-Host $Msg }
        }
    }
    if ($script:_needsRerenderAfterLog -and $script:UseInlineProgress) { $script:_needsRerenderAfterLog = $false; Show-InlineProgress -Percent $script:LastPercentShown -Status $script:LastStatusText }
}

# -------- utils / resolvers ---------------------------------------------------
function Join-Args([string[]]$ArgList){
    $sb = New-Object System.Text.StringBuilder
    foreach ($a in $ArgList) {
        if ($null -eq $a) { continue }
        $q = $a -replace '"','\"'
        if ($q -match '\s|["]') { $q = '"' + $q + '"' }
        if ($sb.Length -gt 0) { $null = $sb.Append(' ') }
        $null = $sb.Append($q)
    }
    $sb.ToString()
}
function Resolve-Tool([string]$name,[string[]]$candidates){
    try { $p=(Get-Command $name -ErrorAction Stop).Source; if ($p -and (Test-Path $p)) { return $p } } catch {}
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    return $null
}
function Resolve-Nvcc {
    $pf = @()
    foreach ($root in @($Env:CUDA_PATH,$Env:CUDA_HOME)) { if ($root) { $pf += (Join-Path $root 'bin\nvcc.exe') } }
    $bases = @("$Env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA","${Env:ProgramFiles(x86)}\NVIDIA GPU Computing Toolkit\CUDA")
    foreach ($base in $bases) {
        if (Test-Path $base) {
            $pf += (Get-ChildItem -Directory $base | Sort-Object { try {[version]($_.Name.TrimStart('v'))} catch {[version]'0.0'} } -Descending | ForEach-Object { Join-Path $_.FullName 'bin\nvcc.exe' })
        }
    }
    return Resolve-Tool 'nvcc.exe' $pf
}
function Resolve-CMake {
    $pf = @("$Env:ProgramFiles\CMake\bin\cmake.exe")
    if (Test-Path $env:VCPKG_ROOT) {
        $cand = Get-ChildItem -Path (Join-Path $env:VCPKG_ROOT 'downloads\tools') -Recurse -Filter cmake.exe -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cand) { $pf += $cand.FullName }
    }
    return Resolve-Tool 'cmake.exe' $pf
}
function Resolve-Dumpbin {
    if ($env:VSINSTALLDIR) {
        $root = Join-Path $env:VSINSTALLDIR 'VC\Tools\MSVC'
        if (Test-Path $root) {
            $cand = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { $p = Join-Path $_.FullName 'bin\Hostx64\x64\dumpbin.exe'; if (Test-Path $p) { $p } } | Select-Object -First 1
            if ($cand) { return $cand }
        }
    }
    return Resolve-Tool 'dumpbin.exe' @()
}
function Resolve-Cl {
    if ($env:VSINSTALLDIR) {
        $root = Join-Path $env:VSINSTALLDIR 'VC\Tools\MSVC'
        if (Test-Path $root) {
            $cand = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { $p = Join-Path $_.FullName 'bin\Hostx64\x64\cl.exe'; if (Test-Path $p) { $p } } | Select-Object -First 1
            if ($cand) { return $cand }
        }
    }
    return Resolve-Tool 'cl.exe' @()
}
function Resolve-GitDir {
    try { return Split-Path (Get-Command git.exe -ErrorAction Stop).Source -Parent } catch {}
    $cands = @("$Env:ProgramFiles\Git\cmd","$Env:ProgramFiles\Git\bin","$Env:ProgramFiles\Git\usr\bin","$Env:ProgramFiles\Git\mingw64\bin","${Env:ProgramFiles(x86)}\Git\cmd","${Env:ProgramFiles(x86)}\Git\bin")
    foreach ($d in $cands) { if ($d -and (Test-Path $d)) { return $d } }
    foreach ($root in @("$Env:ProgramFiles","${Env:ProgramFiles(x86)}")) {
        if ($root -and (Test-Path $root)) {
            $f = Get-ChildItem -Path $root -Filter git.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($f) { return $f.Directory.FullName }
        }
    }
    return $null
}
function Add-ToPath([string]$dir){
    if (-not [string]::IsNullOrWhiteSpace($dir) -and (Test-Path $dir)) {
        $parts = ($env:Path -split ';') | Where-Object { $_ -ne '' }
        if (-not ($parts | Where-Object { $_ -ieq $dir })) { $env:Path = "$dir;$env:Path" }
    }
}

# -------- env / MSVC ----------------------------------------------------------
function Import-VcEnvSafe([string]$VcVarsBat){
    if ([string]::IsNullOrWhiteSpace($VcVarsBat) -or -not (Test-Path $VcVarsBat)) {
        Write-Log WARN "[MSVC] vcvars not found: $VcVarsBat"
        return $false
    }
    $tmpDir = [System.IO.Path]::GetTempPath()
    $tmpOut = Join-Path $tmpDir ("vcvars_env_{0}.txt" -f ([guid]::NewGuid().ToString('N')))
    $tmpBat = Join-Path $tmpDir ("vcvars_capture_{0}.cmd" -f ([guid]::NewGuid().ToString('N')))
@"
@echo off
setlocal enableextensions
call "$VcVarsBat"
if errorlevel 1 exit /b 255
set > "$tmpOut"
"@ | Set-Content -Path $tmpBat -Encoding ASCII

    $p = Start-Process -FilePath "cmd.exe" -ArgumentList @('/d','/c',"`"$tmpBat`"") -Wait -PassThru -NoNewWindow
    $code = $p.ExitCode
    if ($code -ne 0) { Write-Log WARN "[MSVC] vcvars import failed (code=$code)."; return $false }

    if (Test-Path $tmpOut) {
        foreach ($ln in (Get-Content $tmpOut -ErrorAction SilentlyContinue)) {
            if ($ln -match '^([\w]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
        }
        foreach ($f in @($tmpBat,$tmpOut)) { try { Remove-Item $f -Force -ErrorAction SilentlyContinue } catch {} }
        return $true
    } else {
        Write-Log WARN "[MSVC] vcvars produced no output file."
        foreach ($f in @($tmpBat,$tmpOut)) { try { Remove-Item $f -Force -ErrorAction SilentlyContinue } catch {} }
        return $false
    }
}

# -------- .env ----------------------------------------------------------------
function Read-DotEnv([string]$path){
    if (-not (Test-Path $path)) { return }
    Write-Log INFO "[ENV] Loading .env..."
    Get-Content $path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq '' -or $line.StartsWith('#')) { return }
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
            $k = $matches[1]; $v = $matches[2].Trim()
            if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1, $v.Length-2) }
            if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1, $v.Length-2) }
            [Environment]::SetEnvironmentVariable($k, $v, 'Process')
            Write-Log INFO "[ENV] $k set."
        }
    }
}

# -------- size ----------------------------------------------------------------
function Format-Size($bytes){
    try {
        if ($bytes -ge 1GB) { return ('{0:N2} GB' -f ($bytes/1GB)) }
        if ($bytes -ge 1MB) { return ('{0:N2} MB' -f ($bytes/1MB)) }
        if ($bytes -ge 1KB) { return ('{0:N0} KB' -f $bytes) }
        return ('{0:N0} B' -f $bytes)
    } catch { return "$bytes B" }
}

# -------- metrics (JSON only) -------------------------------------------------
function Initialize-BuildMetricsDir { param([string]$ProjectRoot)
    $script:MetricsDir      = Join-Path $ProjectRoot '.build_metrics'
    $script:MetricsHistory  = Join-Path $script:MetricsDir 'history.jsonl'
    $script:MetricsBaseline = Join-Path $script:MetricsDir 'baseline.json'
    if (-not (Test-Path $script:MetricsDir)) { New-Item -ItemType Directory -Force -Path $script:MetricsDir | Out-Null }
}
function Get-GitHeadShort { try { $res = (& git rev-parse --short=10 HEAD 2>$null); if ($res) { return $res.Trim() } } catch {}; return $null }
function Import-BuildMetrics { param([string]$Config,[string]$Generator)
    $script:ExpectedMsByStep = @{}
    $bl = $null
    if (Test-Path $script:MetricsBaseline) {
        try { $tmp = Get-Content $script:MetricsBaseline -Raw | ConvertFrom-Json -Depth 6 } catch { $tmp = $null }
        if ($tmp -and ($tmp.PSObject.Properties.Count -gt 0)) { $bl = $tmp }
    }
    $rebuilt = $false
    if (-not $bl -and (Test-Path $script:MetricsHistory)) {
        $rows = @()
        try {
            foreach ($ln in (Get-Content $script:MetricsHistory -Tail 500)) {
                try { $o = $ln | ConvertFrom-Json -Depth 6 } catch { continue }
                if ($o.Configuration -eq $Config -and $o.Generator -eq $Generator) { $rows += $o }
            }
        } catch {}
        $calc = @{}
        foreach ($stepName in @('CMake configure','CMake build')) {
            $vals = @(); foreach ($r in $rows) { foreach ($s in $r.Steps) { if ($s.Name -eq $stepName -and $s.Ms -gt 0) { $vals += [double]$s.Ms } } }
            if ($vals.Count -gt 0) {
                $vals = $vals | Sort-Object
                $mid=[int]([math]::Floor(($vals.Count-1)/2)); $median = if ($vals.Count%2 -eq 1) { $vals[$mid] } else { ([double]($vals[$mid]+$vals[$mid+1])/2.0) }
                $calc["$Config|$Generator|$stepName"] = @{ Ms = [double]$median; Count = $vals.Count }
            }
        }
        if ($calc.Keys.Count -gt 0) {
            try {
                ($calc | ConvertTo-Json -Depth 6) | Set-Content -Path $script:MetricsBaseline -Encoding UTF8
                $bl = Get-Content $script:MetricsBaseline -Raw | ConvertFrom-Json -Depth 6
                $rebuilt = $true
                Write-Log INFO "[TIME] Baseline rebuilt from history."
            } catch { Write-Log WARN "[TIME] Failed to write rebuilt baseline: $($_.Exception.Message)" }
        }
    }
    if ($bl) {
        foreach ($prop in $bl.PSObject.Properties) {
            if ($prop.Name -match '^(?<cfg>[^|]+)\|(?<gen>[^|]+)\|(?<step>.+)$') {
                if ($matches['cfg'] -eq $Config -and $matches['gen'] -eq $Generator) {
                    try { $ms = [double]$prop.Value.Ms; if ($ms -gt 0) { $script:ExpectedMsByStep[$matches['step']] = $ms } } catch {}
                }
            }
        }
    } elseif (-not $rebuilt) {
        Write-Log WARN "[TIME] No baseline available yet (will learn from this run)."
    }
}
function Get-ExpectedMs([Parameter(Mandatory)][string]$Step){
    if ($script:ExpectedMsByStep.ContainsKey($Step)) { return [double]$script:ExpectedMsByStep[$Step] }
    return $null
}
function Add-RunToHistory {
    param(
        [Parameter(Mandatory)][object[]]$Timings,
        [Parameter(Mandatory)][string]$Config,
        [Parameter(Mandatory)][string]$Generator
    )
    try {
        if (-not $Timings -or $Timings.Count -eq 0) { return }
        $steps = @()
        foreach ($t in $Timings) { if ($t -and $t.Ms -ge 0) { $steps += [PSCustomObject]@{ Name=$t.Name; Ms=[double]$t.Ms } } }
        if ($steps.Count -eq 0) { return }
        $sumMs = 0.0; foreach ($s in $steps) { $sumMs += [double]$s.Ms }
        $rec = [PSCustomObject]@{
            Date          = Get-Date
            Configuration = $Config
            Generator     = $Generator
            Commit        = Get-GitHeadShort
            TotalMs       = [Math]::Round($sumMs,1)
            Steps         = $steps
        }
        if (-not (Test-Path $script:MetricsDir)) { New-Item -ItemType Directory -Force -Path $script:MetricsDir | Out-Null }
        $json = $rec | ConvertTo-Json -Depth 6 -Compress
        Add-Content -Path $script:MetricsHistory -Value $json -Encoding UTF8
    } catch { Write-Log WARN "[TIME] Failed to append history: $($_.Exception.Message)" }
}
function Update-BaselineFromHistory([string]$Config,[string]$Generator){
    try {
        $lines = @(); try { $lines = Get-Content $script:MetricsHistory -Tail 200 } catch {}
        $rows = @()
        foreach ($ln in $lines) { try { $o = $ln | ConvertFrom-Json -Depth 6 } catch { continue }; if ($o.Configuration -eq $Config -and $o.Generator -eq $Generator) { $rows += $o } }
        $bl = @{}
        foreach ($stepName in @('CMake configure','CMake build')) {
            $vals = @(); foreach ($r in $rows) { foreach ($s in $r.Steps) { if ($s.Name -eq $stepName -and $s.Ms -gt 0) { $vals += [double]$s.Ms } } }
            if ($vals.Count -gt 0) {
                $vals = $vals | Sort-Object
                $mid=[int]([math]::Floor(($vals.Count-1)/2)); $median = if ($vals.Count%2 -eq 1) { $vals[$mid] } else { ([double]($vals[$mid]+$vals[$mid+1])/2.0) }
                $bl["$Config|$Generator|$stepName"] = @{ Ms = [double]$median; Count = $vals.Count }
            }
        }
        ($bl | ConvertTo-Json -Depth 6) | Set-Content -Path $script:MetricsBaseline -Encoding UTF8
    } catch { Write-Log WARN "[TIME] Failed to write baseline: $($_.Exception.Message)" }
}

# -------- AutoRun guard for cmd/vcpkg ----------------------------------------
$script:_AutoRunWasTouched = $false
$script:_AutoRunBackup     = $null
function Disable-CmdAutoRun {
    $script:_AutoRunWasTouched = $false
    $script:_AutoRunBackup = @{ HKCU = $null; HKLM = $null }
    $hkcu = 'HKCU:\Software\Microsoft\Command Processor'
    $hklm = 'HKLM:\Software\Microsoft\Command Processor'
    try {
        $script:_AutoRunBackup.HKCU = (Get-ItemProperty -Path $hkcu -Name AutoRun -ErrorAction SilentlyContinue).AutoRun
        $script:_AutoRunBackup.HKLM = (Get-ItemProperty -Path $hklm -Name AutoRun -ErrorAction SilentlyContinue).AutoRun
        if ($script:_AutoRunBackup.HKCU) { Remove-ItemProperty -Path $hkcu -Name AutoRun -ErrorAction SilentlyContinue; $script:_AutoRunWasTouched = $true }
        if ($script:_AutoRunBackup.HKLM) { Remove-ItemProperty -Path $hklm -Name AutoRun -ErrorAction SilentlyContinue; $script:_AutoRunWasTouched = $true }
        if ($script:_AutoRunWasTouched) { Write-Log INFO "[CMD] AutoRun temporarily disabled" }
    } catch {
        Write-Log WARN "[CMD] Could not adjust AutoRun: $($_.Exception.Message)"
    }
}
function Restore-CmdAutoRun {
    if (-not $script:_AutoRunWasTouched) { return }
    $hkcu = 'HKCU:\Software\Microsoft\Command Processor'
    $hklm = 'HKLM:\Software\Microsoft\Command Processor'
    try {
        if ($script:_AutoRunBackup.HKCU) { New-Item -Path $hkcu -Force | Out-Null; Set-ItemProperty -Path $hkcu -Name AutoRun -Value $script:_AutoRunBackup.HKCU }
        if ($script:_AutoRunBackup.HKLM) { New-Item -Path $hklm -Force | Out-Null; Set-ItemProperty -Path $hklm -Name AutoRun -Value $script:_AutoRunBackup.HKLM }
        Write-Log INFO "[CMD] AutoRun restored"
    } catch {
        Write-Log WARN "[CMD] Failed to restore AutoRun: $($_.Exception.Message)"
    } finally {
        $script:_AutoRunWasTouched = $false
        $script:_AutoRunBackup     = $null
    }
}

# -------- vcpkg (via runner or fallback) -------------------------------------
function Invoke-VcpkgInstallSafe {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$ProjectRoot,
        [Parameter(Mandatory)][string]$Triplet,
        [Parameter(Mandatory)][string]$VcpkgExe,
        [Parameter(Mandatory)][string]$InstalledDir
    )
    $log   = Join-Path (Join-Path $ProjectRoot 'build') 'vcpkg-manifest-install.log'
    $tmpCmd= Join-Path ([System.IO.Path]::GetTempPath()) ("vcpkg_run_{0}.cmd" -f ([guid]::NewGuid().ToString('N')))
    $cmdLines = @(
        "@echo off","setlocal",
        "set VCPKG_FEATURE_FLAGS=manifests,versions,registries,binarycaching,quiet",
        "set VCPKG_DEFAULT_TRIPLET=$Triplet",
        "set VCPKG_DEFAULT_BINARY_CACHE=$($ProjectRoot)\vcpkg_cache",
        "set VCPKG_ROOT=$($ProjectRoot)\vcpkg",
        "set VCPKG_KEEP_ENV_VARS=VCPKG_FEATURE_FLAGS;VCPKG_DEFAULT_TRIPLET;VCPKG_DEFAULT_BINARY_CACHE;CUDA_PATH;CUDA_HOME;PATH;INCLUDE;LIB",
        "call `"$VcpkgExe`" install --triplet $Triplet --x-manifest-root=`"$ProjectRoot`" --x-install-root=`"$InstalledDir`""
    )
    Set-Content -Path $tmpCmd -Value ($cmdLines -join "`r`n") -Encoding ASCII

    Disable-CmdAutoRun
    try {
        $code = Invoke-WithTool `
            -SourceTag 'VCPKG' `
            -Exe 'cmd.exe' `
            -ArgumentList @('/d','/c', $tmpCmd) `
            -WorkingDirectory $ProjectRoot `
            -LogPath $log `
            -OnLine { param($line) Update-BuildProgress -Percent -1 -Status 'vcpkg install' }
        if ($code -ne 0) {
            Write-Log ERR "[VCPKG] install failed with $code"
            Write-Log ERR "[VCPKG] ---- last 120 lines of: $($log) ----"
            if (Test-Path $log) {
                Get-Content $log -Tail 120 -ErrorAction SilentlyContinue |
                    Where-Object { $_ -and $_.Trim() -ne '' } |
                    ForEach-Object { Write-Log INFO "[VCPKG] $_" }
                Write-Log ERR "[VCPKG] ---- end of log tail ----"
            } else {
                Write-Log ERR "[VCPKG] Log not found."
            }
            return $false
        }
        return $true
    } finally {
        Restore-CmdAutoRun
        try { Remove-Item $tmpCmd -Force -ErrorAction SilentlyContinue } catch {}
    }
}

# -------- tool output filter --------------------------------------------------
function Select-ToolOutput {
    param([Parameter(ValueFromPipeline=$true)][string]$Line)
    process {
        if ($null -eq $Line) { return }
        $l = $Line.TrimEnd()
        if ($l -match '^\s*The following packages are already installed:') { $script:_FilterState.dropPkg = $true; return }
        if ($script:_FilterState.dropPkg) {
            if ($l -eq '' -or $l -match '^All requested installations completed successfully') { $script:_FilterState.dropPkg = $false; if ($l -ne '') { $l } }
            return
        }
        if ($l -match '^(The package .* is compatible|glfw3 provides (?:CMake targets|pkg-config modules):)') { $script:_FilterState.dropAdvice = $true; return }
        if ($script:_FilterState.dropAdvice) {
            if ($l -eq '' -or $l -match '^(All requested installations completed successfully|-- )') { $script:_FilterState.dropAdvice = $false; if ($l -ne '') { return $l } else { return } }
            return
        }
        $l
    }
}

# -------- output helpers (prefix + de-dupe) ----------------------------------
function Remove-InitialDuplicateTag([string]$line){
    if ($null -eq $line) { return $null }
    if ($line -match '^\[(?<tag>[A-Z0-9_]+)\]\s+(?<rest>.*)$') {
        $tag  = $matches['tag']
        $rest = $matches['rest']
        $pat  = ('^\[{0}\]' -f ([regex]::Escape($tag)))
        if ($rest -match $pat) { return $rest }
    }
    return $line
}
function Write-RustLine {
    [CmdletBinding()]
    param([Parameter(Mandatory,ValueFromPipeline)][string]$Line)
    process {
        if ($null -eq $Line) { return }
        $l = Remove-InitialDuplicateTag $Line
        $fg = 'Gray'
        if ($l -match '^\[\s*WARN\s*\]') { $fg = 'Yellow' }
        elseif ($l -match '^\[\s*ERR\s*\]') { $fg = 'Red' }
        elseif ($l -match '^\[\s*OK\s*\]') { $fg = 'Green' }
        if ($script:UseColor) { Write-Host ("[RUST] " + $l) -ForegroundColor $fg } else { Write-Host ("[RUST] " + $l) }
    }
}

# -------- Rust runner (install) ----------------------------------------------
function Install-RustRunner {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$ProjectRoot)

    $projDir  = Join-Path $ProjectRoot 'rust\otter_proc'
    $manifest = Join-Path $projDir 'Cargo.toml'

    # If no Rust project, fall back without failing the step
    if (-not (Test-Path $manifest)) {
        Write-Log INFO "[RUST] otter_proc project not found; using native PowerShell runner."
        $script:OtterProc = $null
        return
    }

    # cargo lookup
    $cargo = $null
    try { $cargo = (Get-Command cargo.exe -ErrorAction Stop).Source } catch {}
    if (-not $cargo) {
        Write-Log WARN "[RUST] cargo.exe not found; using native PowerShell runner."
        $script:OtterProc = $null
        return
    }

    Write-Log INFO "[RUST] Building otter_proc (release)..."
    $p = Start-Process -FilePath $cargo -ArgumentList @('build','--release','--manifest-path', $manifest, '--quiet') -WorkingDirectory $projDir -Wait -PassThru -NoNewWindow
    if ($p.ExitCode -ne 0) {
        Write-Log WARN ("[RUST] cargo build failed (code={0}); using native PowerShell runner." -f $p.ExitCode)
        $script:OtterProc = $null
        return
    }

    $candidate = Join-Path $projDir 'target\release\otter_proc.exe'
    if (Test-Path $candidate) {
        $script:OtterProc = $candidate
        Write-Log INFO ("[RUST] Runner ready: {0}" -f $candidate)
    } else {
        Write-Log WARN "[RUST] Built runner not found; using native PowerShell runner."
        $script:OtterProc = $null
    }
}

# -------- generic process runner (fallback + rust) ---------------------------
function Invoke-WithTool {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$SourceTag,         # CMAKE|NINJA|GEN|SUPPORT|VCPKG
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$LogPath,
        [Parameter(Mandatory)][ScriptBlock]$OnLine
    )
    if ($script:OtterProc) {
        return Invoke-WithOtterProc `
            -SourceTag $SourceTag `
            -Exe $Exe `
            -ArgumentList $ArgumentList `
            -WorkingDirectory $WorkingDirectory `
            -LogPath $LogPath `
            -OnLine $OnLine
    } else {
        return Invoke-WithNativeProc `
            -SourceTag $SourceTag `
            -Exe $Exe `
            -ArgumentList $ArgumentList `
            -WorkingDirectory $WorkingDirectory `
            -LogPath $LogPath `
            -OnLine $OnLine
    }
}
function Invoke-WithNativeProc {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$SourceTag,
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$LogPath,
        [Parameter(Mandatory)][ScriptBlock]$OnLine
    )
    $dir = Split-Path $LogPath -Parent
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $Exe
    $psi.Arguments              = (Join-Args $ArgumentList)
    $psi.WorkingDirectory       = $WorkingDirectory
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute        = $false
    $psi.CreateNoWindow         = $true
    try { $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
    try { $psi.StandardErrorEncoding  = [System.Text.Encoding]::UTF8 } catch {}

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $null = $proc.Start()

    $out = New-Object System.IO.StreamReader($proc.StandardOutput.BaseStream, [Text.UTF8Encoding]::new($false,$true))
    $err = New-Object System.IO.StreamReader($proc.StandardError.BaseStream,  [Text.UTF8Encoding]::new($false,$true))

    $logWriter = $null
    try { $logWriter = New-Object System.IO.StreamWriter($LogPath, $false, [Text.UTF8Encoding]::new($false)) } catch {}

    while (-not $proc.HasExited) {
        while (-not $out.EndOfStream) {
            $line = $out.ReadLine()
            if ($null -ne $line) {
                $flt = Select-ToolOutput $line
                if ($null -eq $flt) { continue }
                $final = Remove-InitialDuplicateTag $flt
                if ($logWriter) { $logWriter.WriteLine($final) }
                Write-RustLine $final
                & $OnLine $final
            }
        }
        while (-not $err.EndOfStream) {
            $line = $err.ReadLine()
            if ($null -ne $line) {
                $flt = Select-ToolOutput $line
                if ($null -eq $flt) { continue }
                $final = Remove-InitialDuplicateTag $flt
                if ($logWriter) { $logWriter.WriteLine($final) }
                Write-RustLine $final
                & $OnLine $final
            }
        }
        Update-BuildProgress -Percent -1 -Status $script:LastStatusText
        Start-Sleep -Milliseconds 50
    }
    while (-not $out.EndOfStream) {
        $line = $out.ReadLine()
        if ($null -ne $line) {
            $flt = Select-ToolOutput $line
            if ($null -eq $flt) { continue }
            $final = Remove-InitialDuplicateTag $flt
            if ($logWriter) { $logWriter.WriteLine($final) }
            Write-RustLine $final
            & $OnLine $final
        }
    }
    while (-not $err.EndOfStream) {
        $line = $err.ReadLine()
        if ($null -ne $line) {
            $flt = Select-ToolOutput $line
            if ($null -eq $flt) { continue }
            $final = Remove-InitialDuplicateTag $flt
            if ($logWriter) { $logWriter.WriteLine($final) }
            Write-RustLine $final
            & $OnLine $final
        }
    }

    try { if ($logWriter) { $logWriter.Flush(); $logWriter.Dispose() } } catch {}
    $proc.WaitForExit()
    return $proc.ExitCode
}

# -------- Rust exec wrapper via otter_proc -----------------------------------
function Invoke-WithOtterProc {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$SourceTag,         # CMAKE|NINJA|GEN|SUPPORT|VCPKG
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$LogPath,
        [Parameter(Mandatory)][ScriptBlock]$OnLine
    )
    if (-not $script:OtterProc) {
        # Fallback seamlessly if runner not available
        return Invoke-WithNativeProc -SourceTag $SourceTag -Exe $Exe -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -LogPath $LogPath -OnLine $OnLine
    }

    $dir = Split-Path $LogPath -Parent
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

    $opArgs = @("--exe=$Exe")
    foreach ($a in $ArgumentList) { if ($null -ne $a) { $opArgs += @("--arg=$a") } }
    if ($WorkingDirectory) { $opArgs += @("--cwd=$WorkingDirectory") }
    $opArgs += @("--log=$LogPath", "--source=$SourceTag")

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = $script:OtterProc
    $psi.Arguments              = (Join-Args $opArgs)
    $psi.WorkingDirectory       = $WorkingDirectory
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute        = $false
    $psi.CreateNoWindow         = $true
    try { $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
    try { $psi.StandardErrorEncoding  = [System.Text.Encoding]::UTF8 } catch {}

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $null = $proc.Start()

    $out = New-Object System.IO.StreamReader($proc.StandardOutput.BaseStream, [Text.UTF8Encoding]::new($false,$true))
    $err = New-Object System.IO.StreamReader($proc.StandardError.BaseStream,  [Text.UTF8Encoding]::new($false,$true))

    $echoRaw = $true
    if ($env:OTTER_ECHO_TOOL_STDOUT -and ($env:OTTER_ECHO_TOOL_STDOUT -match '^(0|false)$')) { $echoRaw = $false }

    while (-not $proc.HasExited) {
        while (-not $out.EndOfStream) {
            $line = $out.ReadLine()
            if ($null -ne $line) {
                $flt = Select-ToolOutput $line
                if ($null -eq $flt) { continue }
                if ($echoRaw) { Write-RustLine $flt }
                & $OnLine $flt
            }
        }
        while (-not $err.EndOfStream) {
            $line = $err.ReadLine()
            if ($null -ne $line) {
                $flt = Select-ToolOutput $line
                if ($null -eq $flt) { continue }
                if ($echoRaw) { Write-RustLine $flt }
                & $OnLine $flt
            }
        }
        Update-BuildProgress -Percent -1 -Status $script:LastStatusText
        Start-Sleep -Milliseconds 50
    }
    while (-not $out.EndOfStream) {
        $line = $out.ReadLine()
        if ($null -ne $line) {
            $flt = Select-ToolOutput $line
            if ($null -eq $flt) { continue }
            if ($echoRaw) { Write-RustLine $flt }
            & $OnLine $flt
        }
    }
    while (-not $err.EndOfStream) {
        $line = $err.ReadLine()
        if ($null -ne $line) {
            $flt = Select-ToolOutput $line
            if ($null -eq $flt) { continue }
            if ($echoRaw) { Write-RustLine $flt }
            & $OnLine $flt
        }
    }

    $proc.WaitForExit()
    return $proc.ExitCode
}

# -------- Git repair (stuck rebase) ------------------------------------------
function Repair-GitRebaseIfStuck {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$RepoRoot)
    try {
        $gitDir = Join-Path $RepoRoot '.git'
        if (-not (Test-Path $gitDir)) { return $false }
        $hasMerge = Test-Path (Join-Path $gitDir 'rebase-merge')
        $hasApply = Test-Path (Join-Path $gitDir 'rebase-apply')
        if (-not ($hasMerge -or $hasApply)) { return $false }

        Write-Log WARN "[GIT] Detected in-progress rebase; attempting safe abort."
        $git = 'git'
        $p = Start-Process -FilePath $git -ArgumentList @('rebase','--abort') -WorkingDirectory $RepoRoot -Wait -PassThru -NoNewWindow
        if ($p.ExitCode -eq 0) { Write-Log INFO "[GIT] Rebase aborted."; return $true }

        Write-Log WARN "[GIT] rebase --abort failed (code=$($p.ExitCode)). Removing rebase state dirs..."
        foreach ($d in @('rebase-merge','rebase-apply')) {
            $path = Join-Path $gitDir $d
            if (Test-Path $path) {
                try { Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue } catch {}
            }
        }
        Write-Log INFO "[GIT] Rebase state removed. You may 'git fetch' and re-run."
        return $true
    } catch {
        Write-Log WARN "[GIT] Repair failed: $($_.Exception.Message)"
        return $false
    }
}

# -------- exports -------------------------------------------------------------
Export-ModuleMember -Function `
    Write-Log, Write-Rule, Write-SuccessBox, `
    Start-BuildProgress, Update-BuildProgress, Stop-BuildProgress, `
    Initialize-BuildMetricsDir, Read-DotEnv, `
    Resolve-Nvcc, Resolve-CMake, Resolve-Dumpbin, Resolve-Cl, Resolve-GitDir, Import-VcEnvSafe, Add-ToPath, `
    Import-BuildMetrics, Get-ExpectedMs, Add-RunToHistory, Update-BaselineFromHistory, `
    Install-RustRunner, Invoke-WithTool, Invoke-WithOtterProc, Invoke-WithNativeProc, Select-ToolOutput, `
    Disable-CmdAutoRun, Restore-CmdAutoRun, Invoke-VcpkgInstallSafe, `
    Repair-GitRebaseIfStuck, `
    Format-Size, `
    Remove-InitialDuplicateTag, Write-RustLine
