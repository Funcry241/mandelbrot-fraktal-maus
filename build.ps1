# Otter: Three modes — (no args)=local build (no git), /build=build+upload, /branch=branch op, /export=source ZIP. Rust owns DIST sync & trailer.
# Schneefuchs: PS 5.1-safe; English help; ASCII logs; clean exit codes; /j:/parallel passthrough to Rust.
# Maus: Minimal & deterministic; accepts / or -; default branch "wupp"; export supports /out, /keep, /dry(-run).
# Datei: .\build.ps1

$ErrorActionPreference = 'Stop'

function Info([string]$m){ Write-Host "[RUN] $m" }
function Err ([string]$m){ Write-Host "[RUN] [ERR] $m" -ForegroundColor Red; exit 1 }

function Show-Help {
  Write-Host @"
Usage:
  .\build.ps1                 Run a LOCAL BUILD (no Git upload)
  .\build.ps1 /build          Build + commit/push via Rust runner
  .\build.ps1 /branch         Create/switch 'wupp' and push -u origin/wupp
  .\build.ps1 /export         Create a source ZIP only (no build)

Options:
  -h, /h, -?, /?              Show this help
  /j:<N>, /parallel:<N>       Force parallel jobs for CMake build (Rust fallback auto-detects if omitted)

Export options (when using /export):
  /out:<PATH>                 Output directory for ZIPs (passed to --out-dir)
  /keep:<N>                   Keep at most N latest ZIPs (default 5)
  /dry or /dry-run            Dry run (no ZIP written)

Notes:
  - Rust handles DIST sync & the pretty trailer; PowerShell does not copy artifacts.
  - Local build sets OTTER_UPLOAD=0 so the Rust runner skips autogit.
  - /build sets OTTER_UPLOAD=1 (commit+push on current branch).
  - /branch uses OTTER_OP=branch (always pushes -u origin/wupp).
  - /export maps to 'otter_proc export' and works without a successful build.
"@
}

# Parse args (PS 5.1-safe)
$wantHelp      = $false
$mode          = ''    # '', 'build', 'branch', 'export'
$seenBuild     = $false
$seenBranch    = $false
$seenExport    = $false
[int]$jobs     = 0     # /j:/parallel passthrough to Rust

# /export parameters
[string]$exportOutDir = ''
[int]$exportKeep = 5
[bool]$exportDry  = $false

foreach ($a in $args) {
  $al = ($a + '').ToLower()

  if ($al -in @('-h','/h','-?','/?')) { $wantHelp = $true; continue }

  if ($al -eq '/branch' -or $al -eq '-branch') { $seenBranch = $true; $mode = 'branch'; continue }
  if ($al -eq '/build'  -or $al -eq '-build')  { $seenBuild  = $true; $mode = 'build' ; continue }
  if ($al -eq '/export' -or $al -eq '-export') { $seenExport = $true; $mode = 'export'; continue }

  # export-specific shorthands: /out:<path> or /out=<path>
  if ($al -match '^[\-/](out|o)[:=](.+)$') {
    $sep = $a.IndexOfAny(@(':','='))
    if ($sep -ge 0 -and $sep + 1 -lt $a.Length) {
      $exportOutDir = $a.Substring($sep + 1)
    }
    continue
  }

  # /keep:<N> or /keep=<N>
  if ($al -match '^[\-/](keep|k)[:=](\d+)$') {
    try { $exportKeep = [int]$matches[2] } catch { }
    continue
  }

  # /dry or /dry-run
  if ($al -in @('/dry','-dry','/dry-run','-dry-run')) { $exportDry = $true; continue }

  # /j:<N> or /parallel:<N>
  if ($al -match '^[\-/](j|parallel)[:=](\d+)$') {
    try { $jobs = [int]$matches[2] } catch { $jobs = 0 }
    continue
  }
}

if ($wantHelp) { Show-Help; exit 0 }
if ( ($seenBuild  -and $seenBranch) -or
     ($seenBuild  -and $seenExport) -or
     ($seenBranch -and $seenExport) ) {
  Err "Choose only ONE of /build, /branch, or /export."
}

# Repo root
$root = $PSScriptRoot
if (-not $root) {
  if ($PSCommandPath) { $root = Split-Path -Path $PSCommandPath -Parent } else { $root = (Get-Location).Path }
}
try { $root = (Resolve-Path -LiteralPath $root).Path } catch { Err "Root not found" }

# cargo present?
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargo) { Err "'cargo' not found in PATH" }

# Runner dir
$runnerDir = Join-Path $root 'rust\otter_proc'
if (-not (Test-Path -LiteralPath $runnerDir)) { Err "Runner directory missing: $runnerDir" }

# Common ENV (compat)
$env:OTTER_ROOT = $root

Push-Location -LiteralPath $runnerDir
try {
  if ($mode -eq 'branch') {
    $env:OTTER_OP     = 'branch'
    $env:OTTER_BRANCH = 'wupp'
    Remove-Item Env:OTTER_UPLOAD -ErrorAction SilentlyContinue
    Info "[STEP] cargo run --release    (OTTER_OP=branch OTTER_BRANCH=wupp)"
    & cargo run --release --
  }
  elseif ($mode -eq 'build') {
    Remove-Item Env:OTTER_OP -ErrorAction SilentlyContinue
    $env:OTTER_UPLOAD = '1'

    $cli = @('--root', $root, 'full', '--cfg', 'RelWithDebInfo')
    if ($jobs -gt 0) { $cli += @('--parallel', "$jobs") }

    Info "[STEP] cargo run --release -- $($cli -join ' ') (upload=ON)"
    & cargo run --release -- @cli
  }
  elseif ($mode -eq 'export') {
    Remove-Item Env:OTTER_UPLOAD -ErrorAction SilentlyContinue
    $env:OTTER_OP = 'export'

    $cli = @('export', '--max-keep', "$exportKeep")
    if ($exportOutDir) { $cli += @('--out-dir', $exportOutDir) }
    if ($exportDry)    { $cli += '--dry-run' }

    Info "[STEP] cargo run --release -- $($cli -join ' ')"
    & cargo run --release -- @cli
  }
  else {
    Remove-Item Env:OTTER_OP -ErrorAction SilentlyContinue
    $env:OTTER_UPLOAD = '0'

    $cli = @('--root', $root, 'full', '--cfg', 'RelWithDebInfo')
    if ($jobs -gt 0) { $cli += @('--parallel', "$jobs") }

    Info "[STEP] cargo run --release -- $($cli -join ' ') (upload=OFF)"
    & cargo run --release -- @cli
  }

  $code = $LASTEXITCODE
}
finally {
  Pop-Location
}

# Rust runner prints the pretty trailer (and performs DIST sync). Avoid duplicate success line here.
if ($code -ne 0) { Err "cargo run failed (code=$code)" }
exit 0
