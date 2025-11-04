##### Otter: Three modes — (no args)=local build (no git), /build=build+upload, /branch=branch op.
##### Schneefuchs: PS 5.1-safe; English help (-h | -? | /h | /?); ASCII logs; clean exit codes; neutral tag.
##### Maus: Minimal & deterministic; accepts / or -; default branch name "wupp".
##### Datei: .\build.ps1

$ErrorActionPreference = 'Stop'

function Info([string]$m){ Write-Host "[RUN] $m" }
function Err ([string]$m){ Write-Host "[RUN] [ERR] $m" -ForegroundColor Red; exit 1 }

function Show-Help {
  Write-Host @"
Usage:
  .\build.ps1                 Run a LOCAL BUILD (no Git upload)
  .\build.ps1 /build          Build + commit/push via Rust runner
  .\build.ps1 /branch         Create/switch 'wupp' and push -u origin/wupp

Options:
  -h, /h, -?, /?              Show this help

Notes:
  - Local build sets OTTER_UPLOAD=0 so the Rust runner skips autogit.
  - /build sets OTTER_UPLOAD=1 (commit+push on current branch).
  - /branch uses OTTER_OP=branch (always pushes -u origin/wupp).
"@
}

# Parse args (PS 5.1-safe)
$wantHelp = $false
$mode = ''   # '', 'build', 'branch'
$seenBuild = $false
$seenBranch = $false
foreach ($a in $args) {
  $al = ($a + '').ToLower()
  if ($al -in @('-h','/h','-?','/?')) { $wantHelp = $true; continue }
  if ($al -eq '/branch' -or $al -eq '-branch') { $seenBranch = $true; $mode = 'branch'; continue }
  if ($al -eq '/build'  -or $al -eq '-build')  { $seenBuild  = $true; $mode = 'build' ; continue }
}
if ($wantHelp) { Show-Help; exit 0 }
if ($seenBuild -and $seenBranch) { Err "Choose either /build or /branch, not both." }

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

# Common ENV
$env:OTTER_ROOT = $root

Push-Location -LiteralPath $runnerDir
try {
  if ($mode -eq 'branch') {
    # Branch mode (create/switch & push -u origin/wupp)
    $env:OTTER_OP     = 'branch'
    $env:OTTER_BRANCH = 'wupp'
    Remove-Item Env:OTTER_UPLOAD -ErrorAction SilentlyContinue
    Info "[STEP] cargo run --release    (OTTER_OP=branch OTTER_BRANCH=wupp)"
    & cargo run --release --
  } elseif ($mode -eq 'build') {
    # Build + upload (commit+push on current branch)
    Remove-Item Env:OTTER_OP -ErrorAction SilentlyContinue
    $env:OTTER_UPLOAD = '1'
    Info "[STEP] cargo run --release -- --root $root full --cfg RelWithDebInfo (upload=ON)"
    & cargo run --release -- '--root' $root 'full' '--cfg' 'RelWithDebInfo'
  } else {
    # Default: local build (NO upload)
    Remove-Item Env:OTTER_OP -ErrorAction SilentlyContinue
    $env:OTTER_UPLOAD = '0'
    Info "[STEP] cargo run --release -- --root $root full --cfg RelWithDebInfo (upload=OFF)"
    & cargo run --release -- '--root' $root 'full' '--cfg' 'RelWithDebInfo'
  }
  $code = $LASTEXITCODE
} finally {
  Pop-Location
}

if ($code -ne 0) { Err "cargo run failed (code=$code)" }
Info "OK (code=0)"
exit 0
