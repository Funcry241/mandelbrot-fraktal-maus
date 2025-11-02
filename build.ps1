##### Otter: Forwarder – /branch (delegiert an Rust) oder /build (delegiert an Rust).
##### Schneefuchs: PS 5.1-safe; keine Git-Logik mehr in PS; saubere Exitcodes; ASCII-Logs.
##### Maus: Minimal & deterministisch; akzeptiert / oder -; Branchname fix "wupp".
##### Datei: .\build.ps1

$ErrorActionPreference = 'Stop'

function Info([string]$m){ Write-Host "[PS] $m" }
function Err ([string]$m){ Write-Host "[PS] [ERR] $m" -ForegroundColor Red; exit 1 }

# Repo-Root (Ordner des Skripts)
$root = $PSScriptRoot
if (-not $root) {
  if ($PSCommandPath) { $root = Split-Path -Path $PSCommandPath -Parent } else { $root = (Get-Location).Path }
}
try { $root = (Resolve-Path -LiteralPath $root).Path } catch { Err "Root nicht gefunden" }

# Mode aus Args: /branch oder /build
$mode = ''
foreach ($a in $args) {
  $al = ($a + '').ToLower()
  if ($al -eq '/branch' -or $al -eq '-branch') { $mode = 'branch' }
  if ($al -eq '/build'  -or $al -eq '-build')  { $mode = 'build'  }
}
if ($mode -eq '') { Err "Nutze:  .\build.ps1 /branch   oder   .\build.ps1 /build" }

# cargo vorhanden?
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargo) { Err "'cargo' nicht im PATH" }

# Runner-Ordner
$runnerDir = Join-Path $root 'rust\otter_proc'
if (-not (Test-Path -LiteralPath $runnerDir)) { Err "Runner-Verzeichnis fehlt: $runnerDir" }

# Gemeinsame ENV
$env:OTTER_ROOT = $root

Push-Location -LiteralPath $runnerDir
try {
  if ($mode -eq 'branch') {
    # Rust-Branchmodus via ENV
    $env:OTTER_OP     = 'branch'
    $env:OTTER_BRANCH = 'wupp'
    Info "[STEP] cargo run --release    (OTTER_OP=branch OTTER_BRANCH=wupp)"
    & cargo run --release --
  } else {
    # Full-Build + Autogit (Branch autodetect)
    Remove-Item Env:OTTER_OP -ErrorAction SilentlyContinue
    Info "[STEP] cargo run --release -- --root $root full --cfg RelWithDebInfo"
    & cargo run --release -- '--root' $root 'full' '--cfg' 'RelWithDebInfo'
  }
  $code = $LASTEXITCODE
} finally {
  Pop-Location
}

if ($code -ne 0) { Err "cargo run fehlgeschlagen (code=$code)" }
Info "OK (code=0)"
exit 0
