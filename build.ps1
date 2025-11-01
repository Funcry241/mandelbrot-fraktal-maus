# Otter: Ultra-thin wrapper — Cargo runs the Rust orchestrator; no PS “magie”.
# Schneefuchs: PS 5.1-safe; clean exit-code pass-through; strict paths.
# Maus: One-line ASCII steps; deterministic; fail fast if cargo/runner fails.
# Datei: .\build.ps1
param(
  [ValidateSet('Debug','Release','RelWithDebInfo','MinSizeRel')]
  [string]$Configuration = 'RelWithDebInfo'
)

$ErrorActionPreference = 'Stop'

# repo root (robust, PS 5.1-safe)
$root = if ($PSScriptRoot) { $PSScriptRoot } elseif ($PSCommandPath) { Split-Path -Path $PSCommandPath -Parent } else { (Get-Location).Path }
$root = (Resolve-Path -LiteralPath $root).Path

# runner path
$runnerDir = Join-Path $root 'rust\otter_proc'

Write-Host "[PS] [INFO] === Build (Cargo-run) started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# preflight: cargo present?
$cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargoCmd) {
  Write-Host "[PS] [ERR] 'cargo' not found in PATH"
  exit 88
}

# preflight: runner dir present?
if (-not (Test-Path -LiteralPath $runnerDir)) {
  Write-Host "[PS] [ERR] Runner directory missing: $runnerDir"
  exit 87
}

# optional: also provide OTTER_ROOT for runner convenience
$env:OTTER_ROOT = $root

Write-Host "[PS] [INFO] [STEP] cargo run --release -- --root $root full --cfg $Configuration"

Push-Location -LiteralPath $runnerDir
try {
  & cargo run --release -- '--root' $root 'full' '--cfg' $Configuration
  $code = $LASTEXITCODE
}
finally {
  Pop-Location
}

if ($code -ne 0) {
  Write-Host "[PS] [ERR] cargo run failed (code=$code)"
  exit $code
}

Write-Host "[PS] [INFO] === Build finished (code=0) ==="
exit 0
