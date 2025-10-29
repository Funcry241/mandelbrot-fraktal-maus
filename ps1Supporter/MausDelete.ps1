##### Otter: Fast & safe cleanup — dir-first; provider filters; dist preserved.
##### Schneefuchs: Weniger FS-Pässe; Excludes sauber; /WX-safe.
##### Maus: Einzeilige Logs mit Farben; DryRun; am Ende klare Summary.
##### Datei: ps1Supporter/MausDelete.ps1
#requires -version 5.1

[CmdletBinding()]
param(
  [switch]$DryRun = $false,
  [string]$Root   = $PSScriptRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'
$ConfirmPreference     = 'None'

# ---- mini logger (Maus-Stil) -------------------------------------------------
$script:UseColor = $Host -and $Host.UI -and $Host.UI.RawUI
function MDLog {
  param([ValidateSet('INFO','OK','WARN','ERR','DRY')][string]$Level='INFO',[Parameter(Mandatory)][string]$Msg)
  $p = '[MausDelete] '
  if (-not $script:UseColor) { Write-Host ($p + $Msg); return }
  switch ($Level) {
    'INFO' { Write-Host $p -NoNewline -ForegroundColor Gray;   Write-Host $Msg }
    'OK'   { Write-Host $p -NoNewline -ForegroundColor Green;  Write-Host $Msg }
    'WARN' { Write-Host $p -NoNewline -ForegroundColor Yellow; Write-Host $Msg }
    'ERR'  { Write-Host $p -NoNewline -ForegroundColor Red;    Write-Host $Msg }
    'DRY'  { Write-Host $p -NoNewline -ForegroundColor Cyan;   Write-Host $Msg }
  }
}

# Avoid following junctions/symlinks
function NotReparse { param($item) return -not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) }

# Löschen mit SupportsShouldProcess (PSScriptAnalyzer-konform)
function Remove-Found {
  [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
  param(
    [Parameter(Mandatory)]$items,
    [Parameter(Mandatory)][string]$tag,
    [switch]$IsDir
  )
  $n = 0
  foreach ($i in $items) {
    if ($DryRun) {
      MDLog DRY ("[$tag] $($i.FullName)")
    } else {
      try {
        if ($PSCmdlet.ShouldProcess($i.FullName, 'Remove')) {
          Remove-Item -LiteralPath $i.FullName -Recurse:([bool]$IsDir) -Force -ErrorAction Stop
        }
      } catch {
        MDLog WARN ("[$tag] could not remove: $($i.FullName)  ($($_.Exception.Message))")
      }
    }
    $n++
  }
  return $n
}

function FormatCount { '{0,4}' -f [int]$args[0] }

# ---- Guards ------------------------------------------------------------------
try {
  $rootItem = Get-Item -LiteralPath $Root -ErrorAction Stop
} catch {
  throw "Root path not found: $Root"
}
if ($rootItem.PSProvider.Name -ne 'FileSystem') { throw "Only FileSystem provider is allowed. Root='$Root'." }

$Root = $rootItem.FullName
if ($Root -match '^[A-Za-z]:\\$') { throw "Refusing to run on drive root: $Root" }
if ( ($Root -split '\\').Length -lt 3 ) { throw "Path too shallow for cleanup: $Root" }

Write-Host ''
MDLog INFO "Starting cleanup in: $Root"
if ($DryRun) { MDLog INFO "DryRun mode: nothing will be deleted." }

# === Konfiguration ============================================================
$fileExtensions  = @('.obj','.o','.ilk','.pdb','.log','.tmp','.tlog')
$filenameExact   = @('CMakeCache.txt','CMakeGenerate.stamp')
$filenameInclude = @('*.VC.db','*~')

$folderNames     = @('CMakeFiles','Debug','Release','x64','.vs','.idea')
$excludedFolders = @('dist')  # nie rekursiv löschen

# === Zähler ===================================================================
[int]$countFiles = 0
[int]$countDirs  = 0
[int]$countDist  = 0

$distPath = Join-Path $Root 'dist'
$rootStar = Join-Path $Root '*'

# === 1) Ganze Müll-Ordner zuerst (schnell), aber nie excluded ================
$dirs = Get-ChildItem -Path $Root -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object { (NotReparse $_) -and ($folderNames -contains $_.Name) -and ($excludedFolders -notcontains $_.Name) }
$countDirs += Remove-Found -items $dirs -tag 'dir' -IsDir

# === 2) Dateien in einem Rutsch: -Include für Exts/Names/Patterns ============
$extPatterns = $fileExtensions | ForEach-Object { "*$($_)" } # -> '*.obj', ...
$allInclude  = $extPatterns + $filenameExact + $filenameInclude

$files = Get-ChildItem -Path $rootStar -Recurse -File -Include $allInclude -ErrorAction SilentlyContinue |
         Where-Object { NotReparse $_ }

# Nichts aus dist/ (das räumen wir separat)
if (Test-Path -LiteralPath $distPath) {
  $distRoot = (Get-Item -LiteralPath $distPath).FullName
  $files = $files | Where-Object { $_.FullName -notlike "$distRoot\*" }
}
$countFiles += Remove-Found -items $files -tag 'file'

# === 3) dist/ leeren (Inhalt), Ordner behalten ===============================
if (Test-Path -LiteralPath $distPath) {
  $distItems = Get-ChildItem -Path $distPath -Force -ErrorAction SilentlyContinue |
               Where-Object { NotReparse $_ }
  $countDist += Remove-Found -items $distItems -tag 'dist'
}

MDLog OK ("Done. Files: {0}  Dirs: {1}  dist/: {2}" -f (FormatCount $countFiles),(FormatCount $countDirs),(FormatCount $countDist))
exit 0
##### Otter: Fast & safe cleanup — dir-first; provider filters; dist preserved.
##### Schneefuchs: Weniger FS-Pässe; Excludes sauber; /WX-safe.
##### Maus: Einzeilige Logs mit Farben; DryRun; am Ende klare Summary.
##### Datei: ps1Supporter/MausDelete.ps1

# -----------------------------------------------------------------------------
# MausDelete – Fast path
# Idee: Erst Ordner löschen, dann Dateien über Include/Filter; dist/ nur leeren.
# Besonderer Schutz: Rust-Build-Ausgaben (…\target\release) NIE entfernen.
# -----------------------------------------------------------------------------

param(
  [switch]$DryRun = $false,
  [string]$Root = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'
$ConfirmPreference     = 'None'

# ---- mini logger (Maus-Stil) -------------------------------------------------
$script:UseColor = $Host -and $Host.UI -and $Host.UI.RawUI
function MDLog {
  param([ValidateSet('INFO','OK','WARN','ERR','DRY')][string]$Level='INFO',[string]$Msg)
  $p = "[MausDelete] "
  if (-not $script:UseColor) { Write-Host ($p + $Msg); return }
  switch ($Level) {
    'INFO' { Write-Host $p -NoNewline -ForegroundColor Gray;   Write-Host $Msg }
    'OK'   { Write-Host $p -NoNewline -ForegroundColor Green;  Write-Host $Msg }
    'WARN' { Write-Host $p -NoNewline -ForegroundColor Yellow; Write-Host $Msg }
    'ERR'  { Write-Host $p -NoNewline -ForegroundColor Red;    Write-Host $Msg }
    'DRY'  { Write-Host $p -NoNewline -ForegroundColor Cyan;   Write-Host $Msg }
  }
}

# Entfernt gefundene Items; schützt optional Verzeichnisse; unterstützt DryRun
function Remove-Found {
  [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
  param(
    [Parameter(Mandatory)][System.IO.FileSystemInfo[]]$Items,
    [Parameter(Mandatory)][string]$Tag,
    [switch]$IsDir
  )
  $n = 0
  foreach ($i in $Items) {
    $path = $i.FullName
    if ($DryRun) {
      MDLog DRY ("[$Tag] $path")
    } else {
      if ($PSCmdlet.ShouldProcess($path, 'Remove-Item')) {
        try {
          Remove-Item -LiteralPath $path -Recurse:([bool]$IsDir) -Force -ErrorAction Stop
        } catch {
          MDLog WARN ("[$Tag] could not remove: $path  ($($_.Exception.Message))")
        }
      }
    }
    $n++
  }
  return $n
}

function Format-Count($n){ '{0,4}' -f [int]$n }

Write-Host ""
MDLog INFO "Starting cleanup..."

# === Konfiguration ============================================================
# Exts: so kurz wie möglich; Filter via -Include (ein Pass)
$fileExtensions  = @('.obj','.o','.ilk','.pdb','.log','.tmp','.tlog')
$filenameExact   = @('CMakeCache.txt','CMakeGenerate.stamp')
$filenameInclude = @('*.VC.db','*~')

# Achtung: Namen wie 'Release' kollidieren mit Rusts 'target\release'.
# Wir schützen '...\target\release' zusätzlich per Pfadfilter.
$folderNames     = @('CMakeFiles','Debug','Release','x64','.vs','.idea')
$excludedFolders = @('dist')  # nie rekursiv löschen

# === Vorbereitung =============================================================
[int]$countFiles = 0
[int]$countDirs  = 0
[int]$countDist  = 0

$distPath = Join-Path $Root 'dist'
$rootStar = Join-Path $Root '*'

# Regex-Schutz für Rust-Release-Ausgaben (case-insensitive)
$rustReleaseGuard = '\\target\\release(\\|$)'

# === 1) Ganze Müll-Ordner zuerst (schnell), aber nie excluded ================
$dirs = Get-ChildItem -Path $Root -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
          ($folderNames -contains $_.Name) -and
          ($excludedFolders -notcontains $_.Name) -and
          ($_.FullName -notmatch $rustReleaseGuard)
        }
$countDirs += Remove-Found -Items $dirs -Tag 'dir' -IsDir

# === 2) Dateien in einem Rutsch: -Include für Exts/Names/Patterns ============
$extPatterns = $fileExtensions  | ForEach-Object { "*$($_)" }       # -> '*.obj', ...
$allInclude  = $extPatterns + $filenameExact + $filenameInclude

$files = Get-ChildItem -Path $rootStar -Recurse -File -Include $allInclude -ErrorAction SilentlyContinue

# Nichts aus dist/ (das räumen wir separat)
if (Test-Path $distPath) {
  $files = $files | Where-Object { $_.FullName -notlike "$distPath\*" }
}

# Nichts aus Rusts target\release\
$files = $files | Where-Object { $_.FullName -notmatch $rustReleaseGuard }

$countFiles += Remove-Found -Items $files -Tag 'file'

# === 3) dist/ leeren (Inhalt), Ordner behalten ===============================
if (Test-Path $distPath) {
  $distItems = Get-ChildItem -Path $distPath -Force -ErrorAction SilentlyContinue
  $countDist += Remove-Found -Items $distItems -Tag 'dist'
}

MDLog OK ("Done. Files: {0}  Dirs: {1}  dist/: {2}" -f (Format-Count $countFiles),(Format-Count $countDirs),(Format-Count $countDist))
exit 0
