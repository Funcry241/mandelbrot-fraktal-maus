# File: ExportMaus.ps1
<# MAUS-COMMENT [🖱️]: Datei für Export-Skript, filtert alle Quell- und Dokumentdateien, ignoriert Build/Dist und speichert relative Pfade im ZIP. Wichtig: $PSScriptRoot, Patterns, Ignore-Folders. #>

<#+
.SYNOPSIS
    ExportMaus.ps1 – Exportiert relevante Projektdateien als ZIP-Archiv, inklusive README.MAUS.

.DESCRIPTION
    Sammelt rekursiv Quellcode-, Konfigurations-, Dokumentations- und .MAUS-Dateien
    (C/C++, CUDA, CMakeLists, JSON, MD, PS1, TXT, MAUS) und erstellt ein ZIP-Archiv.
    Ignoriert dabei Build-, Dist- und temporäre Ordner sowie das Ziel-Archiv selbst.

.PARAMETER Output
    Optionaler Pfad und Name des ZIP-Archivs (Standard: ./OtterDream_Project.zip).

.EXAMPLE
    .\ExportMaus.ps1
    Erstellt OtterDream_Project.zip im aktuellen Verzeichnis.

.EXAMPLE
    .\ExportMaus.ps1 -Output "C:\Temp\MeinProjekt.zip"
#>

param(
    [string]$Output = "$PSScriptRoot\OtterDream_Project.zip"
)

# Zu ignorierende Ordner
$ignoreFolders   = @('build','build-vs','dist','tmp','.git','.vs','vcpkg','vcpkg_installed')
# Zulässige Dateimuster (jetzt auch .MAUS)
$includePatterns = @(
    '*.cpp','*.c','*.h','*.hpp','*.cu','*.cuh',
    'CMakeLists.txt','*.json','*.md','*.ps1','*.txt','*.MAUS'
)

Write-Host "[INFO] Sammle Dateien für Export..."
# Alle Dateien finden und filtern
$allFiles = Get-ChildItem -Path $PSScriptRoot -Recurse -File | Where-Object {
    $path = $_.FullName

    # Nur erlaubte Dateitypen
    $matchesPattern = $includePatterns | Where-Object { $path -like "*$_" }
    if (-not $matchesPattern) { return $false }

    # Ignoriere Ordner
    foreach ($fld in $ignoreFolders) {
        if ($path -match "\\$fld\\") { return $false }
    }

    # Ignoriere ZIP-Archiv selbst
    $outputFull = [System.IO.Path]::GetFullPath($Output)
    if ($path -eq $outputFull) { return $false }

    return $true
}

if (-not $allFiles -or $allFiles.Count -eq 0) {
    Write-Error "[ERROR] Keine Dateien zum Archivieren gefunden."
    exit 1
}

# Lösche altes Archiv
if (Test-Path $Output) {
    Remove-Item $Output -Force
}

# Wechsel ins Projektverzeichnis, um relative Pfade zu erzeugen
Push-Location $PSScriptRoot

# Erzeuge relative Pfade (ohne führendes "\"), damit Compress-Archive die
# gesamte Ordnerstruktur ab $PSScriptRoot beibehält.
$relativePaths = $allFiles | ForEach-Object {
    $_.FullName.Substring($PSScriptRoot.Length + 1)
}

Write-Host "[INFO] Erstelle ZIP-Archiv: $Output"
try {
    Compress-Archive `
        -Path $relativePaths `
        -DestinationPath $Output `
        -CompressionLevel Fastest `
        -Force

    Pop-Location

    Write-Host "[OK] Export erfolgreich! Enthält $($allFiles.Count) Dateien."
} catch {
    Pop-Location
    Write-Error "[ERROR] Fehler beim Erstellen des ZIP: $_"
    exit 1
}
