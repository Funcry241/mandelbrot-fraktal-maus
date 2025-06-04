<#
.SYNOPSIS
    MausGitAutoCommit.ps1
.DESCRIPTION
    Automatischer Commit und Push aller Änderungen im bestehenden Git-Repository.
#>

param(
    [string]$Message = "Auto-Commit: Aenderungen OtterDream"
)

# Stoppe bei Fehlern
$ErrorActionPreference = 'Stop'

# 1) Git-Repository prüfen
if (-not (Test-Path ".git")) {
    Write-Error "[GIT] Kein Git-Repository gefunden! Bitte initialisieren mit MausGit.ps1."
    exit 1
}

# 2) Änderungen zum Commit vormerken
Write-Host "[GIT] Stage Aenderungen..."
git add .

# 3) Prüfen, ob es etwas zu committen gibt
$status = git status --porcelain
if (-not $status) {
    Write-Host "[GIT] Keine Aenderungen zum Commit." -ForegroundColor Yellow
    exit 0
}

# 4) Commit erstellen (jetzt mit Datum/Uhrzeit)
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = "$Message $timestamp"

Write-Host "[GIT] Erstelle Commit mit Nachricht:" -NoNewline
Write-Host " $fullMessage" -ForegroundColor Cyan
git commit -m "$fullMessage"

# 5) Aktuellen Branch herausfinden
$branch = git rev-parse --abbrev-ref HEAD

# 6) Push auf Remote
Write-Host "[GIT] Pushe auf Branch '$branch'..."
git push origin $branch

Write-Host "[GIT] ✅ Aenderungen erfolgreich gepusht!" -ForegroundColor Green
