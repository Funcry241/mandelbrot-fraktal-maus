<#
.SYNOPSIS
    MausGitAutoCommit.ps1
.DESCRIPTION
    Automatically commit and push all changes in the existing Git repository.
#>

param(
    [string]$Message = "Auto-Commit: Changes OtterDream"
)

$ErrorActionPreference = 'Stop'

# 1) Prüfe, ob Git-Repository existiert
if (-not (Test-Path ".git")) {
    Write-Error "[GIT] No Git repository found! Please initialize with MausGit.ps1."
    exit 1
}

# 2) Dateien zum Commit vormerken (kein Output)
git add . >$null 2>&1

# 3) Prüfe auf Änderungen
$status = git status --porcelain
if (-not $status) {
    Write-Host "[GIT] No changes to commit."
    exit 0
}

# 4) Commit
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = "$Message $timestamp"
git commit -m "$fullMessage" >$null 2>&1
Write-Host "[GIT] Commit created: $fullMessage"

# 5) Aktuellen Branch ermitteln
$branch = git rev-parse --abbrev-ref HEAD

# 6) Push ohne Lärm
git push origin $branch --quiet
Write-Host "[GIT] Pushed to '$branch'."

# 7) Seltenes Otter-Artwork
if ((Get-Random -Minimum 1 -Maximum 21) -eq 1) {
    @(
        "    .--.",
        "   /    \\",
        "  /_    _\\",
        " // \\  / \\\\",
        " |\\__/\\__|",
        " \\    /\\   /",
        "  '--'  '--'"
    ) | ForEach-Object { Write-Host $_ }
    Write-Host "Rare Otter Encounter!"
}

Write-Host "[GIT] AutoGit completed."
exit 0
