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

# 2) Dateien zum Commit vormerken
Write-Host "[GIT] Staging changes..."
git add .

# 3) Prüfe, ob es etwas zu committen gibt
$status = git status --porcelain
if (-not $status) {
    Write-Host "[GIT] No changes to commit."
    exit 0
}

# 4) Commit mit Zeitstempel erstellen
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = "$Message $timestamp"

Write-Host "[GIT] Creating commit with message: $fullMessage"
git commit -m "$fullMessage"

# 5) Aktuellen Branch ermitteln
$branch = git rev-parse --abbrev-ref HEAD

# 6) Push auf Remote
Write-Host "[GIT] Pushing to branch '$branch'..."
git push origin $branch

Write-Host "[GIT] Changes successfully pushed."

# 7) Optionale Überraschung (1 von 20)
$rareChance = Get-Random -Minimum 1 -Maximum 21
if ($rareChance -eq 1) {
    $otterArt = @(
        "    .--.",
        "   /    \\",
        "  /_    _\\",
        " // \\  / \\\\",
        " |\\__/\\__|",
        " \\    /\\   /",
        "  '--'  '--'"
    )
    foreach ($line in $otterArt) {
        Write-Host $line
    }
    Write-Host "Rare Otter Encounter!"
}

Write-Host "[GIT] AutoGit script completed."
exit 0
