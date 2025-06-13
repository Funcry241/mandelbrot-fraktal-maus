<#
.SYNOPSIS
    MausGitAutoCommit.ps1
.DESCRIPTION
    Automatically commit and push all changes in the existing Git repository.
#>

param(
    [string]$Message = "Auto-Commit: Changes OtterDream"
)

# Stop on errors
$ErrorActionPreference = 'Stop'

# 1) Check if Git repository exists
if (-not (Test-Path ".git")) {
    Write-Error "[GIT] No Git repository found! Please initialize with MausGit.ps1."
    exit 1
}

# 2) Stage changes
Write-Host "[GIT] Staging changes..."
git add .

# 3) Check if there is anything to commit
$status = git status --porcelain
if (-not $status) {
    Write-Host "[GIT] No changes to commit." -ForegroundColor Yellow
    exit 0
}

# 4) Create commit (with timestamp)
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = "$Message $timestamp"

Write-Host "[GIT] Creating commit with message:" -NoNewline
Write-Host " $fullMessage" -ForegroundColor Cyan
git commit -m "$fullMessage"

# 5) Find current branch
$branch = git rev-parse --abbrev-ref HEAD

# 6) Push to remote
Write-Host "[GIT] Pushing to branch '$branch'..."
git push origin $branch

Write-Host "[GIT] Changes successfully pushed!" -ForegroundColor Green

# 7) Rare Otter Surprise (1 in 20 chance)
$rareChance = Get-Random -Minimum 1 -Maximum 21
if ($rareChance -eq 1) {
    $otterArt = @(
        '    .--.',
        '   /    \',
        '  /_    _\',
        ' // \  / \\',
        ' |\__\/__/|',
        ' \    /\   /',
        '  ''--''  ''--'''
    )
    foreach ($line in $otterArt) {
        Write-Host $line -ForegroundColor Cyan
    }
    Write-Host "Rare Otter Encounter!" -ForegroundColor Magenta
} # <-- diese schließende Klammer fehlte ursprünglich
