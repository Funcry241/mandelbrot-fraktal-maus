#MAUS
<#
.SYNOPSIS
    MausGitAutoCommit.ps1
.DESCRIPTION
    Automatically commit and push all changes in the existing Git repository.
    Ensures the 100th-frame capture (dist/frame_0100.bmp) is included even if /dist is ignored.
#>

param(
    [string]$Message = "Auto-Commit: Changes OtterDream"
)

$ErrorActionPreference = 'Stop'

# 1) Repo present?
if (-not (Test-Path ".git")) {
    Write-Error "[GIT] No Git repository found! Please initialize with MausGit.ps1."
    exit 1
}

# 2) Stage changes (force-add capture if present)
Write-Host "[GIT] Staging changes..."
if (Test-Path "dist\frame_0100.bmp") {
    git add -f "dist/frame_0100.bmp" | Out-Null
    Write-Host "[GIT] Forced add: dist/frame_0100.bmp"
} else {
    Write-Host "[GIT] Note: dist/frame_0100.bmp not found (yet)."
}
git add . | Out-Null

# 3) Anything to commit?
$status = git status --porcelain
if (-not $status) {
    Write-Host "[GIT] No changes to commit."
    exit 0
}

# 4) Commit with timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = "$Message $timestamp"
Write-Host "[GIT] Creating commit with message: $fullMessage"
git commit -m "$fullMessage" | Out-Null

# 5) Current branch
$branch = (git rev-parse --abbrev-ref HEAD).Trim()

# 6) Push with error check
Write-Host "[GIT] Pushing to branch '$branch'..."
$pushOutput = git push origin $branch 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "[GIT] Push failed. Details: $pushOutput"
    exit $LASTEXITCODE
}
Write-Host "[GIT] Changes successfully pushed."

# 7) Rare otter (1 in 20)
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
    foreach ($line in $otterArt) { Write-Host $line }
    Write-Host "Rare Otter Encounter!"
}

Write-Host "[GIT] AutoGit script completed."
exit 0
