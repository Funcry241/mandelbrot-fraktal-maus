<#
.SYNOPSIS
    MausGitAutoCommit.ps1
.DESCRIPTION
    Automatically commit and push all changes in the existing Git repository.
    Ensures the 100th-frame capture (dist/frame_0100.bmp) is included even if /dist is ignored.
    Robust push: tries SSH first; on 'Permission denied (publickey)' falls back to HTTPS automatically.
    ASCII-only logs.
#>

param(
    [string]$Message = "Auto-Commit: Changes OtterDream"
)

$ErrorActionPreference = 'Stop'

function Convert-SshToHttpsUrl([string]$url) {
    # Convert "git@github.com:User/Repo.git" -> "https://github.com/User/Repo.git"
    if ($url -match '^git@github\.com:(.+)$') {
        return "https://github.com/$($Matches[1])"
    }
    if ($url -match '^ssh://git@github\.com/(.+)$') {
        return "https://github.com/$($Matches[1])"
    }
    if ($url -match '^https://github\.com/.+$') {
        return $url
    }
    return $null
}

function Invoke-GitPush([string]$remote, [string]$branch) {
    Write-Host "[GIT] Pushing to '$remote/$branch'..."
    $out = git push $remote $branch 2>&1
    $code = $LASTEXITCODE
    return @{ Code = $code; Out = $out }
}

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
    # 7) Rare otter (still may appear even if nothing to commit? Keep it fun but deterministic -> no)
    Write-Host "[GIT] AutoGit script completed."
    exit 0
}

# 4) Commit with timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = "$Message $timestamp"
Write-Host "[GIT] Creating commit with message: $fullMessage"
git commit -m "$fullMessage" | Out-Null

# 5) Determine current branch
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if (-not $branch) {
    Write-Error "[GIT] Could not determine current branch."
    exit 1
}

# 6) Determine 'origin' URL
try {
    $originUrl = (git remote get-url origin).Trim()
} catch {
    Write-Error "[GIT] Remote 'origin' not found. Please add it: git remote add origin <ssh-or-https-url>"
    exit 1
}

Write-Host "[GIT] Remote 'origin' URL: $originUrl"

# 6a) Try SSH push first (if URL is SSH), fallback to HTTPS on publickey error
$oldUrl = $originUrl
$tryPush = Invoke-GitPush -remote "origin" -branch $branch

if ($tryPush.Code -ne 0) {
    $outStr = ($tryPush.Out | Out-String)
    $pubKeyError = $outStr -match "Permission denied \(publickey\)"
    $repoNotFound = $outStr -match "Repository not found" -or $outStr -match "Could not read from remote repository"

    if ($pubKeyError -or $repoNotFound) {
        $httpsUrl = Convert-SshToHttpsUrl $originUrl
        if (-not $httpsUrl) {
            Write-Error "[GIT] Push failed and HTTPS URL could not be derived from '$originUrl'. Details: $outStr"
            exit $tryPush.Code
        }
        Write-Host "[GIT] SSH failed ('publickey' or repo access). Switching 'origin' to HTTPS for retry..."
        git remote set-url origin $httpsUrl | Out-Null

        $retry = Invoke-GitPush -remote "origin" -branch $branch
        if ($retry.Code -ne 0) {
            # Restore original remote and fail
            git remote set-url origin $oldUrl | Out-Null
            Write-Error "[GIT] Push failed after HTTPS retry. Details: $($retry.Out | Out-String)"
            exit $retry.Code
        } else {
            Write-Host "[GIT] Push succeeded via HTTPS."
        }
    } else {
        Write-Error "[GIT] Push failed. Details: $outStr"
        exit $tryPush.Code
    }
} else {
    Write-Host "[GIT] Changes successfully pushed (SSH or HTTPS as configured)."
}

# 7) Rare otter (1 in 20) — MUST STAY 🦦
$rareChance = Get-Random -Minimum 1 -Maximum 21
if ($rareChance -eq 1) {
    $otterArt = @(
        "    .--.",
        "   /    \",
        "  /_    _\",
        " // \  / \\",
        " |\__/\\__|",
        " \    /\   /",
        "  '--'  '--'"
    )
    foreach ($line in $otterArt) { Write-Host $line }
    Write-Host "Rare Otter Encounter!"
}

Write-Host "[GIT] AutoGit script completed."
exit 0
