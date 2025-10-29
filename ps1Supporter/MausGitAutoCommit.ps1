##### Otter: Auto-commit/push with pre-push pull --rebase --autostash and non-FF retry; SSH->HTTPS fallback; ASCII-only logs; no Start-Process.
##### Schneefuchs: Fix CRLF warning abort; HEAD-safe branch detection; unified git wrapper; approved verb rename Normalize->Resolve.
##### Maus: Zero interaction on happy path; clear exit codes; deterministic output; tidy logs.
##### Datei: ps1Supporter/MausGitAutoCommit.ps1

param(
    [string]$Message = "Auto-Commit: Changes OtterDream"
)

# Keep native tools from turning stderr into terminating errors
$ErrorActionPreference = 'Continue'
$global:PSNativeCommandUseErrorActionPreference = $false

function Convert-SshToHttpsUrl([string]$url) {
    if ($url -match '^git@github\.com:(.+)$')      { return "https://github.com/$($Matches[1])" }
    if ($url -match '^ssh://git@github\.com/(.+)$') { return "https://github.com/$($Matches[1])" }
    if ($url -match '^https://github\.com/.+$')     { return $url }
    return $null
}

# Treat non-fatal EOL warnings (CRLF/LF) as success.
function Resolve-GitResult([int]$Code, [string]$Out) {
    if ($Code -ne 0) {
        $eolWarn =
            ($Out -match '(?i)\bCRLF will be replaced by LF\b') -or
            ($Out -match '(?i)\bLF will be replaced by CRLF\b') -or
            ($Out -match '(?i)\bwarning:\s+in the working copy of\b.*\bwill be replaced\b')
        if ($eolWarn) { return @{ Code = 0; Out = $Out } }
    }
    return @{ Code = $Code; Out = $Out }
}

# Generic git runner: returns @{ Code=int; Out=string }.
function Invoke-Git {
    param([string[]]$GitArgs)
    $safe = @($GitArgs | Where-Object { $_ -ne $null -and $_ -ne "" })
    if (-not $safe -or $safe.Count -eq 0) { return @{ Code = 1; Out = "[GIT] empty argument list" } }
    $out  = & git @safe 2>&1
    $code = $LASTEXITCODE
    return (Resolve-GitResult -Code $code -Out ($out -join "`n"))
}

function Invoke-GitPush([string]$remote, [string]$branch) {
    Write-Host "[GIT] Pushing to '$remote/$branch'..."
    return Invoke-Git @('push', $remote, $branch)
}

function Get-CurrentBranch {
    # 1) Preferred (git>=2.22)
    $a = Invoke-Git @('branch','--show-current')
    if ($a.Code -eq 0 -and $a.Out.Trim()) { return $a.Out.Trim() }

    # 2) symbolic-ref HEAD
    $b = Invoke-Git @('symbolic-ref','--quiet','--short','HEAD')
    if ($b.Code -eq 0) {
        $val = $b.Out.Trim()
        if ($val -and $val -ne 'HEAD') { return $val }
    }

    # 3) origin/HEAD -> extract default branch (e.g. origin/main)
    $c = Invoke-Git @('symbolic-ref','--quiet','--short','refs/remotes/origin/HEAD')
    if ($c.Code -eq 0 -and $c.Out.Trim()) {
        $parts = $c.Out.Trim() -split '/'
        if ($parts.Length -ge 2) { return $parts[-1] }
    }

    return $null
}

# 1) Repo present?
if (-not (Test-Path ".git")) { Write-Host "[GIT] No Git repository found!"; exit 1 }

# 2) Stage changes (avoid CRLF warning aborts)
Write-Host "[GIT] Staging changes..."
$addAll = Invoke-Git @('-c','core.safecrlf=false','add','-A')
if ($addAll.Code -ne 0) { Write-Host $addAll.Out; exit $addAll.Code }

# 3) Anything to commit?
$status = Invoke-Git @('status','--porcelain')
if ($status.Code -ne 0) { Write-Host $status.Out; exit $status.Code }
if ([string]::IsNullOrWhiteSpace($status.Out)) {
    Write-Host "[GIT] No changes to commit."
    Write-Host "[GIT] AutoGit script completed."
    exit 0
}

# 4) Commit with timestamp
$timestamp   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$fullMessage = "$Message $timestamp"
Write-Host "[GIT] Creating commit with message: $fullMessage"
$commit = Invoke-Git @('commit','-m', $fullMessage)
if ($commit.Code -ne 0) { Write-Host $commit.Out; exit $commit.Code }

# 5) Current branch (robust)
$branch = Get-CurrentBranch
if (-not $branch) {
    Write-Host "[GIT] Detached HEAD or unknown branch -- skipping pull/push (commit is local)."
    Write-Host "[GIT] AutoGit script completed."
    exit 0
}

# 6) Ensure 'origin' exists
$originUrlOut = Invoke-Git @('remote','get-url','origin')
if ($originUrlOut.Code -ne 0) {
    Write-Host "[GIT] Remote 'origin' not found. Add it: git remote add origin <url>"
    exit 1
}
$originUrl = $originUrlOut.Out.Trim()
Write-Host "[GIT] Remote 'origin' URL: $originUrl"

# 7) Pre-push sync: fetch + pull --rebase --autostash
Write-Host "[GIT] Sync: fetching from 'origin'..."
$fetch = Invoke-Git @('fetch','origin')
if ($fetch.Code -ne 0) { Write-Host "[GIT] Fetch failed.`n$($fetch.Out)"; exit $fetch.Code }

Write-Host "[GIT] Sync: pulling (rebase, autostash) from origin/$branch..."
$pull = Invoke-Git @('pull','--rebase','--autostash','origin', $branch)
if ($pull.Code -ne 0) {
    if ($pull.Out -match 'CONFLICT|could not apply|Resolve all conflicts') {
        Write-Host "[GIT] Rebase conflict detected. Aborting rebase..."
        [void](Invoke-Git @('rebase','--abort'))
        Write-Host "[GIT] Pull (rebase) failed due to conflicts. Please resolve locally and re-run."
    } else {
        Write-Host "[GIT] Pull (rebase) failed."
    }
    Write-Host $pull.Out
    exit $pull.Code
}

# 8) Push (first try)
Write-Host "[GIT] Pushing to 'origin/$branch'..."
$result = Invoke-Git @('push','origin', $branch)
if ($result.Code -ne 0) {
    $out = $result.Out
    $sshAuthErr = ($out -match 'Permission denied \(publickey\)|Could not read from remote repository|Repository not found|Authentication failed')
    $nonFFErr   = ($out -match 'fetch first|non-fast-forward|remote contains work that you do not have locally|tip of your current branch is behind')

    if ($sshAuthErr) {
        $httpsUrl = Convert-SshToHttpsUrl $originUrl
        if ($httpsUrl) {
            Write-Host "[GIT] SSH push failed. Switching 'origin' to HTTPS and retrying..."
            $set1 = Invoke-Git @('remote','set-url','origin', $httpsUrl)
            if ($set1.Code -ne 0) { Write-Host $set1.Out; exit $set1.Code }
            $retry = Invoke-Git @('push','origin', $branch)
            if ($retry.Code -ne 0) {
                Write-Host "[GIT] Push failed after HTTPS retry."
                Write-Host $retry.Out
                [void](Invoke-Git @('remote','set-url','origin', $originUrl))
                exit $retry.Code
            } else {
                Write-Host "[GIT] Push succeeded via HTTPS."
            }
        } else {
            Write-Host "[GIT] Push failed and HTTPS URL could not be derived from '$originUrl'."
            Write-Host $out
            exit $result.Code
        }
    }
    elseif ($nonFFErr) {
        Write-Host "[GIT] Remote ahead. Syncing (rebase, autostash) and retrying push..."
        $pull2 = Invoke-Git @('pull','--rebase','--autostash','origin', $branch)
        if ($pull2.Code -ne 0) {
            if ($pull2.Out -match 'CONFLICT|could not apply|Resolve all conflicts') {
                Write-Host "[GIT] Rebase conflict detected on retry. Aborting rebase..."
                [void](Invoke-Git @('rebase','--abort'))
            }
            Write-Host "[GIT] Retry pull (rebase) failed."
            Write-Host $pull2.Out
            exit $pull2.Code
        }
        $retry2 = Invoke-Git @('push','origin', $branch)
        if ($retry2.Code -ne 0) {
            Write-Host "[GIT] Push still failing after sync."
            Write-Host $retry2.Out
            exit $retry2.Code
        } else {
            Write-Host "[GIT] Changes successfully pushed after sync."
        }
    }
    else {
        Write-Host "[GIT] Push failed."
        Write-Host $out
        exit $result.Code
    }
} else {
    Write-Host "[GIT] Changes successfully pushed."
}

# 9) Rare otter (1 in 20)
$rareChance = Get-Random -Minimum 1 -Maximum 21
if ($rareChance -eq 1) {
@'
    .--.
   /    \
  /_    _\
 // \  / \\
 |\__/\\__|
 \    /\   /
  '--'  '--'
'@ -split "`n" | ForEach-Object { Write-Host $_ }
    Write-Host "Rare Otter Encounter!"
}

Write-Host "[GIT] AutoGit script completed."
exit 0
