#MAUS
<#
  MausSecret: ermis-17
  Initialisiert Git-Repo, erstellt .gitignore (falls nötig), richtet Remote ein und pusht.
  Leise [GIT]-Logs; pusht auch dist/frame_0100.bmp (100. Frame) trotz /dist Ignore.
#>

$ErrorActionPreference = 'Stop'

$remoteName   = "MandelbrotFraktalMaus"
$remoteUrl    = "git@github.com:Funcry241/mandelbrot-fraktal-maus.git"
$gitignore    = ".gitignore"
$capturePath  = "dist/frame_0100.bmp"

# Repo init
if (-not (Test-Path ".git" -PathType Container)) {
    git init | Out-Null
    Write-Host "[GIT] Repository initialisiert."
}

# .gitignore anlegen/ergänzen (mit Ausnahme fuer den Capture)
if (-not (Test-Path $gitignore)) {
@'
# Build-Ordner
/build/
/dist/
/x64/
/Debug/
/Release/
*.obj
*.lib
*.dll
*.exe
*.pdb
*.ilk
CMakeFiles/
CMakeCache.txt
cmake_install.cmake
*.log
*.tmp
*.tlog
Thumbs.db
*.vcxproj*
*.suo
*.user
*.VC.db
*.VC.opendb
.vscode/
.idea/
.DS_Store
vcpkg/
vcpkg_installed/

# Einzelne Freigabe: 100. Frame
!/dist/frame_0100.bmp
'@ | Out-File -Encoding UTF8 $gitignore
    Write-Host "[GIT] .gitignore erstellt."
} else {
    $gi = Get-Content -LiteralPath $gitignore -Raw
    if ($gi -notmatch [regex]::Escape("!/dist/frame_0100.bmp")) {
        Add-Content -LiteralPath $gitignore -Value "`n!/dist/frame_0100.bmp"
        Write-Host "[GIT] .gitignore-Ausnahme fuer dist/frame_0100.bmp ergänzt."
    }
}

# Dateien stagen (Capture ggf. forcen)
if (Test-Path $capturePath) {
    git add -f $capturePath | Out-Null
}
git add . | Out-Null

# Initial-Commit (falls noetig)
if (-not (git log -1 2>$null)) {
    git commit -m "Initial import: OtterDream Mandelbrot Fraktal-Projekt" | Out-Null
    Write-Host "[GIT] Initial-Commit erstellt."
}

# Remote setzen/angleichen
if (-not (git remote | Select-String -SimpleMatch $remoteName)) {
    git remote add $remoteName $remoteUrl
    Write-Host "[GIT] Remote hinzugefügt: $remoteName"
} else {
    $currentUrl = (git remote get-url $remoteName).Trim()
    if ($currentUrl -ne $remoteUrl) {
        git remote set-url $remoteName $remoteUrl | Out-Null
        Write-Host "[GIT] Remote-URL aktualisiert."
    }
}

# Branch sicherstellen -> main
$currentBranch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($currentBranch -ne "main") {
    git branch -M main | Out-Null
    Write-Host "[GIT] Branch nach 'main' umbenannt."
}

# Push (Upstream setzen, Fehler ausgeben)
Write-Host "[GIT] Push nach $remoteName/main..."
$pushOutput = git push --set-upstream $remoteName main 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "[GIT] Push fehlgeschlagen. Details: $pushOutput"
    exit $LASTEXITCODE
}
Write-Host "[GIT] Push abgeschlossen."
