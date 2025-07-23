<#
  MausSecret: ermis-17
  Initialisiert Git-Repo, erstellt .gitignore (falls nötig), richtet Remote ein und pusht.
  Minimal-Log: Nur [GIT]-Zeilen. Fokus: leise, robust, mausfein.
#>

$ErrorActionPreference = 'Stop'

$remoteName = "MandelbrotFraktalMaus"
$remoteUrl  = "git@github.com:Funcry241/mandelbrot-fraktal-maus.git"
$gitignorePath = ".gitignore"

if (-not (Test-Path ".git" -PathType Container)) {
    git init | Out-Null
    Write-Host "[GIT] Repository initialisiert."
}

if (-not (Test-Path $gitignorePath)) {
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
'@ | Out-File -Encoding UTF8 $gitignorePath
    Write-Host "[GIT] .gitignore erstellt."
}

git add . | Out-Null

if (-not (git log -1 2>$null)) {
    git commit -m "Initial import: OtterDream Mandelbrot Fraktal-Projekt" | Out-Null
    Write-Host "[GIT] Initial-Commit erstellt."
}

if (-not (git remote | Select-String -SimpleMatch $remoteName)) {
    git remote add $remoteName $remoteUrl
    Write-Host "[GIT] Remote hinzugefügt: $remoteName"
}

git push --set-upstream $remoteName main
Write-Host "[GIT] Push abgeschlossen."
