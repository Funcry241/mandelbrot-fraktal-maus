<#
  MausSecret: ermis-17
  Dieses Skript initialisiert ein Git-Repository, erzeugt bei Bedarf eine .gitignore,
  richtet den Remote „MandelbrotFraktalMaus“ ein und pusht auf GitHub.
  Fokus: robust, wiederholbar, mausfein.
#>

$ErrorActionPreference = 'Stop'
Write-Host "`n-- [MAUS-GIT] Initialisiere --`n"

# Konfiguration
$remoteName = "MandelbrotFraktalMaus"
$remoteUrl  = "git@github.com:Funcry241/mandelbrot-fraktal-maus.git"
$gitignorePath = ".gitignore"

# Repository initialisieren, falls nötig
if (-not (Test-Path ".git" -PathType Container)) {
    Write-Host "[GIT] Initialisiere Git-Repository..."
    git init | Out-Null
} else {
    Write-Host "[GIT] Git-Repository bereits vorhanden."
}

# .gitignore erzeugen (falls nicht vorhanden)
if (-not (Test-Path $gitignorePath)) {
    Write-Host "[GIT] Erzeuge .gitignore"
@'
# Build-Ordner
/build/
/build-vs/
/dist/
/x64/
/Debug/
/Release/

# Visual Studio Dateien
*.vcxproj*
*.suo
*.user
*.vcxproj.filters
*.VC.db
*.VC.opendb

# VSCode
.vscode/
!.vscode/c_cpp_properties.json

# Temporäre Dateien & Logs
*.log
*.tmp
*.tlog
Thumbs.db
Desktop.ini
*~

# vcpkg
vcpkg/
vcpkg_installed/

# Binary Output
*.obj
*.lib
*.dll
*.exe
*.pdb
*.ilk

# CMake-Artefakte
CMakeFiles/
CMakeCache.txt
cmake_install.cmake
Makefile

# IDE-Projekte (CLion, Rider, Xcode, JetBrains)
.idea/
.DS_Store
'@ | Out-File -Encoding UTF8 $gitignorePath
} else {
    Write-Host "[GIT] .gitignore existiert bereits."
}

# Git-Add & Commit (nur bei leerem Repo)
Write-Host "[GIT] Füge Dateien hinzu..."
git add . | Out-Null

$headExists = git rev-parse --verify HEAD 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[GIT] Erstelle Initial-Commit..."
    git commit -m "Initial import: OtterDream Mandelbrot Fraktal-Projekt" | Out-Null
} else {
    Write-Host "[GIT] Bereits ein Commit vorhanden."
}

# Remote hinzufügen, falls nicht vorhanden
$existingRemotes = git remote
if ($existingRemotes -notcontains $remoteName) {
    Write-Host "[GIT] Füge Remote '$remoteName' hinzu -> $remoteUrl"
    git remote add $remoteName $remoteUrl
} else {
    Write-Host "[GIT] Remote '$remoteName' existiert bereits."
}

# Push auf main (mit Upstream setzen)
Write-Host "[GIT] Pushe auf Branch 'main' zu '$remoteName'..."
git push --set-upstream $remoteName main

Write-Host "`n-- [MAUS-GIT] Abgeschlossen --`n"
exit 0
