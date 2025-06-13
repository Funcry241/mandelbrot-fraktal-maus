<#
  MausSecret: ερμής-17
  Dieses Skript initialisiert ein Git-Repository, erstellt ggf. eine .gitignore,
  fügt Remote „MandelbrotFraktalMaus“ hinzu und pusht auf GitHub.
  🐭 Kompakt, verständlich, elegant.
#>

$ErrorActionPreference = 'Stop'
Write-Host "– 🐭 MausGit startet –"

# 🔧 Konfiguration
$remoteName = "MandelbrotFraktalMaus"
$remoteUrl  = "git@github.com:Funcry241/mandelbrot-fraktal-maus.git"
$gitignorePath = ".gitignore"

# 📁 Repository initialisieren, falls nötig
if (-not (Test-Path ".git" -PathType Container)) {
    Write-Host "[GIT] Initialisiere Repository..."
    git init
} else {
    Write-Host "[GIT] Repository bereits vorhanden."
}

# 📝 .gitignore erzeugen (wenn fehlt)
if (-not (Test-Path $gitignorePath)) {
    Write-Host "[GIT] Erzeuge .gitignore"
@'
# Build
/build/
/dist/
/build-vs/

# Visual Studio
*.vcxproj*
*.suo
*.user
*.vcxproj.filters

# VSCode
.vscode/
!.vscode/c_cpp_properties.json

# Temp & Logs
*.log
Thumbs.db
Desktop.ini
*~

# vcpkg
vcpkg/
vcpkg_installed/

# Output
*.obj
*.lib
*.dll
*.exe
*.pdb

# CMake
CMakeFiles/
CMakeCache.txt
cmake_install.cmake
Makefile

# IDEs
.idea/
.DS_Store
'@ | Out-File -Encoding UTF8 $gitignorePath
} else {
    Write-Host "[GIT] .gitignore existiert bereits."
}

# 📦 Dateien hinzufügen und ersten Commit erstellen (falls nötig)
Write-Host "[GIT] Dateien zum Commit vormerken..."
git add .

# Nur committen, wenn noch kein Commit existiert
git rev-parse --verify HEAD 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[GIT] Erstelle Initial-Commit..."
    git commit -m "Initial import: OtterDream Mandelbrot Fraktal-Projekt"
} else {
    Write-Host "[GIT] Bereits ein Commit vorhanden."
}

# 🌐 Remote hinzufügen, wenn fehlt
if (-not (git remote | Where-Object { $_ -eq $remoteName })) {
    Write-Host "[GIT] Füge Remote '$remoteName' hinzu → $remoteUrl"
    git remote add $remoteName $remoteUrl
} else {
    Write-Host "[GIT] Remote '$remoteName' existiert bereits."
}

# ⬆️ Push auf main (mit Upstream setzen)
Write-Host "[GIT] Pushe auf Branch 'main' zu '$remoteName'..."
git push --set-upstream $remoteName main

Write-Host "– ✅ MausGit abgeschlossen –"
exit 0
