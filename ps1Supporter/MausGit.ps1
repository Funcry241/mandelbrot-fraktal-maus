<#
.SYNOPSIS
    MausGit.ps1
.DESCRIPTION
    Dieses PowerShell-Skript initialisiert ein Git-Repository im Projektverzeichnis,
    legt eine .gitignore an, fügt (falls noch nicht vorhanden) den Remote „MandelbrotFraktalMaus“ hinzu,
    erstellt einen ersten Commit und pusht den Branch „main“ zu GitHub.
.NOTES
    Stelle sicher, dass Du die Variable $remoteUrl unten mit Deiner tatsächlichen Repository-URL ersetzt.
#>

# ------------------------------------------------------------
# Konfiguration: Name und URL des Git-Remotes
$remoteName = "MandelbrotFraktalMaus"
# Ersetze den folgenden Wert durch Deine eigene GitHub-SSH-URL:
$remoteUrl  = "git@github.com:Funcry241/mandelbrot-fraktal-maus.git"

# ------------------------------------------------------------
# 1) Repository initialisieren, falls noch kein .git existiert
if (-not (Test-Path -Path ".git" -PathType Container)) {
    Write-Host "[GIT] Repository initialisieren..."
    git init
} else {
    Write-Host "[GIT] Repository bereits initialisiert"
}

# ------------------------------------------------------------
# 2) .gitignore erstellen, wenn noch nicht vorhanden
$gitignorePath = ".gitignore"
if (-not (Test-Path $gitignorePath)) {
    Write-Host "[GIT] .gitignore erstellen"
    @"
# ----------------------------------------------------------------------
# Build-Ordner und Distribution
/build/
/dist/

# ----------------------------------------------------------------------
# Visual Studio-Dateien (Projekte & Lösungen)
/*.vcxproj*
*.suo
*.user
*.vcxproj.filters

# ----------------------------------------------------------------------
# VSCode-Einstellungen
.vscode/

# ----------------------------------------------------------------------
# Temporäre Dateien und Logs
*.log
Thumbs.db

# ----------------------------------------------------------------------
# vcpkg-Ordner (Abhängigkeiten & Pakete)
vcpkg/

# ----------------------------------------------------------------------
# (Optional) Weitere Artefakte, die üblicherweise nicht versioniert werden
# Object- & Bibliotheksdateien
*.obj
*.lib
*.dll
*.exe
*.pdb

# ----------------------------------------------------------------------
# (Optional) macOS-spezifisch
.DS_Store

# ----------------------------------------------------------------------
# (Optional) Linux-spezifisch
*~
"@ | Out-File -Encoding UTF8 $gitignorePath
} else {
    Write-Host "[GIT] .gitignore existiert bereits"
}

# ------------------------------------------------------------
# 3) Dateien zum Commit vormerken und Commit erzeugen
Write-Host "[GIT] Dateien zum Commit vormerken"
git add .

# Prüfe, ob bereits ein Commit existiert
$hasCommits = git rev-parse --verify HEAD 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[GIT] Erstelle Initial-Commit"
    git commit -m "Initial import: OtterDream Mandelbrot Fraktal-Projekt"
} else {
    Write-Host "[GIT] Es existiert bereits mindestens ein Commit"
}

# ------------------------------------------------------------
# 4) Remote hinzufügen, falls noch nicht vorhanden
$existingRemotes = git remote
if ($existingRemotes -notcontains $remoteName) {
    Write-Host "[GIT] Remote '$remoteName' hinzufügen: $remoteUrl"
    git remote add $remoteName $remoteUrl
} else {
    Write-Host "[GIT] Remote '$remoteName' existiert bereits"
}

# ------------------------------------------------------------
# 5) Push zu GitHub (Branch main)
Write-Host "[GIT] Pushe Branch 'main' zu '$remoteName'"
# Wenn der lokale Branch main noch keinen Tracking-Branch hat, wird dieser gesetzt
git push --set-upstream $remoteName main

Write-Host "[GIT] Push abgeschlossen"
