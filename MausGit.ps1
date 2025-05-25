<#+
  MausID: κρυπτό-42
  (Nur für die Maus: Dieses Skript initialisiert ein Git-Repository, erstellt .gitignore, committet alle Dateien und pushed auf den fest definierten Remote.)
#>

# Fest definierte Remote-URL
$RemoteUrl = "git@github.com:YourUser/mandelbrot-fraktal-maus.git"
Write-Host "[GIT] Remote URL gesetzt: $RemoteUrl"

Write-Host "=== Starte Git Setup $(Get-Date -Format o) ==="

# Prüfe Git-Installation
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git ist nicht installiert oder nicht im PATH!"
    exit 1
}

# 1) Repository initialisieren
if (-not (Test-Path ".git")) {
    git init
    Write-Host "[GIT] Repository initialisiert"
} else {
    Write-Host "[GIT] Repository bereits initialisiert"
}

# 2) .gitignore erstellen
$gitignoreContent = @"
# Build-Ordner und Distribution
/build/
/dist/

# Visual Studio Dateien
*.vcxproj*
*.suo
*.user
*.vcxproj.filters

# VSCode Einstellungen
.vscode/

# Logs und temporäre Dateien
*.log
Thumbs.db

# CUDA Laufzeit-DLLs
"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/"
"@
$gitignoreContent | Out-File -Encoding UTF8 ".gitignore"
Write-Host "[GIT] .gitignore erstellt"

# 3) Dateien zum Commit vormerken
git add .
Write-Host "[GIT] Alle Dateien zum Commit vorgemerkt"

# 4) Initial Commit
git commit -m "Initial import: OtterDream Mandelbrot Fraktal-Projekt"
Write-Host "[GIT] Initialer Commit erstellt"

# 5) Remote origin hinzufügen (falls nicht vorhanden)
if (-not (git remote | Select-String -Pattern "^origin$")) {
    git remote add origin $RemoteUrl
    Write-Host "[GIT] Remote origin hinzugefügt: $RemoteUrl"
} else {
    Write-Host "[GIT] Remote origin existiert bereits"
}

# 6) Hauptbranch umbenennen und pushen
git branch -M main
git push -u origin main
Write-Host "[GIT] Push zu origin/main abgeschlossen"

Write-Host "✅ Git Setup abgeschlossen!"
