<#+
  MausID: κρυπτό-42
  (Nur für die Maus: Dieses Skript initialisiert ein Git-Repo, erstellt .gitignore, commitet und pusht nach remote.)
#>

param(
    [string]$RemoteUrl = $(Read-Host "Enter remote repository URL (z.B. git@github.com:User/repo.git)")
)

Write-Host "=== Starte Git Setup $(Get-Date -Format o) ==="

# Prüfe Git Installation
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git ist nicht installiert oder nicht im PATH!"
    exit 1
}

# 1) Repo initialisieren
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

# VSCode
.vscode/

# Logs und temporäre Dateien
*.log
Thumbs.db

# CUDA Toolkit Laufzeit-DLLs
C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/
"@
$gitignoreContent | Out-File -Encoding UTF8 ".gitignore"
Write-Host "[GIT] .gitignore erstellt"

# 3) Dateien zum Commit vormerken
git add .
Write-Host "[GIT] Alle Dateien zum Commit vorgemerkt"

# 4) Initial commit
git commit -m "Initial import: OtterDream Mandelbrot CUDA Project mit Build- und Deploy-Skripten"
Write-Host "[GIT] Initialer Commit erstellt"

# 5) Remote hinzufügen
if (-not (git remote | Select-String -Pattern "^origin$")) {
    git remote add origin $RemoteUrl
    Write-Host "[GIT] Remote origin hinzugefügt: $RemoteUrl"
} else {
    Write-Host "[GIT] Remote origin existiert bereits"
}

# 6) Branch umbenennen und pushen
git branch -M main
git push -u origin main
Write-Host "[GIT] Push zu origin/main abgeschlossen"

Write-Host "✅ Git Setup abgeschlossen!"
