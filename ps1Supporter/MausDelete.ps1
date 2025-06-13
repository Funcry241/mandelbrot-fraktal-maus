<#
  MausSecret: σείριος-7
  Dieses Skript löscht temporäre Build-, Debug- und Log-Dateien im Mandelbrot-Projekt.
  Optimiert: Eine rekursive Traversierung, Prüfung per Datei- und Ordnername.
  Nur "Dreck" wird entfernt – Sourcecode bleibt sicher!
#>

$ErrorActionPreference = 'Stop'
Write-Host "– 🚮 MausDelete startet –"

# 🔍 Dateierweiterungen, die gelöscht werden sollen
$fileExtensions = @('.obj', '.o', '.ilk', '.pdb', '.log', '.tmp')

# 🔍 Spezifische Dateinamenmuster (mit Wildcards)
$filenamePatterns = @('CMakeCache.txt', '*~')

# 🗂️ Ordnernamen, die gelöscht werden sollen
$folderNames = @('CMakeFiles', 'build')

# 📁 Alle Dateien & Verzeichnisse im Projekt durchsuchen (rekursiv)
$allItems = Get-ChildItem -Recurse -Force

foreach ($item in $allItems) {
    try {
        $shouldDelete = $false

        # 📄 Prüfung für Dateien
        if (-not $item.PSIsContainer) {
            if ($fileExtensions -contains $item.Extension) {
                $shouldDelete = $true
            } elseif ($filenamePatterns | Where-Object { $item.Name -like $_ }) {
                $shouldDelete = $true
            }
        }

        # 📁 Prüfung für Ordner
        if ($item.PSIsContainer -and ($folderNames -contains $item.Name)) {
            $shouldDelete = $true
        }

        # 🚮 Löschung, wenn markiert
        if ($shouldDelete) {
            Remove-Item $item.FullName -Recurse -Force
            Write-Host "  Entfernt: $($item.FullName)"
        }
    } catch {
        Write-Warning "  Fehler beim Löschen: $($item.FullName)"
    }
}

Write-Host "– ✅ MausDelete abgeschlossen –"
exit 0
