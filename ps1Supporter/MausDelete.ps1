<#
  MausSecret: σείριος-7
  Zweck: Löscht temporäre Build-, Debug- und Log-Dateien im Mandelbrot-Projekt.
  🐭 Prinzip: Keine .cu/.cpp/.hpp/.h/.toml/.md/.ps1-Dateien werden angefasst. Nur echter Müll fliegt raus.
  Hinweis: Debug-Modus via $dryRun = $true aktivierbar.
#>

$ErrorActionPreference = 'Stop'
$dryRun = $false  # 🐭 Debug-Modus: true = zeigt nur an, löscht aber nicht

Write-Host "`n– 🚮 MausDelete startet –`n"

# 🎯 Zieldefinition: temporäre Dateierweiterungen, Dateinamenmuster, Build-Ordner
$fileExtensions   = @('.obj', '.o', '.ilk', '.pdb', '.log', '.tmp', '.tlog')
$filenamePatterns = @('CMakeCache.txt', 'CMakeGenerate.stamp', '*.VC.db', '*~')
$folderNames      = @('CMakeFiles', 'build', 'Debug', 'Release', 'x64', '.vs', '.idea')

# 🔍 Rekursiver Suchlauf ab Projektwurzel
$allItems = Get-ChildItem -Recurse -Force -ErrorAction SilentlyContinue

foreach ($item in $allItems) {
    try {
        $isTrash = $false

        if (-not $item.PSIsContainer) {
            $ext = $item.Extension.ToLowerInvariant()
            if ($fileExtensions -contains $ext) {
                $isTrash = $true
            }
            elseif ($filenamePatterns | Where-Object { $item.Name -like $_ }) {
                $isTrash = $true
            }
        }
        elseif ($folderNames -contains $item.Name) {
            $isTrash = $true
        }

        if ($isTrash) {
            if ($dryRun) {
                Write-Host "  💡 (DRY) Würde löschen: $($item.FullName)"
            } else {
                Remove-Item $item.FullName -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "  🗑️ Entfernt: $($item.FullName)"
            }
        }
    } catch {
        Write-Warning "  ⚠️ Fehler beim Löschen: $($item.FullName)"
    }
}

Write-Host "`n– ✅ MausDelete abgeschlossen –"
exit 0
