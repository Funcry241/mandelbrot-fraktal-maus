<#
  MausSecret: σείριος-7
  Zweck: Löscht temporäre Build-, Debug- und Log-Dateien im Mandelbrot-Projekt.
  🐭 Prinzip: Keine .cu/.cpp/.hpp/.h/.toml/.md/.ps1-Dateien werden angefasst. Nur echter Müll fliegt raus.
  Hinweis: Debug-Modus via $dryRun = $true aktivierbar.
#>

$ErrorActionPreference = 'Stop'
$dryRun = $false  # 🐭 Debug-Modus: true = zeigt nur an, löscht aber nicht

Write-Host "`n--- MausDelete gestartet ---`n"

# 🎯 Zieldefinition: temporäre Dateierweiterungen, Dateinamenmuster, Build-Ordner
$fileExtensions   = @('.obj', '.o', '.ilk', '.pdb', '.log', '.tmp', '.tlog')
$filenamePatterns = @('CMakeCache.txt', 'CMakeGenerate.stamp', '*.VC.db', '*~')
$folderNames = @('CMakeFiles', 'Debug', 'Release', 'x64', '.vs', '.idea')  # build raus
$excludedFolders  = @('dist')  # 🛡️ Diese Ordner werden NIE gelöscht (aber Inhalt ggf. gesondert behandelt)

# 🔍 Rekursiver Suchlauf ab Projektwurzel
$allItems = Get-ChildItem -Recurse -Force -ErrorAction SilentlyContinue

foreach ($item in $allItems) {
    try {
        if ($item.PSIsContainer -and ($excludedFolders -contains $item.Name)) {
            continue  # dist/ etc. werden hier grundsätzlich übersprungen
        }

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
                Write-Host "  (DRY RUN) Würde löschen: $($item.FullName)"
            } else {
                Remove-Item $item.FullName -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "  Entfernt: $($item.FullName)"
            }
        }
    } catch {
        Write-Warning "  Fehler beim Löschen: $($item.FullName)"
    }
}

# 🔧 Spezialfall: dist/-Inhalt leeren, aber Ordner behalten
$distPath = Join-Path -Path $PSScriptRoot -ChildPath "dist"
if (Test-Path $distPath) {
    Get-ChildItem -Path $distPath -Force | Where-Object {
        $_.Name
    } | ForEach-Object {
        try {
            if ($dryRun) {
                Write-Host "  (DRY RUN) Würde in dist/ löschen: $($_.FullName)"
            } else {
                Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "  Entfernt aus dist/: $($_.FullName)"
            }
        } catch {
            Write-Warning "  Fehler beim Löschen in dist/: $($_.FullName)"
        }
    }
}

Write-Host "`n--- MausDelete abgeschlossen ---"
exit 0
