<#
  MausSecret: σείριος-7
  Dieses Skript löscht temporäre Build-, Debug- und Log-Dateien im Mandelbrot-Projekt.
  Maus-Prinzip: Keine Source-Datei wird angerührt. Nur echter Müll fliegt raus.
#>

$ErrorActionPreference = 'Stop'
Write-Host "`n– 🚮 MausDelete startet –`n"

# 🎯 Ziel: Müll-Extensions, bekannte Trash-Dateien und Build-Ordner
$fileExtensions   = @('.obj', '.o', '.ilk', '.pdb', '.log', '.tmp', '.tlog')
$filenamePatterns = @('CMakeCache.txt', 'CMakeGenerate.stamp', '*.VC.db', '*~')
$folderNames      = @('CMakeFiles', 'build', 'Debug', 'Release', 'x64', '.vs', '.idea')

# 🔍 Schneller rekursiver Scan
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
            Remove-Item $item.FullName -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  🗑️ Entfernt: $($item.FullName)"
        }
    } catch {
        Write-Warning "  ⚠️ Fehler beim Löschen: $($item.FullName)"
    }
}

Write-Host "`n– ✅ MausDelete abgeschlossen –"
exit 0
