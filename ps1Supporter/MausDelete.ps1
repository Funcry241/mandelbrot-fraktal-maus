<#
  MausSecret: σείριος-7
  Zweck: Löscht temporäre Build-, Debug- und Log-Dateien im Mandelbrot-Projekt.
  🐭 Prinzip: Keine .cu/.cpp/.hpp/.h/.toml/.md/.ps1-Dateien werden angefasst. Nur echter Müll fliegt raus.
  Hinweis: Debug-Modus via $dryRun = $true aktivierbar.
#>

$ErrorActionPreference = 'Stop'
$dryRun = $false  # 🐭 Debug-Modus: true = zeigt nur an, löscht aber nicht

Write-Host "`n[MausDelete] Starting cleanup..."

$fileExtensions   = @('.obj', '.o', '.ilk', '.pdb', '.log', '.tmp', '.tlog')
$filenamePatterns = @('CMakeCache.txt', 'CMakeGenerate.stamp', '*.VC.db', '*~')
$folderNames      = @('CMakeFiles', 'Debug', 'Release', 'x64', '.vs', '.idea')
$excludedFolders  = @('dist')

$allItems = Get-ChildItem -Recurse -Force -ErrorAction SilentlyContinue

[int]$countFiles = 0
[int]$countDirs  = 0
[int]$countDist  = 0

foreach ($item in $allItems) {
    try {
        if ($item.PSIsContainer -and ($excludedFolders -contains $item.Name)) {
            continue
        }

        $trash = $false
        if (-not $item.PSIsContainer) {
            $ext = $item.Extension.ToLowerInvariant()
            if ($fileExtensions -contains $ext -or ($filenamePatterns | Where-Object { $item.Name -like $_ })) {
                $trash = $true
                $countFiles++
            }
        }
        elseif ($folderNames -contains $item.Name) {
            $trash = $true
            $countDirs++
        }

        if ($trash) {
            if ($dryRun) {
                Write-Host "  [DryRun] $($item.FullName)"
            } else {
                Remove-Item $item.FullName -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

    } catch {
        Write-Warning "[MausDelete] Error deleting: $($item.FullName)"
    }
}

# 🔧 dist/-Ordner-Inhalte
$distPath = Join-Path -Path $PSScriptRoot -ChildPath "dist"
if (Test-Path $distPath) {
    Get-ChildItem -Path $distPath -Force | ForEach-Object {
        try {
            if ($dryRun) {
                Write-Host "  [DryRun:dist] $($_.FullName)"
            } else {
                Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
                $countDist++
            }
        } catch {
            Write-Warning "[MausDelete] Error in dist/: $($_.FullName)"
        }
    }
}

Write-Host "[MausDelete] Done. Files: $countFiles  Dirs: $countDirs  dist/: $countDist"
exit 0
