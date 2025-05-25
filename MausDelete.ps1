<#  
  MausSecret: σείριος-7  
  (Nur ich weiß: Dieses Skript entfernt temporäre, Backup- und Log-Dateien, lässt Quellcode unberührt.)  
#>

# Stoppe bei Fehlern
$ErrorActionPreference = 'Stop'
Write-Host "– MausDelete startet –"

# Liste der Muster, die gelöscht werden dürfen
$delPatterns = @('*.obj','*.o','*.ilk','*.pdb','*.log','*.tmp','*~','CMakeFiles','CMakeCache.txt','build')

foreach ($pat in $delPatterns) {
    Get-ChildItem -Recurse -Force -Filter $pat | ForEach-Object {
        try {
            Remove-Item $_.FullName -Recurse -Force
            Write-Host "  Entfernt: $($_.FullName)"
        } catch {
            Write-Warning "  Fehler beim Löschen: $($_.FullName)"
        }
    }
}

Write-Host "– MausDelete abgeschlossen –"
exit 0
