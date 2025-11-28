# Write-ZipFileSizes.ps1
# Zählt Zeilen aller relevanten Dateien im aktuellsten ZIP im out/-Verzeichnis
# und schreibt eine Tabelle nach out\FileSizes.txt.

param(
    # Basisverzeichnis für die Export-ZIPs.
    # Standard: "out" relativ zum Speicherort dieses Skripts.
    [string]$OutDir = (Join-Path -Path (Split-Path -Parent $PSCommandPath) -ChildPath 'out')
)

if (-not (Test-Path -Path $OutDir -PathType Container)) {
    Write-Error "Out directory '$OutDir' does not exist."
    exit 1
}

# Aktuellstes ZIP im out/-Verzeichnis suchen
$latestZip = Get-ChildItem -Path $OutDir -Filter '*.zip' -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latestZip) {
    Write-Error "No ZIP file found in '$OutDir'."
    exit 1
}

# Ziel-Datei innerhalb von out/
$outFile = Join-Path -Path $OutDir -ChildPath 'FileSizes.txt'

# Relevante (typisch textbasierte) Dateitypen, die wir zählen wollen.
$textExtensions = @(
    '.txt', '.md', '.json', '.toml', '.yaml', '.yml',
    '.rs', '.cpp', '.c', '.h', '.hpp', '.cu', '.cuh',
    '.ps1', '.sh', '.bat', '.ini', '.cfg', '.log'
)

# .NET-Typen für ZIP-Zugriff laden
Add-Type -AssemblyName System.IO.Compression.FileSystem

$results = @()

$archive = [System.IO.Compression.ZipFile]::OpenRead($latestZip.FullName)
try {
    foreach ($entry in $archive.Entries) {
        # Verzeichnisse überspringen
        if ([string]::IsNullOrEmpty($entry.Name)) {
            continue
        }

        # Nur relevante Extensions
        $ext = [System.IO.Path]::GetExtension($entry.FullName)
        if (-not $ext) { continue }
        $ext = $ext.ToLowerInvariant()
        if ($textExtensions -notcontains $ext) { continue }

        $lineCount = 0
        try {
            $stream = $entry.Open()
            $reader = New-Object System.IO.StreamReader($stream)

            while ($null -ne ($line = $reader.ReadLine())) {
                $lineCount++
            }

            $reader.Close()
            $stream.Close()
        }
        catch {
            $lineCount = 'N/A'
        }

        $results += [PSCustomObject]@{
            File  = $entry.FullName   # Pfad im ZIP
            Lines = $lineCount
        }
    }
}
finally {
    $archive.Dispose()
}

$results = $results | Sort-Object File

# Header + Tabelle erzeugen
$header = @()
$header += "# FileSizes.txt - line counts per file in latest ZIP export"
$header += "# Out directory : $OutDir"
$header += "# ZIP file      : $($latestZip.Name)"
$header += "# Generated at  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$header += ""

$table = $results | Format-Table -Property File, Lines -AutoSize | Out-String

$header + $table | Set-Content -Path $outFile -Encoding UTF8

"FileSizes written to: $outFile"
