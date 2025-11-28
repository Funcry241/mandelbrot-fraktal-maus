# count_src_lines.ps1
param(
    # Standard: ./src relativ zum Speicherort des Skripts
    [string]$SrcDir = (Join-Path $PSScriptRoot 'src')
)

Get-ChildItem -Path $SrcDir -Recurse -File |
    ForEach-Object {
        $lineCount = (Get-Content -Path $_.FullName -ErrorAction SilentlyContinue |
                      Measure-Object -Line).Lines

        [PSCustomObject]@{
            Lines = $lineCount
            Path  = $_.FullName
        }
    } |
    Sort-Object -Property Lines -Descending |
    Format-Table -AutoSize
