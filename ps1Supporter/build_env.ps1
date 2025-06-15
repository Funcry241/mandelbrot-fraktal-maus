# 🧠 build_env.ps1 – Initialisiert MSVC-Umgebung für CUDA + cl.exe (automatisch)

# Ermittle Developer Command Prompt für VS 2022
$vcvarsall = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-Not (Test-Path $vcvarsall)) {
    Write-Error "[ENV] vcvars64.bat nicht gefunden unter $vcvarsall"
    exit 1
}

# Führe vcvars64.bat im Hintergrund aus und übernehme alle gesetzten Variablen
$vcVars = cmd /c "\"$vcvarsall\" >nul && set"
foreach ($line in $vcVars) {
    $parts = $line -split '=', 2
    if ($parts.Length -eq 2) {
        [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
    }
}

Write-Host "[ENV] Visual Studio Build-Umgebung erfolgreich geladen." -ForegroundColor Green
