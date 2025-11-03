///// Otter: Branch-Orchestrator – Build → packe relevante Quellen (ZIP) → Autogit Push → Summary.
///// Schneefuchs: Pack nur unter Windows (PowerShell Compress-Archive), robustes Filtering; keine Extra-Crates.
///// Maus: ASCII-Logs, env-Overrides (OTTER_*), keine Magie; Default-Branch „wupp“.
///// Datei: rust/otter_proc/src/ops_branch.rs

use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::artifact::find_artifact;
use crate::commands;
use crate::runner::runner_term;
use crate::summary;
use crate::utils::epoch_ms;

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_opt_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|s| s.parse::<u32>().ok())
}

fn current_branch_or_default(root: &Path) -> String {
    crate::vcs::git_current_branch(root)
        .or_else(|| std::env::var("OTTER_BRANCH").ok())
        .unwrap_or_else(|| "wupp".to_string())
}

#[cfg(windows)]
fn powershell_exe() -> &'static str { "powershell.exe" }
#[cfg(not(windows))]
fn powershell_exe() -> &'static str { "pwsh" } // Fallback (wird unten dennoch mit Err quittiert)

/// Packe relevante Projektdateien als ZIP in ./dist.
/// – keine Build- oder Cache-Verzeichnisse
/// – gängige Quell-/Build-Inputs (Rust/CUDA/CMake/JSON/PS1/BAT/GLSL)
#[cfg(windows)]
fn pack_sources_with_powershell(project_root: &Path) -> io::Result<PathBuf> {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let root = project_root.canonicalize().unwrap_or_else(|_| project_root.to_path_buf());
    let dist = root.join("dist");
    let _ = fs::create_dir_all(&dist);

    // Zeitstempel (YYYYMMDD_HHMMSS)
    let ts = {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // simple UTC-ish formatting ohne chrono
        // (YYYYMMDD_HHMMSS) aus UNIX Sekunden
        // wir lassen PowerShell die hübsche Formatierung übernehmen → liefert den finalen Pfad zurück
        now.to_string()
    };

    // Temporäres PS-Skript schreiben
    let script = r#"$ErrorActionPreference='Stop'
param([string]$Root)

$root = Resolve-Path -LiteralPath $Root
$dest = Join-Path $root 'dist'
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$zip = Join-Path $dest ("otter_sources_{0}.zip" -f $ts)

# Einschluss nach Endung / Namen
$inclExt = @(
  '.rs','.toml','.lock',
  '.cu','.cuh','.c','.cpp','.cxx','.hpp','.h','.inl',
  '.cmake','.glsl','.vert','.frag','.comp','.geom',
  '.json','.md','.ps1','.bat'
)
$inclNames = @(
  'CMakeLists.txt','CMakePresets.json','CTestConfig.cmake',
  '.gitignore','.editorconfig','vcpkg.json','vcpkg-configuration.json'
)

# Ausschluss-Verzeichnisse (Teilpfade)
$exDirs = @(
  '\build','\build-','\dist',
  '\vcpkg','\vcpkg_installed','\vcpkg_downloads','\vcpkg_buildtrees','\vcpkg_packages','\vcpkg_cache',
  '\target','\rust\otter_proc\target',
  '\.git','\.vs','\.vscode'
)

$files = Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object {
  $p = $_.FullName
  foreach($ex in $exDirs){ if($p -like ('*'+$ex+'*')){ return $false } }

  $ext = [System.IO.Path]::GetExtension($p).ToLower()
  if($inclExt -contains $ext){ return $true }
  if($inclNames -contains $_.Name){ return $true }
  return $false
}

if(-not $files -or $files.Count -eq 0){
  throw 'No files matched for packaging.'
}

Compress-Archive -Path ($files | Select-Object -Expand FullName) -DestinationPath $zip -CompressionLevel Optimal -Force
Write-Output $zip
"#;

    // Tempfile anlegen
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("otter_pack_{}.ps1", ts));
    std::fs::write(&tmp, script)?;

    // PowerShell starten
    let output = Command::new(powershell_exe())
        .args([
            "-NoProfile",
            "-ExecutionPolicy","Bypass",
            "-File", tmp.to_string_lossy().as_ref(),
            "-Root", root.to_string_lossy().as_ref(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()?;

    // Tempfile aufräumen (best effort)
    let _ = std::fs::remove_file(&tmp);

    if !output.status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "pack (PowerShell) failed"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        return Err(io::Error::new(io::ErrorKind::Other, "pack produced no output path"));
    }
    Ok(PathBuf::from(stdout))
}

#[cfg(not(windows))]
fn pack_sources_with_powershell(_project_root: &Path) -> io::Result<PathBuf> {
    Err(io::Error::new(
        io::ErrorKind::Other,
        "pack only supported on Windows (PowerShell Compress-Archive)",
    ))
}

/// Öffentlicher Einstieg für den Branch-Pfad.
/// Ablauf: Build (Full) → Pack → Autogit push → Summary.
/// Rückgabe: Prozess-Exitcode.
pub fn exec(root: &Path) -> i32 {
    runner_term::enable_ansi();

    let start_ms = epoch_ms();
    let cfg   = env_str("OTTER_CFG", "RelWithDebInfo");
    let cp    = std::env::var("OTTER_CONFIGURE_PRESET").ok();
    let bp    = std::env::var("OTTER_BUILD_PRESET").ok();
    let par   = env_opt_u32("OTTER_PARALLEL");
    let br    = current_branch_or_default(root);

    crate::runner::runner_term::out_info(
        "RUNNER",
        &format!("branch-mode start ts_ms={} root={} cfg={} branch={}",
                 start_ms, crate::prockit::display_path(root), cfg, br),
    );

    // 1) Full Build fahren
    let build_rc = commands::full::run(
        root,
        &cfg,
        cp.as_deref(),
        bp.as_deref(),
        par,
    );
    let (ok_build, code_build) = match build_rc {
        Ok(code) => (code == 0, code),
        Err(e) => {
            eprintln!("[ERROR] full build error: {}", e);
            return 1;
        }
    };
    if !ok_build {
        // Früh zusammenfassen (Build fehlgeschlagen)
        let end_ms = epoch_ms();
        let artifact = find_artifact(root);
        summary::print_end_summary(summary::EndSummary {
            success: false,
            exit_code: code_build,
            started_ms: start_ms,
            elapsed_ms: end_ms.saturating_sub(start_ms),
            artifact_path: artifact.map(|p| p.to_string_lossy().to_string()),
            commit_short: crate::vcs::git_short_hash(root),
            commit_branch: Some(format!("origin/{}", br)),
            autogit_pushed: false,
            notes: vec!["build failed before packing".into()],
        });
        return code_build;
    }

    // 2) Quellen packen (nur Windows)
    let zip_path = match pack_sources_with_powershell(root) {
        Ok(p) => {
            crate::runner::runner_term::out_info("RUNNER",
                &format!("packed sources: {}", p.display()));
            Some(p)
        }
        Err(e) => {
            crate::runner::runner_term::out_warn("RUNNER",
                &format!("packing skipped/failed: {}", e));
            None
        }
    };

    // 3) Autogit push (auch wenn Pack scheitert — Build war OK)
    let mut autogit_ok = false;
    let msg = if let Some(z) = &zip_path {
        format!("chore: branch build + pack ({})", z.file_name().and_then(OsStr::to_str).unwrap_or("zip"))
    } else {
        "chore: branch build".to_string()
    };
    if commands::autogit::run(root, Some(msg), false, "origin", Some(&br), true).unwrap_or(1) == 0 {
        autogit_ok = true;
    }

    // 4) Abschluss-Summary
    let end_ms = epoch_ms();
    let artifact = find_artifact(root);
    let mut notes = Vec::new();
    if let Some(z) = &zip_path {
        notes.push(format!("sources_zip={}", z.display()));
    }

    summary::print_end_summary(summary::EndSummary {
        success: true,
        exit_code: 0,
        started_ms: start_ms,
        elapsed_ms: end_ms.saturating_sub(start_ms),
        artifact_path: artifact.map(|p| p.to_string_lossy().to_string()),
        commit_short: crate::vcs::git_short_hash(root),
        commit_branch: Some(format!("origin/{}", br)),
        autogit_pushed: autogit_ok,
        notes,
    });

    0
}
