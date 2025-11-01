///// Otter: Voll-Build-Command – Cache-Guard (UNC/Case-Normalisierung) + CMake-Fahrt.
///// Schneefuchs: Robust gegen \\?\-UNC, Slash-Normalisierung, Case-insensitive Vergleich.
///// Maus: Minimal-invasive Änderung, behält Presets bei; optionales --parallel wird durchgereicht.
///// Datei: rust/otter_proc/src/commands/full.rs

use std::fs;
use std::io::{self, Read};
use std::path::Path;

use crate::commands::winenv;

/// Normalisiert Pfade für einen robusten Vergleich:
/// - entfernt führendes \\?\ oder //?/
/// - ersetzt Backslashes durch Slashes
/// - lowercase
/// - entfernt einen evtl. abschließenden Slash
fn norm_for_compare<S: AsRef<str>>(s: S) -> String {
    let mut t = s.as_ref().replace('\\', "/");
    if let Some(rest) = t.strip_prefix("\\\\?\\").or_else(|| t.strip_prefix("//?/")) {
        t = rest.to_string();
    }
    t = t.to_lowercase();
    while t.ends_with('/') {
        t.pop();
    }
    t
}

/// Liest aus CMakeCache.txt die Source-Root (CMAKE_HOME_DIRECTORY),
/// vergleicht gegen den aktuellen Projekt-Root und löscht bei Mismatch das build/-Verzeichnis.
fn ensure_cache_matches_source(project_root: &Path, _configure_preset: &str) -> io::Result<()> {
    let build_dir = project_root.join("build");
    let cache_file = build_dir.join("CMakeCache.txt");
    if !cache_file.exists() {
        return Ok(());
    }

    // Aktuellen Root normalisieren
    let curr = norm_for_compare(project_root.to_string_lossy());

    // CMAKE_HOME_DIRECTORY aus Cache extrahieren
    let mut content = String::new();
    fs::File::open(&cache_file)?.read_to_string(&mut content)?;
    let mut cache_src: Option<String> = None;
    for line in content.lines() {
        // Typische Form: CMAKE_HOME_DIRECTORY:INTERNAL=C:/path/to/source
        if let Some((_, rhs)) = line.split_once("CMAKE_HOME_DIRECTORY") {
            if let Some((_, val)) = rhs.split_once('=') {
                cache_src = Some(norm_for_compare(val));
                break;
            }
        }
    }

    if let Some(cache) = cache_src {
        if cache != curr {
            println!("[RUNNER][WARN] CMake cache source mismatch");
            println!("  cache={}", cache);
            println!("  curr ={}", curr);
            println!("  -> removing {}", build_dir.display());
            let _ = fs::remove_dir_all(&build_dir);
        }
    } else {
        println!("[RUNNER][WARN] CMake cache lacks CMAKE_HOME_DIRECTORY -> removing {}", build_dir.display());
        let _ = fs::remove_dir_all(&build_dir);
    }

    let _ = fs::create_dir_all(build_dir);
    Ok(())
}

/// Hauptlauf: Cache-Guard -> CMake configure -> CMake build (mit optionalem --parallel)
pub fn run(
    root: &Path,
    build_cfg: &str,
    configure_preset: Option<&str>,
    build_preset: Option<&str>,
    parallel: Option<u32>,
) -> io::Result<i32> {
    let cfg_preset = configure_preset.unwrap_or("windows-msvc");
    let bld_preset = build_preset.unwrap_or("windows-build");

    // Vor dem Lauf: Cache gegen Source-Root absichern (mit UNC/Case-Normalisierung)
    ensure_cache_matches_source(root, cfg_preset)?;

    // Danach die Windows-spezifische CMake-Fahrt (inkl. VsDev/vcvars Kette + Fallback)
    winenv::run_cmake_windows(root, cfg_preset, bld_preset, build_cfg, parallel)
}
