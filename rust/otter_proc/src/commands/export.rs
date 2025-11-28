///// Otter: Export – erzeugt immer ein Quellen-ZIP unter out/exports/ (Root-Scan; out/, build/, vcpkg/ etc. werden exkludiert)
///// Schneefuchs: Dateiname enthält OP und STATUS; Self-exclude (robust via canonicalize); ASCII-Logs; E0716-Fix.
///// Maus: Pruning pro Sorte (OP+STATUS), max_keep je Sorte; kompatibel zu anyhow::Result.
///// Datei: rust/otter_proc/src/commands/export.rs

use std::{
    fs,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use walkdir::WalkDir;
use zip::{write::FileOptions, CompressionMethod, ZipWriter};

fn epoch_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

// -------------------- Filter-Logik ------------------------------------------------

fn is_archive_name(name_lower: &str) -> bool {
    // Harte Exklusion gängiger Container inkl. Mehrfach-Suffixe.
    const SUFFIXES: &[&str] = &[
        ".zip", ".7z", ".rar",
        ".tar", ".tar.gz", ".tgz",
        ".tar.bz2", ".tbz2",
        ".tar.xz", ".txz",
        ".gz", ".bz2", ".xz",
    ];
    SUFFIXES.iter().any(|s| name_lower.ends_with(s))
}

fn is_binary_ext(name_lower: &str) -> bool {
    // Binaries/Objekte strikt raus
    const BIN: &[&str] = &[
        ".exe", ".dll", ".pdb", ".lib", ".obj", ".o", ".ilk", ".dmp",
        ".so", ".dylib", ".a", ".lo", ".class",
        ".ico", ".png", ".jpg", ".jpeg", ".gif", ".ttf", ".otf", ".dat", ".bin",
    ];
    BIN.iter().any(|s| name_lower.ends_with(s))
}

fn is_allowed_source_file(base_lower: &str, name_lower: &str) -> bool {
    // Whitelist für "relevante Analyse-Quellen"
    if base_lower == "cmakelists.txt"
        || base_lower == "cmakepresets.json"
        || base_lower == "cmakeuserpresets.json"
    {
        return true;
    }
    if base_lower == ".gitignore"
        || base_lower == ".gitattributes"
        || base_lower == ".editorconfig"
        || base_lower == ".clang-format"
        || base_lower == ".clang-tidy"
    {
        return true;
    }

    const EXT_OK: &[&str] = &[
        // C/C++/CUDA/GLSL
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh",
        ".glsl", ".vert", ".frag", ".comp",
        // Rust
        ".rs", ".toml", ".lock",
        // Scripts / Build
        ".cmake", ".ps1", ".psm1", ".bat", ".cmd", ".sh",
        // Config/Daten
        ".json", ".yml", ".yaml", ".txt",
        // Doku
        ".md",
    ];
    EXT_OK.iter().any(|s| name_lower.ends_with(s))
}

fn is_excluded_dir(rel_lower: &str) -> bool {
    // Große/irrelevante Bäume raus – wir wollen NUR Quellcode/Build-Konfigs.
    // Achtung: rel_lower ist immer ein Pfad ohne abschließenden Slash, z.B. "out", "out/exports", "src/main.cpp".
    // Wir wollen ganze Verzeichnisse inkl. Unterbäumen exkludieren.
    const DIRS_SEGMENT: &[&str] = &[
        "vcpkg",
        "vcpkg_installed",
        "vcpkg_downloads",
        "vcpkg_buildtrees",
        "vcpkg_packages",
        "vcpkg_cache",
        "build",
        "out/exports",
        "out",
        "target",
        "rust/otter_proc/target",
        ".git",
        ".vs",
        ".idea",
    ];
    const DIRS_PREFIX: &[&str] = &[
        // z. B. build-debug, build-rel, ...
        "build-",
    ];

    // 1) Präfix-Regeln (z.B. build-* Verzeichnisse)
    if DIRS_PREFIX.iter().any(|p| rel_lower.starts_with(p)) {
        return true;
    }

    // 2) Segment-Regeln: exakt der Name oder "<name>/..."
    DIRS_SEGMENT.iter().any(|p| {
        if rel_lower == *p {
            return true;
        }
        if rel_lower.len() > p.len() && rel_lower.starts_with(p) {
            // Nächstes Zeichen nach dem Segment muss ein '/' sein, damit wir "out" matchen,
            // aber z.B. "outtakes" nicht.
            return rel_lower.as_bytes()[p.len()] == b'/';
        }
        false
    })
}

fn vscode_whitelist(name_lower: &str) -> bool {
    // Aus .vscode nur die beiden "nützlichen" Dateien
    name_lower == ".vscode/c_cpp_properties.json" || name_lower == ".vscode/settings.json"
}

// -------------------- ZIP-Bau -----------------------------------------------------

fn zip_from_root_sources(root: &Path, zip_path: &Path) -> Result<()> {
    // Kanonisch, damit starts_with/== stabil funktionieren
    let root = root.canonicalize()?;
    let zip_file = File::create(zip_path)?;
    let mut zip = ZipWriter::new(zip_file);
    let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

    // Self-Exclude robust (wenn Ziel im Baum liegt)
    let zip_cmp = zip_path
        .canonicalize()
        .unwrap_or_else(|_| zip_path.to_path_buf());

    for entry in WalkDir::new(&root)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // ZIP selbst ausschließen
        if path == zip_cmp {
            continue;
        }

        let rel = match path.strip_prefix(&root) {
            Ok(r) => r,
            Err(_) => continue,
        };

        // portable Relativnamen
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.is_empty() {
            continue;
        }
        let rel_lower = rel_str.to_ascii_lowercase();

        // Verzeichnis-Guards
        if entry.file_type().is_dir() {
            if is_excluded_dir(&rel_lower) {
                // keine Verzeichniseinträge für exkludierte Bäume (kein leeres out/, build/, vcpkg/ im ZIP)
                continue;
            }
            // Verzeichnis anlegen (nur wenn nicht exkludiert)
            zip.add_directory(rel_str, options)?;
            continue;
        }

        // Datei-Guards
        if rel_lower.starts_with(".vscode/") && !vscode_whitelist(&rel_lower) {
            continue;
        }
        if is_excluded_dir(&rel_lower) {
            // Datei unter einem ausgeschlossenen Baum – ignorieren
            continue;
        }
        if is_archive_name(&rel_lower) {
            continue;
        }
        if is_binary_ext(&rel_lower) {
            continue;
        }

        // Nur erlaubte Source-/Build-/Doc-Dateien aufnehmen
        let base_lower = rel
            .file_name()
            .map(|s| s.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if !is_allowed_source_file(&base_lower, &rel_lower) {
            continue;
        }

        // Datei schreiben
        let mut f = File::open(path)?;
        zip.start_file(rel_str, options)?;
        let mut buf = Vec::with_capacity(64 * 1024);
        f.read_to_end(&mut buf)?;
        zip.write_all(&buf)?;
    }

    zip.finish()?;
    Ok(())
}

fn prune_old_zips(out_dir: &Path, op: &str, status: &str, keep: usize) -> Result<()> {
    // Sorte = OP + STATUS  ->  z.B.  branch_ok_*.zip, branch_fail_*.zip
    let prefix = format!("{}_{}_", op, status);
    let mut zips: Vec<(PathBuf, std::time::SystemTime)> = fs::read_dir(out_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            // E0716-Fix: OsString binden, dann lossy-String ableiten
            let name_os = e.file_name();
            let n = name_os.to_string_lossy();
            e.path().is_file() && n.starts_with(&prefix) && n.ends_with(".zip")
        })
        .filter_map(|e| {
            let mtime = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((e.path(), mtime))
        })
        .collect();

    zips.sort_by_key(|(_, t)| std::cmp::Reverse(*t));
    if zips.len() > keep {
        for (p, _) in zips.into_iter().skip(keep) {
            let _ = fs::remove_file(p);
        }
    }
    Ok(())
}

/// Erzeugt ein ZIP **unter `out/exports/`**, das den **Quellcode aus dem Projekt-Root** packt
/// (nur relevante Source-/Build-/Config-/Doc-Dateien; keine Binaries/Archive; keine `out/`, keine `target/`, kein `.git/`).
/// Rückgabe: Pfad zur erzeugten ZIP-Datei.
pub fn run(root: &Path, out_dir: Option<&Path>, max_keep: usize, dry_run: bool) -> Result<PathBuf> {
    // Zielverzeichnis (Default: out/exports)
    let exports_dir = out_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| root.join("out").join("exports"));
    fs::create_dir_all(&exports_dir)?;

    let op = std::env::var("OTTER_OP").unwrap_or_else(|_| "export".to_string());
    let status = std::env::var("OTTER_STATUS").unwrap_or_else(|_| "any".to_string());
    let ts = epoch_ms();

    // Namensschema pro Sorte: <op>_<status>_<epoch>.zip
    let zip_name = format!("{}_{}_{}.zip", op, status, ts);
    let zip_path = exports_dir.join(&zip_name);

    if dry_run {
        println!("[Otter/export] DRY-RUN would write: {}", zip_path.display());
        return Ok(zip_path);
    }

    // Packe **Projekt-Root** (nur Quellen, siehe Filter)
    zip_from_root_sources(root, &zip_path)?;
    println!("[Otter/export] wrote {}", zip_path.display());

    if max_keep > 0 {
        // Pruning pro Sorte im exports-dir
        let _ = prune_old_zips(&exports_dir, &op, &status, max_keep);
    }

    Ok(zip_path)
}
