///// Otter: Packt Quellen in ZIP (Rust/CUDA/C/C++/CMake/TOML/JSON/Shader/Skripte) – Excludes wie vcpkg/.vscode-Whitelist aktiv.
///// Schneefuchs: Deterministische Sortierung; Windows-freundliche Pfade → / im ZIP; kein PowerShell.
///// Maus: ASCII-Logs via runner_term; Default-Ziel out/exports/OtterSources_yyyyMMdd_HHmm.zip; Rückgabe des ZIP-Pfads.
///// Datei: rust/otter_proc/src/commands/pack.rs

use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};

use walkdir::WalkDir;
use zip::write::FileOptions;
use zip::CompressionMethod;

use chrono::Local;

use crate::runner::runner_term::out_info;

// ---------- verbosity ---------------------------------------------------------

fn is_verbose() -> bool {
    // OTTER_PACK_VERBOSE=1|true|yes|verbose  OR  OTTER_PACK_LOG=verbose
    // Alles andere → compact one-line summary.
    use std::env;
    if let Ok(v) = env::var("OTTER_PACK_VERBOSE") {
        let s = v.to_ascii_lowercase();
        return s == "1" || s == "true" || s == "yes" || s == "verbose";
    }
    if let Ok(v) = env::var("OTTER_PACK_LOG") {
        let s = v.to_ascii_lowercase();
        return s == "verbose";
    }
    false
}

fn rel_path(root: &Path, p: &Path) -> Option<String> {
    let rp = p.strip_prefix(root).ok()?;
    let s = rp.to_string_lossy().replace('\\', "/");
    if s.is_empty() { None } else { Some(s) }
}

fn ensure_parent_dirs(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn default_zip_path(root: &Path) -> PathBuf {
    let ts = Local::now().format("%Y%m%d_%H%M").to_string();
    root.join("out").join("exports").join(format!("OtterSources_{}.zip", ts))
}

fn is_excluded_dir(rel: &str, include_dist: bool) -> bool {
    let r = rel.to_ascii_lowercase();

    // dist/ optional
    if !include_dist && r.starts_with("dist/") { return true; }

    // Wuchtige/irrelevante Bäume strikt ausschließen
    let dir_prefixes = [
        "vcpkg/",                         // kompletter vendor-ordner raus
        "vcpkg_installed/", "vcpkg_downloads/", "vcpkg_buildtrees/", "vcpkg_packages/", "vcpkg_cache/",
        "build/", "build-",
        "out/",
        "target/", "rust/otter_proc/target/",
        ".git/", ".vs/", ".idea/",
        // KEIN .vscode/ hier — Whitelist greift im File-Filter
    ];
    for d in &dir_prefixes {
        if r.starts_with(d) { return true; }
    }
    false
}

fn is_excluded_file(rel: &str) -> bool {
    let r = rel.to_ascii_lowercase();

    // .vscode-Whitelist: nur diese beiden Dateien mitnehmen
    if r.starts_with(".vscode/") {
        if r == ".vscode/c_cpp_properties.json" || r == ".vscode/settings.json" {
            // whitelisted
        } else {
            return true;
        }
    }

    let ext_exe = [".obj",".o",".lib",".dll",".exe",".pdb",".dmp",".ilk"];
    if ext_exe.iter().any(|e| r.ends_with(e)) { return true; }

    let ext_misc = [".log",".tmp",".bak",".zip",".7z"];
    if ext_misc.iter().any(|e| r.ends_with(e)) { return true; }

    let junk = ["thumbs.db","desktop.ini",".ds_store"];
    if junk.iter().any(|j| r.ends_with(j)) { return true; }

    false
}

#[derive(Default)]
struct Counts {
    rust_: u64, cuda: u64, c: u64, cxx: u64, hdr: u64,
    cmake: u64, toml: u64, json: u64, glsl: u64, scripts: u64, docs: u64, other: u64,
}

fn bump_counts(c: &mut Counts, rel: &str) {
    let r = rel.to_ascii_lowercase();
    if r.ends_with(".rs") { c.rust_ += 1; return; }
    if r.ends_with(".cu") || r.ends_with(".cuh") { c.cuda += 1; return; }
    if r.ends_with(".cpp") || r.ends_with(".cc") || r.ends_with(".cxx") { c.cxx += 1; return; }
    if r.ends_with(".c")   { c.c += 1; return; }
    if r.ends_with(".h") || r.ends_with(".hpp") || r.ends_with(".hh") || r.ends_with(".hxx") { c.hdr += 1; return; }
    if r.ends_with("cmakelists.txt") || r.ends_with(".cmake") || r.ends_with("cmakepresets.json") || r.ends_with("cmakeuserpresets.json") { c.cmake += 1; return; }
    if r.ends_with(".toml") { c.toml += 1; return; }
    if r.ends_with(".json") { c.json += 1; return; }
    if r.ends_with(".glsl") || r.ends_with(".vert") || r.ends_with(".frag") || r.ends_with(".comp") { c.glsl += 1; return; }
    if r.ends_with(".ps1") || r.ends_with(".bat") || r.ends_with(".cmd") || r.ends_with(".sh") { c.scripts += 1; return; }
    if r.ends_with(".md") || r.ends_with(".txt") || r.ends_with(".rst") { c.docs += 1; return; }
    c.other += 1;
}

/// Entry point: pack all relevant sources into a ZIP.
/// Returns: Ok(PathBuf to ZIP) on success.
pub fn run(root: &Path, out: Option<&Path>, include_dist: bool) -> io::Result<PathBuf> {
    let verbose = is_verbose();

    let out_path = out.map(|p| p.to_path_buf()).unwrap_or_else(|| default_zip_path(root));

    if verbose {
        out_info("ZIP", &format!("root={}", root.display()));
        out_info("ZIP", &format!("zip={}", out_path.display()));
        if !include_dist { out_info("ZIP", "include-dist=no"); } else { out_info("ZIP", "include-dist=YES"); }
    }

    ensure_parent_dirs(&out_path)?;

    // Collect all candidate files (deterministic order)
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() { continue; }
        let abs = entry.path().to_path_buf();
        let Some(rel) = rel_path(root, &abs) else { continue; };
        if is_excluded_dir(&rel, include_dist) { continue; }
        if is_excluded_file(&rel) { continue; }
        files.push((abs, rel));
    }
    files.sort_by(|a, b| a.1.cmp(&b.1));

    // Create ZIP
    let file = File::create(&out_path)?;
    let buf = BufWriter::new(file);
    let mut zip = zip::ZipWriter::new(buf);
    let opts = FileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .unix_permissions(0o644);

    let mut counts = Counts::default();
    let mut added: u64 = 0;

    for (abs, rel) in &files {
        zip.start_file(rel, opts)?;
        let mut f = File::open(abs)?;
        std::io::copy(&mut f, &mut zip)?;
        added += 1;
        bump_counts(&mut counts, rel);
    }
    // flush + close
    zip.finish()?.into_inner()?; 

    // Output
    if verbose {
        out_info("ZIP", &format!("files={}", added));
        out_info("ZIP", &format!("Rust   = {}", counts.rust_));
        out_info("ZIP", &format!("CUDA   = {}", counts.cuda));
        out_info("ZIP", &format!("C/C++  = C={} CXX={} Headers={}", counts.c, counts.cxx, counts.hdr));
        out_info("ZIP", &format!("CMake  = {}", counts.cmake));
        out_info("ZIP", &format!("TOML   = {}", counts.toml));
        out_info("ZIP", &format!("JSON   = {}", counts.json));
        out_info("ZIP", &format!("GLSL   = {}", counts.glsl));
        out_info("ZIP", &format!("Scripts= {}", counts.scripts));
        out_info("ZIP", &format!("Docs   = {}", counts.docs));
        out_info("ZIP", &format!("Other  = {}", counts.other));
        out_info("ZIP", "done.");
    } else {
        // Eine kompakte, einzelne Zusammenfassungszeile
        out_info(
            "PACK",
            &format!(
                "zip={} files={} Rust={} CUDA={} C={} CXX={} Headers={} CMake={} TOML={} JSON={} GLSL={} Scripts={} Docs={} Other={} include-dist={}",
                out_path.display(),
                added,
                counts.rust_, counts.cuda, counts.c, counts.cxx, counts.hdr,
                counts.cmake, counts.toml, counts.json, counts.glsl,
                counts.scripts, counts.docs, counts.other,
                if include_dist { "YES" } else { "no" }
            ),
        );
    }

    Ok(out_path)
}
