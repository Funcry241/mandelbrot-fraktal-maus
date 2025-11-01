///// Otter: CMake cache guard helpers – parked (not wired yet).
///// Schneefuchs: Dead-code allowed for now; functions remain tested later.
///// Maus: Pure helpers, no side-effects; ASCII-only.
///// Datei: rust/otter_proc/src/commands/cacheguard.rs
#![allow(dead_code)]

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub fn binary_dir_for_configure(root: &Path, configure: &str) -> PathBuf {
    // Mirrors common layout: <root>/build
    let _ = configure; // reserved for future variants
    root.join("build")
}

pub fn ensure_cache_matches_source(root: &Path, configure: &str) -> io::Result<()> {
    let build = binary_dir_for_configure(root, configure);
    let cache = build.join("CMakeCache.txt");
    if cache.exists() {
        // Lightweight sanity: if cache exists but points to other source dir, remove build dir.
        // Full parsing delayed; removal handled by caller in actual wiring.
        // Keeping placeholder to avoid unused warnings elsewhere.
        let _ = fs::metadata(&cache)?;
    }
    Ok(())
}

fn canonical_lower(p: &Path) -> io::Result<String> {
    let abs = fs::canonicalize(p)?;
    Ok(abs.to_string_lossy().to_string().to_lowercase())
}
