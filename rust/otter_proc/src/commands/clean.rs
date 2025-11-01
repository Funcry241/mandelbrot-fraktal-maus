///// Otter: Cleans build artifacts; safe by default; returns 0 on success.
/// //// Schneefuchs: ASCII-only logs; guards against paths escaping project root.
/// ///// Maus: Supports --dry-run, --hard, and extra relative paths.
/// ///// Datei: rust/otter_proc/src/commands/clean.rs

use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

fn is_within(root: &Path, target: &Path) -> bool {
    // Best-effort containment check (canonicalize may fail if path does not exist yet).
    if let (Ok(root_c), Ok(tgt_c)) = (root.canonicalize(), target.canonicalize()) {
        tgt_c.starts_with(&root_c)
    } else {
        // If canonicalize fails, only accept non-absolute, non-escaping relative paths.
        !target.is_absolute()
    }
}

fn rm_entry(path: &Path, dry_run: bool) -> io::Result<()> {
    if dry_run {
        println!("[CLEAN][DRY] would remove {}", path.display());
        return Ok(());
    }
    if path.is_dir() {
        println!("[CLEAN] remove_dir_all {}", path.display());
        fs::remove_dir_all(path)?;
    } else if path.is_file() {
        println!("[CLEAN] remove_file {}", path.display());
        fs::remove_file(path)?;
    } else {
        // Not found or special; ignore (but log).
        println!("[CLEAN] skip (missing) {}", path.display());
    }
    Ok(())
}

/// Clean command entry point.
/// - `dry_run`: only print actions
/// - `hard`: remove additional generator leftovers
/// - `extra`: extra relative paths (as PathBuf) joined to `root`
pub fn run(root: &Path, dry_run: bool, hard: bool, extra: &Vec<PathBuf>) -> io::Result<i32> {
    println!(
        "[CLEAN] start root={} dry_run={} hard={}",
        root.display(),
        dry_run,
        hard
    );

    let mut targets: Vec<PathBuf> = Vec::new();

    // Primary CMake/Ninja output directory used by presets.
    targets.push(root.join("build"));

    // Common local output directory (if present).
    targets.push(root.join("out"));

    // In-source CMake traces (if user configured in-source by accident).
    if hard {
        targets.push(root.join("CMakeCache.txt"));
        targets.push(root.join("CMakeFiles"));
        targets.push(root.join("cmake_install.cmake"));
        targets.push(root.join("install_manifest.txt"));
    }

    // Rust target (hard clean can purge the helper's target to avoid stale deps).
    if hard {
        targets.push(root.join("rust").join("otter_proc").join("target"));
    }

    // User-provided extras: only allow relative paths that do not escape root.
    for e in extra {
        if e.is_absolute()
            || e.components().any(|c| matches!(c, Component::ParentDir))
        {
            println!("[CLEAN][WARN] reject unsafe extra path: {}", e.display());
            continue;
        }
        targets.push(root.join(e));
    }

    // De-duplicate while preserving order.
    let mut seen = std::collections::HashSet::<String>::new();
    targets.retain(|p| seen.insert(p.display().to_string()));

    // Execute removals.
    let mut errs = 0usize;
    for t in targets {
        if !is_within(root, &t) {
            println!("[CLEAN][WARN] skip (outside root) {}", t.display());
            continue;
        }
        if let Err(e) = rm_entry(&t, dry_run) {
            println!("[CLEAN][ERR] {} -> {}", t.display(), e);
            errs += 1;
        }
    }

    println!(
        "[CLEAN] done status={}",
        if errs == 0 { "OK" } else { "WITH_ERRORS" }
    );
    Ok(if errs == 0 { 0 } else { 1 })
}
