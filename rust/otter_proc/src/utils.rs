///// Otter: Utility helpers kept minimal; epoch_ms + path display; ASCII-only.
///// Schneefuchs: Dead-code suppressed at module level to keep future helpers parked.
///// Maus: No doc-comments to avoid inner-attr clash; strictly non-panicking I/O helpers.
///// Datei: rust/otter_proc/src/utils.rs
#![allow(dead_code)]

use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Milliseconds since Unix epoch as u128 (monotonic-ish wall time).
pub fn epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Human-friendly path printing (lossy OK, ASCII-only upstream logs).
pub fn display_path(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

// ------------------------- parked utils (not currently used) -------------------------

pub fn canonicalize(p: &Path) -> io::Result<PathBuf> {
    std::fs::canonicalize(p)
}

pub fn make_relative(child: &Path, root: &Path) -> Option<PathBuf> {
    child.strip_prefix(root).ok().map(|s| s.to_path_buf())
}

pub fn normalize_unique(list: &mut Vec<PathBuf>) {
    // Stable in-place unique by lowercase string form.
    list.sort_by(|a, b| norm(a).cmp(&norm(b)));
    list.dedup_by(|a, b| norm(a) == norm(b));
}

pub fn same_path(a: &Path, b: &Path) -> bool {
    norm(a) == norm(b)
}

fn norm(p: &Path) -> String {
    // Lowercase, forward slashes; best-effort string form.
    let s = p.to_string_lossy().into_owned();
    s.replace('\\', "/").to_lowercase()
}
