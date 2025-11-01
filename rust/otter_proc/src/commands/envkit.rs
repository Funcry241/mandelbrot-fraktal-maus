///// Otter: Env-kit helpers (VS/SDK probing) – parked for future use.
///// Schneefuchs: Entire module allows dead_code to silence warnings until wired.
///// Maus: Pure helpers; zero side-effects; ASCII-only.
///// Datei: rust/otter_proc/src/commands/envkit.rs
#![allow(dead_code)]

use std::io;
use std::path::{Path, PathBuf};

// Lightweight, single-purpose helpers kept here for future wiring.
// They are intentionally unused right now; warnings are suppressed at module level.

fn parent(p: &Path, n: usize) -> Option<PathBuf> {
    let mut cur = p.to_path_buf();
    for _ in 0..n {
        cur = cur.parent()?.to_path_buf();
    }
    Some(cur)
}

fn join<P: AsRef<Path>>(base: P, more: &[&str]) -> PathBuf {
    more.iter().fold(base.as_ref().to_path_buf(), |acc, s| acc.join(s))
}

/// Potential future: build a deterministic env map for process launches.
pub fn build_env() -> io::Result<Vec<(String, String)>> {
    // Placeholder: return empty env add-ons; caller inherits current process env.
    Ok(Vec::new())
}
