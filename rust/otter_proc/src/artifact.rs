///// Otter: Artefakt-Suche – Kandidaten scannen & melden, ein konsistenter Fundort-Log.
///// Schneefuchs: Farbige yes/no bei ANSI; keine I/O-Seiteneffekte außer Logs.
///// Maus: pub(crate); minimal; identische Kandidatenreihenfolge wie zuvor.
///// Datei: rust/otter_proc/src/artifact.rs
#![deny(warnings)]

use std::path::{Path, PathBuf};

/// "yes"/"no" ggf. farbig, abhängig von ANSI-Fähigkeit.
pub(crate) fn fmt_exists(b: bool) -> String {
    if crate::runner::runner_term::color_enabled() {
        if b { "\x1b[32myes\x1b[0m".to_string() } else { "\x1b[33mno\x1b[0m".to_string() }
    } else {
        if b { "yes".to_string() } else { "no".to_string() }
    }
}

/// Kandidat loggen und Existenz prüfen.
pub(crate) fn log_candidate(root: &Path, rel: &str) -> (PathBuf, bool) {
    let p = root.join(rel);
    let exists = p.is_file();
    crate::runner::runner_term::out_info(
        "RUNNER",
        &format!("artifact-candidate: {} exists={}", p.display(), fmt_exists(exists)),
    );
    (p, exists)
}

/// Artefakt anhand üblicher Build-Pfade finden (erste Übereinstimmung gewinnt).
pub(crate) fn find_artifact(root: &Path) -> Option<PathBuf> {
    // Reihenfolge beibehalten, um Log- und Suchverhalten stabil zu halten.
    let candidates = [
        "build/RelWithDebInfo/mandelbrot_otterdream.exe",
        "build/bin/RelWithDebInfo/mandelbrot_otterdream.exe",
        "build/bin/mandelbrot_otterdream.exe",
        "build/mandelbrot_otterdream.exe",
    ];
    for rel in candidates {
        let (p, exists) = log_candidate(root, rel);
        if exists {
            crate::runner::runner_term::out_info("RUNNER", &format!("artifact: {}", p.display()));
            return Some(p);
        }
    }
    None
}
