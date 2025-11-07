///// Otter: Artifact locator with deterministic search order and clean log lines.
///// Schneefuchs: Makes ExistsLike + fmt_exists pub(crate) to satisfy private-bounds lint.
///// Maus: Cross-platform exe name; emits [RUNNER] artifact-candidate and final [RUNNER] artifact.
///// Datei: rust/otter_proc/src/artifact.rs

use std::path::{Path, PathBuf};

// Route messages through the colored runner terminal helpers
use crate::runner::runner_term;

/// Tiny trait so `fmt_exists` can take either a `bool` **or** a `&Path`.
pub(crate) trait ExistsLike {
    fn exists_bool(self) -> bool;
}
impl ExistsLike for bool {
    #[inline] fn exists_bool(self) -> bool { self }
}
impl<'a> ExistsLike for &'a Path {
    #[inline] fn exists_bool(self) -> bool { self.exists() }
}
impl<'a> ExistsLike for &'a PathBuf {
    #[inline] fn exists_bool(self) -> bool { self.as_path().exists() }
}

/// Returns `"yes"` or `"no"` — matches call sites like:
/// `format!("artifact-candidate: {} exists={}", p.display(), fmt_exists(exists))`
#[inline]
pub(crate) fn fmt_exists<E: ExistsLike>(e: E) -> &'static str {
    if e.exists_bool() { "yes" } else { "no" }
}

#[inline]
fn exe_basename() -> &'static str {
    if cfg!(windows) { "mandelbrot_otterdream.exe" } else { "mandelbrot_otterdream" }
}

/// Candidate list in priority order (app-local first).
pub(crate) fn artifact_candidates(root: &Path) -> Vec<PathBuf> {
    let exe = exe_basename();
    let b = root.join("build");

    let mut v = Vec::with_capacity(8);
    // Highest priority: flat app-local in build/
    v.push(b.join(exe));
    // Next: build/bin/
    v.push(b.join("bin").join(exe));

    // Common multi-config layouts
    for cfg in ["RelWithDebInfo", "Release", "Debug"] {
        v.push(b.join(cfg).join(exe));
        v.push(b.join("bin").join(cfg).join(exe));
    }
    v
}

/// Scans candidates and emits compact log lines consumed by the runner UI.
/// Returns the first existing artifact path.
pub fn find_artifact(root: &Path) -> Option<PathBuf> {
    for cand in artifact_candidates(root) {
        runner_term::out_info(
            "RUNNER",
            &format!("artifact-candidate: {} exists={}", cand.display(), fmt_exists(&cand)),
        );
        if cand.exists() {
            runner_term::out_info("RUNNER", &format!("artifact: {}", cand.display()));
            return Some(cand);
        }
    }
    None
}
