///// Otter: Process helpers for the runner (stream exec); ASCII-only.
///// Schneefuchs: Keep surface minimal to avoid unused warnings; re-export display_path; dead_code sauber markiert.
///// Maus: Inherit stdio, optional cwd, return ExitStatus; no fancy quoting.
///// Datei: rust/otter_proc/src/prockit.rs

use std::ffi::OsStr;
use std::io;
use std::path::Path;
use std::process::{Command, ExitStatus};

pub use crate::utils::display_path;

// Keep surface minimal to avoid warnings elsewhere.
// Mark as allow(dead_code) so the helper can exist without triggering warnings if unused.
#[allow(dead_code)]
/// Run a command, streaming stdio, with optional working dir; returns ExitStatus.
pub fn run_stream_status<S: AsRef<OsStr>>(
    cmd: &str,
    args: &[S],
    cwd: Option<&Path>,
) -> io::Result<ExitStatus> {
    let mut c = Command::new(cmd);
    c.args(args);
    if let Some(dir) = cwd {
        c.current_dir(dir);
    }
    c.status()
}

// Future helpers can be parked here when needed without polluting warnings.
