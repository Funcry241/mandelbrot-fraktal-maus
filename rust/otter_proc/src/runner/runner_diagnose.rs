///// Otter: Diagnose – schreibt bei Exit!=0 einen Report in .build_metrics/ (full/min).
///// Schneefuchs: API stabil – tail: Option<&[String]>, last_snippet: Option<&str>, Rückgabe: Option<PathBuf>.
///// Maus: ASCII-only, chrono(clock), leiser Fallback auf .min.log bei Problemen.
///// Datei: rust/otter_proc/src/runner/runner_diagnose.rs

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use chrono::Local;

fn write_fail_report_full(
    root: &Path,
    phase: &str,
    sig: &str,
    exit_code: i32,
    tail: &[String],
    last_snippet: Option<&str>,
) -> std::io::Result<PathBuf> {
    let dir = root.join(".build_metrics");
    fs::create_dir_all(&dir)?;
    let ts = Local::now().format("%Y-%m-%dT%H-%M-%S").to_string();
    let name = format!("fail_{}_{}_{}.log", phase, exit_code, ts);
    let path = dir.join(name);
    let mut f = File::create(&path)?;

    writeln!(f, "OTTER FAIL REPORT")?;
    writeln!(f, "time   : {}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(f, "phase  : {}", phase)?;
    writeln!(f, "sig    : {}", sig)?;
    writeln!(f, "exit   : {}", exit_code)?;
    if let Some(s) = last_snippet {
        if !s.is_empty() { writeln!(f, "snippet: {}", s)?; }
    }
    writeln!(f, "tail   : {} lines", tail.len())?;
    for line in tail {
        writeln!(f, "  > {}", line)?;
    }
    Ok(path)
}

fn write_fail_report_min(
    root: &Path,
    phase: &str,
    sig: &str,
    exit_code: i32,
    last_snippet: Option<&str>,
) -> std::io::Result<PathBuf> {
    let dir = root.join(".build_metrics");
    fs::create_dir_all(&dir)?;
    let ts = Local::now().format("%Y-%m-%dT%H-%M-%S").to_string();
    let name = format!("fail_{}_{}_{}.min.log", phase, exit_code, ts);
    let path = dir.join(name);
    let mut f = File::create(&path)?;

    writeln!(f, "OTTER FAIL (MIN)")?;
    writeln!(f, "time   : {}", Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(f, "phase  : {}", phase)?;
    writeln!(f, "sig    : {}", sig)?;
    writeln!(f, "exit   : {}", exit_code)?;
    if let Some(s) = last_snippet {
        if !s.is_empty() { writeln!(f, "snippet: {}", s)?; }
    }
    Ok(path)
}

/// Schreibe (bei code!=0) einen Diagnose-Report. Full, wenn Tail vorhanden & nicht leer; sonst Min.
/// Rückgabe: Pfad der geschriebenen Datei oder None bei code==0 oder hartem IO-Fehler.
pub fn try_write_on_failure(
    root: &Path,
    phase: &str,
    sig: &str,
    exit_code: i32,
    tail: Option<&[String]>,
    last_snippet: Option<&str>,
) -> Option<PathBuf> {
    if exit_code == 0 { return None; }

    if let Some(t) = tail {
        if !t.is_empty() {
            match write_fail_report_full(root, phase, sig, exit_code, t, last_snippet) {
                Ok(p) => return Some(p),
                Err(_) => { /* fallthrough → min */ }
            }
        }
    }

    write_fail_report_min(root, phase, sig, exit_code, last_snippet).ok()
}
