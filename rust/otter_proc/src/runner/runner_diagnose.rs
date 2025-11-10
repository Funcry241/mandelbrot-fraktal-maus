///// Otter: Diagnose – schreibt Fail-Reports (voll + minimal) bei Exit≠0, inkl. Log-Tail.
/// //// Schneefuchs: Keine Dead-Code-Warnungen; robuste Pfade; ASCII-only; Windows/Linux sicher.
/// //// Maus: Speichert unter `.build_metrics/fail/<ts>_<phase>_<code>.txt` + Kurzfassung `out/last_fail.txt`.
///// Datei: rust/otter_proc/src/runner/runner_diagnose.rs

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::utils;

/// Schreibt den **kompletten** Report in eine datierte Datei im Fail-Ordner.
fn write_fail_report_full(
    workdir: &Path,
    phase: &str,
    sig: &str,
    code: i32,
    tail: Option<&[String]>,
) -> io::Result<PathBuf> {
    let ts = utils::epoch_ms();
    let fail_dir = workdir.join(".build_metrics").join("fail");
    fs::create_dir_all(&fail_dir)?;

    // Dateiname: <ts>_<phase>_<code>.txt – nur ASCII, unsichere Zeichen filtern
    fn sanitize_token(s: &str) -> String {
        s.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect()
    }
    let fname = format!(
        "{}_{}_{}.txt",
        ts,
        sanitize_token(phase),
        code
    );
    let fpath = fail_dir.join(fname);
    let mut f = File::create(&fpath)?;

    // Header
    writeln!(f, "FAIL REPORT")?;
    writeln!(f, "ts_ms: {}", ts)?;
    writeln!(f, "phase: {}", phase)?;
    writeln!(f, "sig: {}", sig)?;
    writeln!(f, "exit_code: {}", code)?;
    writeln!(f, "cwd: {}", workdir.display())?;
    writeln!(f, "------------------------------------------------------------")?;

    // Inhalt (Tail)
    if let Some(lines) = tail {
        writeln!(f, "LOG TAIL ({} lines):", lines.len())?;
        for line in lines {
            // ASCII-only: nicht druckbare Zeichen ersetzen
            let cleaned: String = line.chars().map(|c| if c.is_ascii() { c } else { '?' }).collect();
            writeln!(f, "{}", cleaned)?;
        }
    } else {
        writeln!(f, "No log tail available.")?;
    }

    Ok(fpath)
}

/// Schreibt eine **Kurzfassung** (1–2 Zeilen + ein paar letzte Zeilen) für schnelle Sichtung.
/// Achtung: **wird verwendet** (keine dead_code-Warnung).
pub fn write_fail_report_min(
    workdir: &Path,
    phase: &str,
    sig: &str,
    code: i32,
    tail: Option<&[String]>,
) -> io::Result<PathBuf> {
    let out_dir = workdir.join("out");
    fs::create_dir_all(&out_dir)?;
    let fpath = out_dir.join("last_fail.txt");

    let mut f = File::create(&fpath)?;
    let ts = utils::epoch_ms();

    writeln!(f, "FAIL phase={} sig=\"{}\" code={} ts_ms={}", phase, sig, code, ts)?;

    // Eine sehr knappe Tail-Zusammenfassung (max. 20 Zeilen)
    if let Some(lines) = tail {
        writeln!(f, "tail: {} lines (showing last 20)", lines.len())?;
        let n = lines.len();
        let start = if n > 20 { n - 20 } else { 0 };
        for line in &lines[start..] {
            let cleaned: String = line.chars().map(|c| if c.is_ascii() { c } else { '?' }).collect();
            writeln!(f, "{}", cleaned)?;
        }
    } else {
        writeln!(f, "tail: none")?;
    }

    Ok(fpath)
}

/// Öffentliche API: Beim Fehlschlag beide Reports versuchen (voll + minimal).
/// Fehler beim Schreiben werden bewusst **ignoriert** (Runner soll nicht zusätzlich scheitern).
pub fn try_write_on_failure(
    workdir: &Path,
    phase: &str,
    sig: &str,
    code: i32,
    tail: Option<&[String]>,
) -> io::Result<()> {
    let _ = write_fail_report_full(workdir, phase, sig, code, tail);
    let _ = write_fail_report_min(workdir, phase, sig, code, tail);
    Ok(())
}
