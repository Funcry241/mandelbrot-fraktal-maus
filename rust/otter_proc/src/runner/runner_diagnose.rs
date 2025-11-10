///// Otter: Fehler-Diagnose – schreibt bei Build-Fails einen kompakten Report (Markdown).
///// Schneefuchs: heuristisch, keine Extra-Deps; ASCII-only; dedupliziert Top-Fehler.
///// Maus: Pfad out/logs/otter_fail_report.md; speichert Log-Tail (≤120 Zeilen).
///// Datei: rust/otter_proc/src/runner/runner_diagnose.rs

use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn try_write_on_failure(
    root: &Path,
    phase: &str,
    sig: &str,
    exit_code: i32,
    snapshot: Option<&[String]>,
) -> Option<PathBuf> {
    let lines: Vec<String> = snapshot
        .map(|s| s.iter().cloned().collect())
        .unwrap_or_default();

    write_fail_report(root, phase, sig, exit_code, &lines).ok()
}

pub fn write_fail_report_min(
    root: &Path,
    phase: &str,
    sig: &str,
    exit_code: i32,
) -> std::io::Result<PathBuf> {
    write_fail_report(root, phase, sig, exit_code, &[])
}

pub fn write_fail_report(
    root: &Path,
    phase: &str,
    sig: &str,
    exit_code: i32,
    lines: &[String],
) -> std::io::Result<PathBuf> {
    let logs_dir = root.join("out").join("logs");
    create_dir_all(&logs_dir)?;
    let report_path = logs_dir.join("otter_fail_report.md");

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut last_lines: Vec<String> = Vec::new();
    let keep_tail: usize = 120;

    for raw in lines {
        let l = raw.trim();
        if l.is_empty() { continue; }

        last_lines.push(truncate_ascii(l, 400));
        if last_lines.len() > keep_tail {
            let _ = last_lines.remove(0);
        }

        if looks_like_error(l) {
            let key = normalize_for_counting(l);
            *counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));

    let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let mut f = File::create(&report_path)?;

    writeln!(f, "# OtterFail Report")?;
    writeln!(f)?;
    writeln!(f, "- timestamp: {}", ts)?;
    writeln!(f, "- phase: `{}`", phase)?;
    writeln!(f, "- sig: `{}`", sig)?;
    writeln!(f, "- exit_code: {}", exit_code)?;
    writeln!(f)?;

    writeln!(f, "## Most frequent error lines")?;
    if ranked.is_empty() {
        writeln!(f, "_No classified error lines found. The build failed without obvious error patterns._")?;
    } else {
        for (i, (line, cnt)) in ranked.iter().take(10).enumerate() {
            writeln!(f, "{}. (x{}) `{}`", i + 1, cnt, line)?;
        }
    }
    writeln!(f)?;

    writeln!(f, "## Quick hints")?;
    writeln!(f, "- **MSVC C/C++ error** (`error Cxxxx`): erster Fund ist der relevante. Namespace/Include prüfen.")?;
    writeln!(f, "- **CMake Error**: lokal mit `-Wdev`/Toolchain prüfen (CUDA, OpenGL, GLEW, GLFW).")?;
    writeln!(f, "- **Ninja/FAILED**: zur *ersten* Fehlerstelle scrollen; spätere Zeilen sind oft Rauschen.")?;
    writeln!(f, "- **Linker (LNKxxxx)**: fehlende Lib oder CRT-Mix; `LINK_LIBRARIES` & /MD prüfen.")?;
    writeln!(f, "- **CUDA**: Toolkit ≥13.0 und `CMAKE_CUDA_ARCHITECTURES=80;86;89;90`.")?;
    writeln!(f)?;

    writeln!(f, "## Last {} console lines", last_lines.len())?;
    writeln!(f, "```text")?;
    for l in &last_lines {
        writeln!(f, "{}", l)?;
    }
    writeln!(f, "```")?;

    Ok(report_path)
}

fn looks_like_error(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.contains(" cmake error")
        || l.contains("error c")              // MSVC Cxxxx
        || l.contains(" fatal error")
        || l.contains(" error:")              // GCC/Clang
        || l.contains("ninja: build stopped")
        || l.contains(" failed with exit code")
        || l.starts_with("failed:")
        || l.starts_with("linker error")
        || l.contains(" unresolved external symbol")
        || l.contains("undefined reference to")
        || l.contains(" ist kein member von ") // DE-Ausgabe MSVC
        || l.contains(" is not a member of ")
}

fn normalize_for_counting(s: &str) -> String {
    let mut out = s.trim().to_string();
    if let Some(pos) = out.rfind('\\') { out = out[pos+1..].to_string(); }
    if let Some(pos) = out.rfind('/')  { out = out[pos+1..].to_string(); }
    truncate_ascii(&out, 180)
}

fn truncate_ascii(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    let mut t = s.chars().take(max.saturating_sub(3)).collect::<String>();
    t.push_str("...");
    t
}
