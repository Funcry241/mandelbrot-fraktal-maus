///// Otter: Classify child lines into Info/Warn/Err for colored durable logs.
///// Schneefuchs: Simple heuristics; case-insensitive; avoids false-positives.
///// Maus: Prefer “error:”/“warning” tokens; keeps ASCII; minimal branching.
///// Datei: rust/otter_proc/src/runner/runner_classify.rs

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Sev { Info, Warn, Err }

fn is_error_line(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    l.contains("fatal error")
        || l.contains(" cmake error")
        || l.contains("error:")
        || l.starts_with("error")
        || l.contains("] error")
        || l.contains("build failed")
        || l.contains("build failed.")
}

fn is_warning_line(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    if is_error_line(&l) { return false; }
    l.contains("warning") || l.contains(" cmake warning") || l.starts_with("warning")
}

pub fn classify_line(line: &str) -> Sev {
    if is_error_line(line) { Sev::Err }
    else if is_warning_line(line) { Sev::Warn }
    else { Sev::Info }
}
