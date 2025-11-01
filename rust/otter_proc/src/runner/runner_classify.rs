///// Otter: Klassifizierer für Output-Zeilen (Info/Warn/Err) mit einfacher Heuristik.
///// Schneefuchs: ASCII-only; bewusst konservativ, um False-Positives zu vermeiden.
///// Maus: Minimale API: Sev + classify_line(&str) -> Sev.
///// Datei: rust/otter_proc/src/runner/runner_classify.rs

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Sev { Info, Warn, Err }

fn is_error_line(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    l.contains("fatal error") || l.contains("error:")
        || l.contains(" cmake error") || l.starts_with("error")
        || l.contains("] error")
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
