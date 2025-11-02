///// Otter: ASCII/ANSI-Endblock – kompakte, deterministische Abschlusszusammenfassung mit Farbhilfe.
///// Schneefuchs: Keine Win32-FFI-Doppler; nutzt runner_term::{enable_ansi,color_enabled}; OTTER_COLOR=0 schaltet Farbe aus.
///** Maus: Minimalistisch; sauberer Fallback auf Plain-ASCII (kein NO_COLOR-Zwang).
///// Datei: rust/otter_proc/src/summary.rs

use std::env;

#[inline]
fn want_color() -> bool {
    // Optionaler Global-Kill-Switch via NO_COLOR (falls du das nutzen willst)
    if env::var_os("NO_COLOR").is_some() {
        return false;
    }
    // Projektweiter Schalter: OTTER_COLOR=0 -> aus (kommt aus runner_term)
    crate::runner::runner_term::color_enabled()
}

// ANSI Styles (nur verwenden, wenn want_color() true liefert)
const RST: &str = "\x1b[0m";
const B:   &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const FRED: &str = "\x1b[31m";
const FGREEN: &str = "\x1b[32m";
const FYELLOW: &str = "\x1b[33m";
const FBLUE: &str = "\x1b[34m";
const FCYAN: &str = "\x1b[36m";

#[inline]
fn paint(s: &str, color: &str) -> String {
    if want_color() {
        format!("{color}{s}{RST}")
    } else {
        s.to_string()
    }
}

pub struct EndSummary {
    pub success: bool,
    pub exit_code: i32,
    pub started_ms: u128,
    pub elapsed_ms: u128,
    pub artifact_path: Option<String>,
    pub commit_short: Option<String>,
    pub commit_branch: Option<String>,
    pub autogit_pushed: bool,
    pub notes: Vec<String>,
}

fn fmt_secs(ms: u128) -> String {
    let secs = ms as f64 / 1000.0;
    format!("{:.1} s", secs)
}

pub fn print_end_summary(s: EndSummary) {
    // Tags
    let tag_end  = paint("[END]", &format!("{B}{FCYAN}"));
    let tag_good = paint("[GOOD]", FGREEN);
    let tag_info = paint("[INFO]", FBLUE);

    // Status
    let status_txt = if s.success { "SUCCESS" } else { "FAIL" };
    let status_col = if s.success { FGREEN } else { FRED };
    let status = paint(status_txt, &format!("{B}{status_col}"));

    // Inhalte
    let artifact = s.artifact_path.as_deref().unwrap_or("(not found)");
    let commit   = s.commit_short.as_deref().unwrap_or("n/a");
    let branch   = s.commit_branch.as_deref().unwrap_or("n/a");
    let push_txt = if s.autogit_pushed { "OK" } else { "ERROR" };
    let push     = paint(push_txt, if s.autogit_pushed { FGREEN } else { FRED });

    // Rahmen
    let line = if want_color() {
        paint("+--------------------------------------------------------------+", DIM)
    } else {
        "+--------------------------------------------------------------+".to_string()
    };

    println!("{tag_end} {line}");
    println!("{tag_end} | BUILD SUMMARY: {status} (code={})                              |", s.exit_code);
    println!("{tag_end} {line}");
    println!("{tag_end} | Artifact : {}", artifact);
    println!("{tag_end} | Started  : ts_ms={}", s.started_ms);
    println!("{tag_end} | Elapsed  : ~{}", fmt_secs(s.elapsed_ms));
    println!("{tag_end} | Commit   : {} -> {} (push={})", commit, branch, push);
    println!("{tag_end} {line}");

    // Gute Nachrichten kompakt
    let artifact_exists = if s.artifact_path.is_some() { "yes" } else { "no" };
    println!("{tag_good} Artifact exists: {}", paint(artifact_exists, if s.artifact_path.is_some(){ FGREEN } else { FYELLOW }));
    println!("{tag_good} Autogit: {}", if s.autogit_pushed { paint("push OK", FGREEN) } else { paint("push skipped/failed", FRED) });

    for note in s.notes {
        println!("{tag_info}  {}", note);
    }
}
