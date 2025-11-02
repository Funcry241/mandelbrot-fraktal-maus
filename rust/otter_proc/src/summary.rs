///// Otter: ASCII-Endblock – kompakte, deterministische Abschlusszusammenfassung.
///// Schneefuchs: Feste Struktur; Farbausgabe via ANSI-VT (PS 5.1 tauglich); NO_COLOR respektiert.
///// Maus: Minimalistisch, ohne externe Crates; sauberer Fallback auf Plain-ASCII.
///// Datei: rust/otter_proc/src/summary.rs

use std::env;

// --- ANSI/VT Support (ohne externe Crates) ----------------------------------

static mut WANT_COLOR: bool = true;

#[inline]
fn color_enabled() -> bool {
    unsafe { WANT_COLOR }
}

/// Versucht VT/ANSI-Farben in der Windows-Konsole zu aktivieren.
/// Fällt bei Fehlern still auf Plain-ASCII zurück.
/// Respektiert NO_COLOR=1/true.
/// Ruft diese Funktion idealerweise einmal zu Programmstart auf.
pub fn init_colors() {
    // Env-Override
    if env::var_os("NO_COLOR").is_some() {
        unsafe { WANT_COLOR = false; }
        return;
    }

    // Unixoide: meistens ok – wir lassen an und verlassen uns auf Terminal.
    #[cfg(not(windows))]
    {
        unsafe { WANT_COLOR = true; }
        return;
    }

    // Windows: VT Processing aktivieren
    #[cfg(windows)]
    unsafe {
        const STD_OUTPUT_HANDLE: i32 = -11;
        const STD_ERROR_HANDLE: i32 = -12;
        const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

        extern "system" {
            fn GetStdHandle(nStdHandle: i32) -> isize;
            fn GetConsoleMode(hConsoleHandle: isize, lpMode: *mut u32) -> i32;
            fn SetConsoleMode(hConsoleHandle: isize, dwMode: u32) -> i32;
        }

        unsafe fn enable_for(handle_id: i32) -> bool {
            let h = GetStdHandle(handle_id);
            if h == 0 || h == -1 {
                return false;
            }
            let mut mode: u32 = 0;
            if GetConsoleMode(h, &mut mode as *mut u32) == 0 {
                return false;
            }
            if (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0 {
                return true; // bereits aktiv
            }
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
        }

        let ok_out = enable_for(STD_OUTPUT_HANDLE);
        let ok_err = enable_for(STD_ERROR_HANDLE);

        WANT_COLOR = ok_out || ok_err;
    }
}

// --- Styles ------------------------------------------------------------------

const RST: &str = "\x1b[0m";
const B:   &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const FRED: &str = "\x1b[31m";
const FGREEN: &str = "\x1b[32m";
const FYELLOW: &str = "\x1b[33m";
const FBLUE: &str = "\x1b[34m";
const FCYAN: &str = "\x1b[36m";
const FWHITE: &str = "\x1b[37m";

#[inline]
fn paint(s: &str, color: &str) -> String {
    if color_enabled() {
        format!("{color}{s}{RST}")
    } else {
        s.to_string()
    }
}

// --- Public API --------------------------------------------------------------

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
    // Header-Farben
    let tag_end  = paint("[END]", &format!("{B}{FCYAN}"));
    let tag_good = paint("[GOOD]", FGREEN);
    let tag_info = paint("[INFO]", FBLUE);

    let status_txt = if s.success { "SUCCESS" } else { "FAIL" };
    let status_col = if s.success { FGREEN } else { FRED };
    let status = paint(status_txt, &format!("{B}{status_col}"));

    let artifact = s.artifact_path.as_deref().unwrap_or("(not found)");
    let commit   = s.commit_short.as_deref().unwrap_or("n/a");
    let branch   = s.commit_branch.as_deref().unwrap_or("n/a");
    let push     = if s.autogit_pushed { paint("OK", FGREEN) } else { paint("ERROR", FRED) };

    // Kopf/Frame (Linie leicht abgesetzt)
    let line = if color_enabled() {
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

    // Good-/Info-Section
    let artifact_exists = if s.artifact_path.is_some() { "yes" } else { "no" };
    println!("{tag_good} Artifact exists: {}", paint(artifact_exists, if s.artifact_path.is_some(){ FGREEN } else { FYELLOW }));
    println!("{tag_good} Autogit: {}", if s.autogit_pushed { paint("push OK", FGREEN) } else { paint("push skipped/failed", FRED) });

    for note in s.notes {
        println!("{tag_info}  {}", note);
    }
}
