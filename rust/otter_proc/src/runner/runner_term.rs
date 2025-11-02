///// Otter: Terminal-Helfer – ANSI-Farben & formatierte Tags für Logs (robust, VT-aware).
///// Schneefuchs: Aktiviert VT auf stdout/stderr; Heuristiken (WT_SESSION/ANSICON/ConEmuANSI); PS 5.1-tauglich.
///// Maus: Fällt sauber auf Plain-ASCII zurück; ein globaler Schalter, kein doppeltes FFI.
///// Datei: rust/otter_proc/src/runner/runner_term.rs

use std::env;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

// -----------------------------------------------------------------------------
// Globales VT/ANSI-Flag – wird in enable_ansi() gesetzt, color_enabled() liest es
// -----------------------------------------------------------------------------
static COLOR_ACTIVE: AtomicBool = AtomicBool::new(false);

#[inline]
fn env_supports_vt() -> bool {
    // Häufige Terminals/Layer unter Windows, die ANSI können
    if env::var_os("WT_SESSION").is_some() { return true; }           // Windows Terminal
    if env::var_os("ANSICON").is_some() { return true; }               // ANSICON
    if matches!(env::var("ConEmuANSI"), Ok(v) if v.eq_ignore_ascii_case("on")) { return true; } // ConEmu
    if matches!(env::var("TERM"), Ok(v) if !v.is_empty() && v.to_ascii_lowercase() != "dumb") { return true; }
    false
}

/// Aktiviert ANSI-Sequenzen (Farben/Cursor) – ohne externe Crates.
/// Auf Windows via direktem FFI zu kernel32; auf anderen Plattformen noop.
/// Gibt `true` zurück, wenn Farbe sinnvoll genutzt werden kann.
pub fn enable_ansi() -> bool {
    // Nicht-Windows: i.d.R. immer ok
    #[cfg(not(windows))]
    {
        COLOR_ACTIVE.store(true, Ordering::Relaxed);
        return true;
    }

    // Windows: VT auf stdout/stderr aktivieren; bei Fehler Heuristiken anwenden
    #[cfg(windows)]
    unsafe {
        use std::ffi::c_void;
        type HANDLE = *mut c_void;
        type DWORD = u32;
        type BOOL  = i32;

        const STD_OUTPUT_HANDLE: i32 = -11; // (DWORD)-11
        const STD_ERROR_HANDLE:  i32 = -12; // (DWORD)-12
        const ENABLE_VIRTUAL_TERMINAL_PROCESSING: DWORD = 0x0004;

        #[link(name = "kernel32")]
        extern "system" {
            fn GetStdHandle(nStdHandle: i32) -> HANDLE;
            fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: *mut DWORD) -> BOOL;
            fn SetConsoleMode(hConsoleHandle: HANDLE, dwMode: DWORD) -> BOOL;
        }

        unsafe fn try_enable(handle_id: i32) -> bool {
            let h = GetStdHandle(handle_id);
            if h.is_null() { return false; }
            let mut mode: DWORD = 0;
            if GetConsoleMode(h, &mut mode) == 0 { return false; }
            if (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0 { return true; } // schon aktiv
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
        }

        let ok_out = try_enable(STD_OUTPUT_HANDLE);
        let ok_err = try_enable(STD_ERROR_HANDLE);
        let ok_env = env_supports_vt();

        let active = ok_out || ok_err || ok_env;
        COLOR_ACTIVE.store(active, Ordering::Relaxed);
        active
    }
}

/// Farben global aktiv?
/// - `OTTER_COLOR=0` → aus
/// - sonst: auf Nicht-Windows true; auf Windows true, wenn enable_ansi() Erfolg/Heuristik meldete
pub fn color_enabled() -> bool {
    if matches!(env::var("OTTER_COLOR"), Ok(v) if v.trim() == "0") {
        return false;
    }

    #[cfg(not(windows))]
    { return true; }

    #[cfg(windows)]
    {
        // Falls enable_ansi() noch nicht aufgerufen wurde, heuristisch entscheiden
        if !COLOR_ACTIVE.load(Ordering::Relaxed) {
            if env_supports_vt() {
                COLOR_ACTIVE.store(true, Ordering::Relaxed);
            }
        }
        COLOR_ACTIVE.load(Ordering::Relaxed)
    }
}

// -----------------------------------------------------------------------------
// ANSI Codes (nur verwenden, wenn color_enabled())
// -----------------------------------------------------------------------------
const RESET: &str = "\x1b[0m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const BRIGHT_BLACK: &str = "\x1b[90m";

fn paint(s: &str, code: &str) -> String {
    if color_enabled() { format!("{code}{s}{RESET}") } else { s.to_string() }
}
fn paint_dim(s: &str) -> String { paint(s, BRIGHT_BLACK) }

// -----------------------------------------------------------------------------
// Tag/Output-Utilities
// -----------------------------------------------------------------------------
fn runner_merge_enabled() -> bool {
    match env::var("OTTER_RUNNER_STYLE") {
        Ok(v) if v.trim().eq_ignore_ascii_case("merge") => return true,
        _ => {}
    }
    matches!(env::var("OTTER_RUNNER_MERGE"), Ok(v) if v.trim() == "1" || v.eq_ignore_ascii_case("on"))
}

fn normalize_tag(tag: &str) -> &str {
    if tag == "RUNNER" && runner_merge_enabled() { "RUST" } else { tag }
}

fn tag_colored(src: &str) -> String {
    let s = normalize_tag(src);
    let (txt_owned, col) = match s {
        "PS"     => ("[PS]".to_string(), MAGENTA),
        "RUST"   => ("[RUST]".to_string(), CYAN),
        "PROC"   => ("[PROC]".to_string(), BLUE),
        "RUNNER" => ("[RUNNER]".to_string(), CYAN), // bewusst deutlich
        other    => (format!("[{}]", other), CYAN),
    };
    paint(&txt_owned, col)
}

pub fn out_info(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let t = tag_colored(src);
    let m = msg.trim_end_matches('\n');
    let _ = writeln!(io::stdout(), "{} {}", t, m);
    let _ = io::stdout().flush();
}

pub fn out_warn(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let t = tag_colored(src);
    let m = paint(msg.trim_end_matches('\n'), YELLOW);
    let _ = writeln!(io::stdout(), "{} {}", t, m);
    let _ = io::stdout().flush();
}

pub fn out_err(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let t = tag_colored(src);
    let m = paint(msg.trim_end_matches('\n'), RED);
    let _ = writeln!(io::stdout(), "{} {}", t, m);
    let _ = io::stdout().flush();
}

/// Ephemere Statuszeile zeichnen/aktualisieren (eine Zeile).
pub fn print_ephemeral(s: &str) {
    let _ = write!(io::stdout(), "\r{}\x1b[K", s);
    let _ = io::stdout().flush();
}

/// Ephemere Zeile löschen.
pub fn end_ephemeral() -> io::Result<()> {
    write!(io::stdout(), "\r\x1b[K")?;
    io::stdout().flush()
}

/// Sanfte Bereinigung: CR entfernen, leere Zeilen verwerfen, etwas Rauschen eindampfen.
pub fn sanitize_line(s: &str) -> String {
    let step1 = s.replace('\r', "");
    let step2 = step1.trim_end_matches('\n').trim();
    if step2.is_empty() {
        return String::new();
    }
    if step2.contains("heuristically generated") {
        return String::from("glfw3 provides CMake targets (heuristic).");
    }
    step2.to_string()
}

/// Terminalbreite: $COLUMNS, sonst 120 als praktikabler Default.
pub fn term_cols() -> usize {
    if let Ok(v) = env::var("COLUMNS") {
        if let Ok(n) = v.parse::<usize>() {
            return n.max(40).min(240);
        }
    }
    120
}

/// Minimaler, farbiger Trailer im Stil „Variante A“.
pub fn out_trailer_min(ok: bool, code: i32, secs: f32, extra: Option<&str>) {
    let tag = tag_colored("RUST");
    let status = if ok { paint("OK", GREEN) } else { paint("FAIL", RED) };
    let bullet = " • ";
    let mut line = format!("{tag} DONE{bullet}{status} (code={code}){bullet}{:.1}s", secs);
    if let Some(x) = extra {
        if !x.trim().is_empty() {
            line.push_str(bullet);
            if color_enabled() { line.push_str(&paint_dim(x)); } else { line.push_str(x); }
        }
    }
    let _ = writeln!(io::stdout(), "{line}");
    let _ = io::stdout().flush();
}
