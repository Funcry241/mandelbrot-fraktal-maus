///// Otter: Terminal-Helfer – ANSI/VT + WinConsole-Fallback (PS 5.1-tauglich), **auto-on**.
///// Schneefuchs: Aktiviert VT auf stdout/stderr; Heuristiken (WT_SESSION/ANSICON/ConEmuANSI); kein doppeltes FFI anderswo.
///// Maus: Plain-ASCII nur mit OTTER_COLOR=0; Fallback färbt Tags **und** Warn/Fehlertext.
///// Datei: rust/otter_proc/src/runner/runner_term.rs

use std::env;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

// -----------------------------------------------------------------------------
// Globales VT/ANSI-Flag – enable_ansi() versucht es (Windows) zu aktivieren.
// COLOR_ACTIVE signalisiert "VT wirklich aktiv" (nicht nur Wunsch).
// -----------------------------------------------------------------------------
static COLOR_ACTIVE: AtomicBool = AtomicBool::new(false);

#[inline]
fn env_supports_vt() -> bool {
    // Häufige Terminals/Layer unter Windows, die ANSI können
    if env::var_os("WT_SESSION").is_some() { return true; }                  // Windows Terminal
    if env::var_os("ANSICON").is_some() { return true; }                     // ANSICON
    if matches!(env::var("ConEmuANSI"), Ok(v) if v.eq_ignore_ascii_case("on")) { return true; } // ConEmu
    if matches!(env::var("TERM"), Ok(v) if !v.is_empty() && v.to_ascii_lowercase() != "dumb") { return true; }
    false
}

/// Aktiviert ANSI/VT auf stdout/stderr (Windows) bzw. no-op (non-Windows).
/// Gibt `true` zurück, wenn VT für diesen Prozess **aktiv** ist.
pub fn enable_ansi() -> bool {
    #[cfg(not(windows))]
    {
        COLOR_ACTIVE.store(true, Ordering::Relaxed);
        return true;
    }

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
        let vt_active = ok_out || ok_err || env_supports_vt();
        COLOR_ACTIVE.store(vt_active, Ordering::Relaxed);
        vt_active
    }
}

/// Farben **automatisch an**:
/// - `OTTER_COLOR=0`  -> global aus (Plain-ASCII)
/// - Sonst: **immer true** (wir emittieren ANSI-Codes); WinConsole-Fallback greift nur,
///          wenn wir ANSI bewusst deaktivieren (OTTER_COLOR=0).
///
/// Hintergrund: In Windows Terminal / ConPTY ignoriert die Konsole `SetConsoleTextAttribute`,
/// daher brauchen wir **ANSI by default**. Auf alten ConHosts sieht man dann ggf. ESC-Sequenzen
/// – was hier akzeptiert ist, solange `OTTER_COLOR=0` existiert.
pub fn color_enabled() -> bool {
    if matches!(env::var("OTTER_COLOR"), Ok(v) if v.trim() == "0") {
        return false; // globaler Kill
    }
    true // auto-on: immer ANSI emittieren
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
// WinConsole-Fallback (nur sinnvoll, wenn OTTER_COLOR=0 -> ANSI aus).
// -----------------------------------------------------------------------------
#[cfg(windows)]
mod wincon {
    use super::*;
    use std::ffi::c_void;

    pub(super) type HANDLE = *mut c_void;
    type WORD = u16;
    type BOOL = i32;

    pub(super) const STD_OUTPUT_HANDLE: i32 = -11;

    // Vordergrundfarben
    const FOREGROUND_BLUE:  WORD = 0x0001;
    const FOREGROUND_GREEN: WORD = 0x0002;
    const FOREGROUND_RED:   WORD = 0x0004;
    const FOREGROUND_INTENSITY: WORD = 0x0008;

    const COL_DEFAULT: WORD = 0x0007; // hellgrau auf schwarz
    const COL_CYAN:    WORD = FOREGROUND_GREEN | FOREGROUND_BLUE  | FOREGROUND_INTENSITY;
    const COL_MAGENTA: WORD = FOREGROUND_RED   | FOREGROUND_BLUE  | FOREGROUND_INTENSITY;
    const COL_BLUE:    WORD = FOREGROUND_BLUE  | FOREGROUND_INTENSITY;
    const COL_YELLOW:  WORD = FOREGROUND_RED   | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
    const COL_RED_BRI: WORD = FOREGROUND_RED   | FOREGROUND_INTENSITY;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetStdHandle(nStdHandle: i32) -> HANDLE;
        fn SetConsoleTextAttribute(hConsoleOutput: HANDLE, wAttributes: WORD) -> BOOL;
    }

    #[inline]
    pub(super) fn can_use() -> bool {
        // Fallback nur, wenn ANSI **deaktiviert** ist.
        if super::color_enabled() { return false; }
        let h = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
        !h.is_null()
    }

    #[inline]
    fn color_for_tag(tag: &str) -> WORD {
        match tag {
            "PS"     => COL_MAGENTA,
            "RUST"   => COL_CYAN,
            "PROC"   => COL_BLUE,
            "RUNNER" => COL_CYAN,
            _        => COL_CYAN,
        }
    }

    pub(super) fn print_tag_only(tag_text: &str, tag: &str) {
        unsafe {
            let h = GetStdHandle(STD_OUTPUT_HANDLE);
            if h.is_null() {
                let _ = write!(io::stdout(), "{tag_text}");
                let _ = io::stdout().flush();
                return;
            }
            let _ = SetConsoleTextAttribute(h, color_for_tag(tag));
            let _ = write!(io::stdout(), "{tag_text}");
            let _ = io::stdout().flush();
            let _ = SetConsoleTextAttribute(h, COL_DEFAULT);
        }
    }

    pub(super) fn print_tag_and_msg(tag_text: &str, tag: &str, msg: &str, msg_color: WORD) {
        unsafe {
            let h = GetStdHandle(STD_OUTPUT_HANDLE);
            if h.is_null() {
                let _ = writeln!(io::stdout(), "{tag_text} {msg}");
                let _ = io::stdout().flush();
                return;
            }
            // Tag
            let _ = SetConsoleTextAttribute(h, color_for_tag(tag));
            let _ = write!(io::stdout(), "{tag_text}");
            // Message
            let _ = SetConsoleTextAttribute(h, msg_color);
            let _ = writeln!(io::stdout(), " {msg}");
            let _ = io::stdout().flush();
            // Reset
            let _ = SetConsoleTextAttribute(h, COL_DEFAULT);
        }
    }

    pub(super) fn col_warn() -> WORD { COL_YELLOW }
    pub(super) fn col_err () -> WORD { COL_RED_BRI }
}

#[cfg(not(windows))]
mod wincon {
    pub(super) fn can_use() -> bool { false }
    pub(super) fn print_tag_only(_tag_text: &str, _tag: &str) { /* no-op */ }
    pub(super) fn print_tag_and_msg(_tag_text: &str, _tag: &str, _msg: &str, _msg_color: u16) { /* no-op */ }
    pub(super) fn col_warn() -> u16 { 0 }
    pub(super) fn col_err () -> u16 { 0 }
}

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

#[inline]
fn tag_text<'a>(src: &'a str) -> (&'a str, String) {
    let s = normalize_tag(src);
    (s, format!("[{}]", s))
}

fn tag_colored_ansi(src: &str) -> String {
    let (name, raw) = tag_text(src);
    let col = match name {
        "PS"     => MAGENTA,
        "RUST"   => CYAN,
        "PROC"   => BLUE,
        "RUNNER" => CYAN,
        _        => CYAN,
    };
    paint(&raw, col)
}

// -----------------------------------------------------------------------------
// Öffentliche Ausgaben
// -----------------------------------------------------------------------------
pub fn out_info(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let m = msg.trim_end_matches('\n');

    if color_enabled() {
        let t = tag_colored_ansi(src);
        let _ = writeln!(io::stdout(), "{} {}", t, m);
        let _ = io::stdout().flush();
        return;
    }
    if wincon::can_use() {
        let (raw_tag, ttxt) = tag_text(src);
        wincon::print_tag_only(&ttxt, raw_tag);
        let _ = writeln!(io::stdout(), " {}", m);
        let _ = io::stdout().flush();
        return;
    }
    let (_raw, ttxt) = tag_text(src);
    let _ = writeln!(io::stdout(), "{} {}", ttxt, m);
    let _ = io::stdout().flush();
}

pub fn out_warn(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let m = msg.trim_end_matches('\n');

    if color_enabled() {
        let t = tag_colored_ansi(src);
        let m = paint(m, YELLOW);
        let _ = writeln!(io::stdout(), "{} {}", t, m);
        let _ = io::stdout().flush();
        return;
    }
    if wincon::can_use() {
        let (raw_tag, ttxt) = tag_text(src);
        wincon::print_tag_and_msg(&ttxt, raw_tag, m, wincon::col_warn());
        return;
    }
    let (_raw, ttxt) = tag_text(src);
    let _ = writeln!(io::stdout(), "{} {}", ttxt, m);
    let _ = io::stdout().flush();
}

pub fn out_err(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let m = msg.trim_end_matches('\n');

    if color_enabled() {
        let t = tag_colored_ansi(src);
        let m = paint(m, RED);
        let _ = writeln!(io::stdout(), "{} {}", t, m);
        let _ = io::stdout().flush();
        return;
    }
    if wincon::can_use() {
        let (raw_tag, ttxt) = tag_text(src);
        wincon::print_tag_and_msg(&ttxt, raw_tag, m, wincon::col_err());
        return;
    }
    let (_raw, ttxt) = tag_text(src);
    let _ = writeln!(io::stdout(), "{} {}", ttxt, m);
    let _ = io::stdout().flush();
}

/// Ephemere Statuszeile zeichnen/aktualisieren (eine Zeile).
pub fn print_ephemeral(s: &str) {
    if color_enabled() {
        let _ = write!(io::stdout(), "\r{}\x1b[K", s);
    } else {
        // Ohne ANSI: kein \x1b[K senden (würde sichtbar), nur CR + Text.
        let _ = write!(io::stdout(), "\r{}", s);
    }
    let _ = io::stdout().flush();
}

/// Ephemere Zeile löschen.
pub fn end_ephemeral() -> io::Result<()> {
    if color_enabled() {
        write!(io::stdout(), "\r\x1b[K")?;
    } else {
        // Ohne ANSI: nur an den Zeilenanfang
        write!(io::stdout(), "\r")?;
    }
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
    if color_enabled() {
        let tag = tag_colored_ansi("RUST");
        let status_colored = if ok { paint("OK", GREEN) } else { paint("FAIL", RED) };
        let status = status_colored; // capture
        let bullet = " • ";
        let mut line = format!("{tag} DONE{bullet}{status} (code={code}){bullet}{secs:.1}s");
        if let Some(x) = extra {
            if !x.trim().is_empty() {
                line.push_str(bullet);
                line.push_str(&paint_dim(x));
            }
        }
        let _ = writeln!(io::stdout(), "{line}");
        let _ = io::stdout().flush();
        return;
    }

    if wincon::can_use() {
        // Tag farbig, Rest plain
        let (raw_tag, ttxt) = tag_text("RUST");
        wincon::print_tag_only(&ttxt, raw_tag);
        let status_plain = if ok { "OK" } else { "FAIL" };
        let bullet = " • ";
        let mut line = format!(" DONE{bullet}{status_plain} (code={code}){bullet}{secs:.1}s");
        if let Some(x) = extra {
            if !x.trim().is_empty() {
                line.push_str(bullet);
                line.push_str(x);
            }
        }
        let _ = writeln!(io::stdout(), "{line}");
        let _ = io::stdout().flush();
        return;
    }

    // Plain
    let tag = "[RUST]";
    let status_plain = if ok { "OK" } else { "FAIL" };
    let bullet = " • ";
    let mut line = format!("{tag} DONE{bullet}{status_plain} (code={code}){bullet}{secs:.1}s");
    if let Some(x) = extra {
        if !x.trim().is_empty() {
            line.push_str(bullet);
            line.push_str(x);
        }
    }
    let _ = writeln!(io::stdout(), "{line}");
    let _ = io::stdout().flush();
}
