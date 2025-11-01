///// Otter: Terminal helpers (ANSI enable, pretty tags, ephemeral line, sanitize).
///// Schneefuchs: No external crates; Windows FFI zu kernel32; trims banner noise.
///// Maus: Termbreite aus $COLUMNS (Fallback 120); ASCII-only; ruhiges Verhalten auf Nicht-TTY.
///// Datei: rust/otter_proc/src/runner_term.rs

use std::env;
use std::io::{self, Write};

/// Aktiviert ANSI-Sequenzen (Farben/Cursor) – ohne externe Crates.
/// Auf Windows via direktem FFI zu kernel32; auf anderen Plattformen no-op.
pub fn enable_ansi() {
    #[cfg(windows)]
    unsafe {
        // Minimaler FFI-Sockel, keine winapi/windows-Crates.
        use std::ffi::c_void;
        type HANDLE = *mut c_void;
        type DWORD = u32;
        type BOOL  = i32;

        const STD_OUTPUT_HANDLE: i32 = -11; // (DWORD)-11
        const ENABLE_VIRTUAL_TERMINAL_PROCESSING: DWORD = 0x0004;

        #[link(name = "kernel32")]
        extern "system" {
            fn GetStdHandle(nStdHandle: i32) -> HANDLE;
            fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: *mut DWORD) -> BOOL;
            fn SetConsoleMode(hConsoleHandle: HANDLE, dwMode: DWORD) -> BOOL;
        }

        let h = GetStdHandle(STD_OUTPUT_HANDLE);
        if !h.is_null() {
            let mut mode: DWORD = 0;
            if GetConsoleMode(h, &mut mode) != 0 {
                let _ = SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }
        }
    }
    #[cfg(not(windows))]
    {
        // nix zu tun
    }
}

/// Farben global aktiv?
/// OTTER_COLOR=0 -> aus; alles andere -> an (Default).
pub fn color_enabled() -> bool {
    !matches!(env::var("OTTER_COLOR"), Ok(v) if v.trim() == "0")
}

// ANSI Codes (nur verwenden, wenn color_enabled()).
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

fn tag_colored(s: &str) -> String {
    // L1: Tag-Farben — eigener String, keine temporären Borrows
    let (txt_owned, col) = match s {
        "PS"   => ("[PS]".to_string(), MAGENTA),
        "RUST" => ("[RUST]".to_string(), CYAN),
        "PROC" => ("[PROC]".to_string(), BLUE),
        other  => (format!("[{}]", other), CYAN),
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
    // zwei kurze, besitzende Schritte vermeiden temporäre Borrow-Lifetime-Probleme
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
/// Beispiel:
/// [RUST] DONE • OK (code=0) • 61.4s
/// Optionales `extra` kann vom Aufrufer genutzt werden (z. B. „artifact=… • git: pushed … ✓“).
pub fn out_trailer_min(ok: bool, code: i32, secs: f32, extra: Option<&str>) {
    let tag = tag_colored("RUST");
    let status = if ok { paint("OK", GREEN) } else { paint("FAIL", RED) };
    let bullet = " • ";
    let mut line = format!("{tag} DONE{bullet}{status} (code={code}){bullet}{:.1}s", secs);
    if let Some(x) = extra {
        if !x.trim().is_empty() {
            line.push_str(bullet);
            // Dimmen erlaubt unaufdringliche Zusatzelemente am Ende
            if color_enabled() {
                line.push_str(&paint_dim(x));
            } else {
                line.push_str(x);
            }
        }
    }
    let _ = writeln!(io::stdout(), "{line}");
    let _ = io::stdout().flush();
}
