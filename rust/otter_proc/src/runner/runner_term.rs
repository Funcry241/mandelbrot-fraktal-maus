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

fn tag(s: &str) -> String {
    match s {
        "PS"   => "[PS]".to_string(),
        "RUST" => "[RUST]".to_string(),
        "PROC" => "[PROC]".to_string(),
        other  => format!("[{}]", other),
    }
}

pub fn out_info(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let _ = writeln!(io::stdout(), "{} {}", tag(src), msg.trim_end_matches('\n'));
    let _ = io::stdout().flush();
}

pub fn out_warn(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let _ = writeln!(io::stdout(), "{} {}", tag(src), msg.trim_end_matches('\n'));
    let _ = io::stdout().flush();
}

pub fn out_err(src: &str, msg: &str) {
    let _ = end_ephemeral();
    let _ = writeln!(io::stdout(), "{} {}", tag(src), msg.trim_end_matches('\n'));
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
