///// Otter: Terminal helpers — ANSI enable (Win), colored tags, ephemeral printing.
///// Schneefuchs: No external crates; cross-platform; safe trimming; fixes E0716 by avoiding chained temporaries.
///// Maus: Colors only on tag ([RUST]/[PS]); ASCII-only messages; line sanitizer.
///// Datei: rust/otter_proc/src/runner/runner_term.rs

use std::io::{self, Write};

// ANSI color codes
const RESET: &str = "\x1b[0m";
const CYA:   &str = "\x1b[36m";
const MAG:   &str = "\x1b[35m";
const YEL:   &str = "\x1b[33m";
const RED:   &str = "\x1b[31m";

#[cfg(windows)]
pub fn enable_ansi() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        use std::ffi::c_void;
        type HANDLE = *mut c_void;
        type DWORD = u32;
        const STD_OUTPUT_HANDLE: i32 = -11;
        const STD_ERROR_HANDLE:  i32 = -12;
        const ENABLE_VIRTUAL_TERMINAL_PROCESSING: DWORD = 0x0004;

        #[link(name = "kernel32")]
        extern "system" {
            fn GetStdHandle(nStdHandle: i32) -> HANDLE;
            fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: *mut DWORD) -> i32;
            fn SetConsoleMode(hConsoleHandle: HANDLE, dwMode: DWORD) -> i32;
        }

        let handles = [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE];
        for h in handles {
            let handle = GetStdHandle(h);
            if !handle.is_null() {
                let mut mode: DWORD = 0;
                if GetConsoleMode(handle, &mut mode as *mut DWORD) != 0 {
                    let _ = SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
                }
            }
        }
    });
}

#[cfg(not(windows))]
#[inline]
pub fn enable_ansi() { /* no-op */ }

fn colored_src_tag(source: &str) -> String {
    let (col_l, col_r) = match source {
        "RUST" => (CYA, RESET),
        "PS"   => (MAG, RESET),
        _      => (RESET, RESET),
    };
    format!("{}[{}]{}", col_l, source, col_r)
}

pub fn out_info(source: &str, msg: &str) {
    println!("[INFO]  {} {}", colored_src_tag(source), msg.trim_end());
}
pub fn out_warn(source: &str, msg: &str) {
    println!("{}[WARN]{}  {} {}", YEL, RESET, colored_src_tag(source), msg.trim_end());
}
pub fn out_err(source: &str, msg: &str) {
    eprintln!("{}[ERR]{}   {} {}", RED, RESET, colored_src_tag(source), msg.trim_end());
}

// Ephemeral line helpers (single-line animation)
pub fn print_ephemeral(line: &str, _prev_len: usize) -> usize {
    // Clear current line and print without newline
    print!("\r\x1b[2K{}", line);
    let _ = io::stdout().flush();
    line.len()
}

pub fn end_ephemeral() {
    // Clear line and move to start for durable print that follows
    print!("\r\x1b[2K");
    let _ = io::stdout().flush();
}

// Best-effort terminal width
pub fn term_cols() -> usize {
    if let Ok(v) = std::env::var("OTTER_TERM_COLS").or_else(|_| std::env::var("COLUMNS")) {
        if let Ok(n) = v.parse::<usize>() { return n.max(40).min(240); }
    }
    120
}

// Very light sanitizer: trim, drop empty, collapse long star-only lines
pub fn sanitize_line(s: &str) -> String {
    // Avoid chained temporaries (fixes E0716): bind each step so lifetimes are clear.
    let no_cr = s.replace('\r', "");                 // owned String
    let no_crlf: &str = no_cr.trim_end_matches('\n'); // &str into `no_cr`
    let trimmed: &str = no_crlf.trim();               // &str into `no_cr`

    if trimmed.is_empty() { return String::new(); }
    if trimmed.chars().all(|c| c == '*') { return String::new(); }

    trimmed.to_string()
}
