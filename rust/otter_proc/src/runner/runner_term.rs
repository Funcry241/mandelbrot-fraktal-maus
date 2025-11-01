///// Otter: Terminal-Helfer: ANSI aktivieren (Win), farbige Tags & ephemeres Zeilenlöschen.
///// Schneefuchs: ASCII-only, keine externen Crates; robuste no-op auf non-Windows.
///// Maus: Ruhig arbeiten; clear_ephemeral_line() für Progress-Redraw.
///// Datei: rust/otter_proc/src/runner/runner_term.rs

use std::io::{self, Write};

pub const RED:   &str = "\x1b[31m";
pub const YEL:   &str = "\x1b[33m";
pub const CYA:   &str = "\x1b[36m";
pub const RESET: &str = "\x1b[0m";

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

pub fn out_info(source: &str, msg: &str) {
    println!("[INFO]  [{}] {}", source, msg);
}
pub fn out_warn(source: &str, msg: &str) {
    println!("{}[WARN]{}  [{}] {}", YEL, RESET, source, msg);
}
pub fn out_err(source: &str, msg: &str) {
    eprintln!("{}[ERR]{}   [{}] {}", RED, RESET, source, msg);
}

/// Löscht die aktuelle Zeile „ephemer“ (Progresszeile) und setzt den Cursor an den Anfang.
pub fn clear_ephemeral_line() {
    // konservativ breite Zeile; bewusst ohne Terminal-Query
    print!("\r{:<160}\r", "");
    let _ = io::stdout().flush();
}
