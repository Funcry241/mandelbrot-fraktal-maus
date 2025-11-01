///// Otter: Terminal-Helpers: ANSI aktivieren, farbige Tags, Ephemeral-Line-Steuerung.
/// ///// Schneefuchs: ASCII-only Content; keine externen Crates; Windows-Enable best effort.
/// ///// Maus: color_src("RUST"/"PS"/"CMAKE") + clear/print/end_ephemeral; simple API.
/// ///// Datei: rust/otter_proc/src/runner/runner_term.rs

use std::io::{self, Write};

pub const RED:   &str = "\x1b[31m";
pub const YEL:   &str = "\x1b[33m";
pub const GRN:   &str = "\x1b[32m";
pub const CYA:   &str = "\x1b[36m";
pub const MAG:   &str = "\x1b[35m";
pub const BLU:   &str = "\x1b[34m";
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

pub fn color_src(src: &str) -> String {
    match src {
        "RUST"  => format!("{}RUST{}", MAG, RESET),
        "PS"    => format!("{}PS{}", CYA, RESET),
        "CMAKE" => format!("{}CMAKE{}", BLU, RESET),
        other   => other.to_string(),
    }
}

pub fn out_info(source: &str, msg: &str) {
    println!("[INFO]  [{}] {}", color_src(source), msg);
}
pub fn out_warn(source: &str, msg: &str) {
    println!("{}[WARN]{}  [{}] {}", YEL, RESET, color_src(source), msg);
}
pub fn out_err(source: &str, msg: &str) {
    eprintln!("{}[ERR]{}   [{}] {}",  RED, RESET, color_src(source), msg);
}

// Ephemeral-Line: eine Zeile, die laufend überschrieben wird.
pub fn clear_ephemeral_line() {
    print!("\r{:>160}\r", ""); let _ = io::stdout().flush();
}
pub fn print_ephemeral(s: &str) {
    print!("\r{}", s); let _ = io::stdout().flush();
}
pub fn end_ephemeral() {
    clear_ephemeral_line();
    println!();
}
