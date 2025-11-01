///// Otter: Terminal helpers (ANSI enable, colors, ephemeral line, width).
///// Schneefuchs: Win32 width via GetConsoleScreenBufferInfo; POSIX via COLUMNS.
/// ///// Maus: Colorize tags (RUST/PS), ephemeral print/clear; ASCII-only.
///// Datei: rust/otter_proc/src/runner/runner_term.rs

use std::io::{self, Write};

pub const RED:   &str = "\x1b[31m";
pub const YEL:   &str = "\x1b[33m";
pub const GRN:   &str = "\x1b[32m";
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

        for h in [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
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

pub fn colorize_tag(tag: &str) -> String {
    if tag.eq_ignore_ascii_case("RUST") {
        format!("{}{}{}", CYA, tag, RESET)
    } else if tag.eq_ignore_ascii_case("PS") {
        format!("{}{}{}", GRN, tag, RESET)
    } else {
        tag.to_string()
    }
}

pub fn out_info(source: &str, msg: &str) {
    println!("[INFO]  [{}] {}", colorize_tag(source), msg.trim_end_matches(&['\r','\n'][..]));
}
pub fn out_warn(source: &str, msg: &str) {
    println!("{}[WARN]{}  [{}] {}", YEL, RESET, colorize_tag(source), msg.trim_end_matches(&['\r','\n'][..]));
}
pub fn out_err(source: &str, msg: &str) {
    eprintln!("{}[ERR]{}   [{}] {}",  RED, RESET, colorize_tag(source), msg.trim_end_matches(&['\r','\n'][..]));
}

/// Print/refresh the ephemeral progress line.
pub fn print_ephemeral(line: &str) {
    // \r — go to line start; \x1b[K — clear to end of line.
    print!("\r{}\x1b[K", line);
    let _ = io::stdout().flush();
}

/// Clear any ephemeral content and move to a fresh line for durable output.
pub fn end_ephemeral() {
    print!("\r\x1b[K");
    let _ = io::stdout().flush();
}

/// Best-effort terminal column width.
pub fn term_cols() -> usize {
    // 1) Windows API
    #[cfg(windows)]
    unsafe {
        use std::ffi::c_void;
        type HANDLE = *mut c_void;

        #[allow(non_snake_case)]
        #[repr(C)]
        struct COORD { x: i16, y: i16 }

        #[allow(non_snake_case)]
        #[repr(C)]
        struct SMALL_RECT { left: i16, top: i16, right: i16, bottom: i16 }

        #[allow(non_snake_case)]
        #[repr(C)]
        struct CONSOLE_SCREEN_BUFFER_INFO {
            _dwSize: COORD,
            _dwCursorPosition: COORD,
            _wAttributes: u16,
            srWindow: SMALL_RECT,
            _dwMaximumWindowSize: COORD,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetStdHandle(nStdHandle: i32) -> HANDLE;
            fn GetConsoleScreenBufferInfo(hConsoleOutput: HANDLE, lpConsoleScreenBufferInfo: *mut CONSOLE_SCREEN_BUFFER_INFO) -> i32;
        }
        const STD_OUTPUT_HANDLE: i32 = -11;
        let h = GetStdHandle(STD_OUTPUT_HANDLE);
        if !h.is_null() {
            let mut info: CONSOLE_SCREEN_BUFFER_INFO = std::mem::zeroed();
            if GetConsoleScreenBufferInfo(h, &mut info as *mut _) != 0 {
                let w = (info.srWindow.right - info.srWindow.left + 1) as i32;
                if w > 0 { return w as usize; }
            }
        }
    }

    // 2) POSIX: COLUMNS
    if let Ok(s) = std::env::var("COLUMNS") {
        if let Ok(v) = s.parse::<usize>() {
            if v >= 40 { return v; }
        }
    }

    // 3) Fallback
    100
}
