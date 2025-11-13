///// Otter: Bootstrap-Binary - baut `otter_proc` farbig und startet es mit allen CLI-Args.
///// Schneefuchs: Erzwingt Cargo-/rustc-Farben via ENV; aktiviert VT (Windows) ohne externe Crates; PS 5.1-tauglich.
///// Maus: Minimal, keine Extra-Parsinglogik; stdout/stderr werden durchgereicht (inherit).
///// Datei: rust/otter_proc/src/bin/boot.rs

use std::process::{Command, Stdio};

#[cfg(windows)]
fn enable_vt_windows() {
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
            if (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0 { return true; }
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0
        }

        let _ = try_enable(STD_OUTPUT_HANDLE);
        let _ = try_enable(STD_ERROR_HANDLE);
    }
}

#[cfg(not(windows))]
#[inline]
fn enable_vt_windows() { /* no-op */ }

fn main() {
    // VT/ANSI für die Cargo-Phase auf Windows aktivieren (so werden Cargo/rustc-Farben sichtbar).
    enable_vt_windows();

    // Alles hinter `--` an otter_proc durchreichen:
    let pass_args: Vec<String> = std::env::args().skip(1).collect();

    // 1) otter_proc bauen - farbige Ausgabe erzwingen und 1:1 anzeigen
    println!("[BOOT] building otter_proc …");
    let status = Command::new("cargo")
        .args(["build", "--release", "--bin", "otter_proc"])
        .env("CARGO_TERM_COLOR", "always")
        .env("RUSTC_COLOR_DIAGNOSTICS", "always")
        .env("TERM", "xterm-256color")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("[BOOT] failed to spawn cargo");

    if !status.success() {
        let code = status.code().unwrap_or(1);
        eprintln!("[BOOT] cargo build failed (code={code})");
        std::process::exit(code);
    }

    // 2) otter_proc starten - gibt seine eigene farbige/formatierte Ausgabe aus
    #[cfg(windows)]
    let exe = "target\\release\\otter_proc.exe";
    #[cfg(not(windows))]
    let exe = "target/release/otter_proc";

    println!("[BOOT] launching {}", exe);
    let status = Command::new(exe)
        .args(pass_args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("[BOOT] failed to spawn otter_proc");

    std::process::exit(status.code().unwrap_or(1));
}
