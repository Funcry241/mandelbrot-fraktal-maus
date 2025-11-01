///// Otter: Prozessstart, Streaming-Logs & ephemere Progress-Zeile (Spinner, ETA, % aus Metrics & Builder); ANSI-Farbe aktiv.
///// Schneefuchs: Fehlercodes sauber weiterreichen; CWD optional; ENV-Overlay; .build_metrics zentral über build_metrics.rs (atomar, Seeding).
///// Maus: Ruhig bei Output-Dauerfeuer, 1 Hz-Throttle; keine externen Crates; Metrics-Log nur 1× pro Prozess; Warn-/Error-Triage farbig.
///// Datei: rust/otter_proc/src/runner.rs

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::build_metrics::BuildMetrics;

// --- ANSI colors (always on; best-effort enable on Windows) ------------------
const RED:   &str = "\x1b[31m";
const YEL:   &str = "\x1b[33m";
const CYA:   &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

#[cfg(windows)]
fn enable_ansi() {
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
fn enable_ansi() { /* no-op */ }

// --- simple tagged output helpers (with color) --------------------------------
pub fn out_info(source: &str, msg: &str) { println!("[INFO]  [{}] {}", source, msg); }
pub fn out_warn(source: &str, msg: &str) { println!("{}[WARN]{}  [{}] {}", YEL, RESET, source, msg); }
pub fn out_err (source: &str, msg: &str) { eprintln!("{}[ERR]{}   [{}] {}",  RED, RESET, source, msg); }

#[derive(Default)]
pub struct RunResult { pub code: i32 }

// ---- progress helpers --------------------------------------------------------

fn fmt_hms(mut secs: u64) -> String {
    let h = secs / 3600; secs %= 3600;
    let m = secs / 60;   let s = secs % 60;
    if h > 0 { format!("{:02}:{:02}:{:02}", h, m, s) } else { format!("{:02}:{:02}", m, s) }
}

fn progress_enabled() -> bool {
    // Default: enabled. Disable with OTTER_NO_PROGRESS=1
    std::env::var("OTTER_NO_PROGRESS").map(|v| v == "0" || v.to_ascii_lowercase() == "false").unwrap_or(true)
}

fn last_nonempty_snippet(buf: &str, max_len: usize) -> String {
    let s = buf.trim_end_matches(&['\r', '\n'][..]).trim();
    if s.is_empty() { String::new() }
    else {
        let mut snip = s.replace('\t', " ");
        if snip.len() > max_len { snip.truncate(max_len); }
        snip
    }
}

// very lightweight %[0-100] parser (e.g. "[ 68%]" or " 68% ]")
fn parse_percent(line: &str) -> Option<u32> {
    if let Some(pos) = line.find('%') {
        let left = &line[..pos];
        let digits: String = left.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() { return None; }
        let rev = digits.chars().rev().collect::<String>();
        if let Ok(v) = rev.parse::<u32>() { if v <= 100 { return Some(v); } }
    }
    None
}

struct PhaseDetect {
    phase: String,
    sig: String,
}

fn detect_phase_and_sig(exe: &str, args: &[String]) -> PhaseDetect {
    let mut phase = "proc".to_string();
    let mut cfg: Option<String> = None;
    let mut preset: Option<String> = None;

    for i in 0..args.len() {
        if args[i] == "--config" && i + 1 < args.len()      { cfg = Some(args[i + 1].clone()); }
        else if args[i] == "--preset" && i + 1 < args.len() { preset = Some(args[i + 1].clone()); }
        else if args[i] == "--build"                         { phase = "build".to_string(); }
    }

    if exe.eq_ignore_ascii_case("cmake") && phase != "build" { phase = "configure".to_string(); }

    let sig = if exe.eq_ignore_ascii_case("cmake") {
        format!("cmake:{}:{}:{}", phase, preset.unwrap_or_else(|| "-".into()), cfg.unwrap_or_else(|| "-".into()))
    } else {
        let mut short = String::new();
        for a in args.iter().take(4) { if !short.is_empty() { short.push(' '); } short.push_str(a); }
        format!("{}:{}", exe, short)
    };

    PhaseDetect { phase, sig }
}

// --- warn/error triage -------------------------------------------------------

#[derive(Copy, Clone, Eq, PartialEq)]
enum Sev { Info, Warn, Err }

fn is_error_line(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    l.contains("fatal error") || l.contains("error:")
        || l.contains(" cmake error") || l.starts_with("error")
        || l.contains("] error")
}

fn is_warning_line(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    if is_error_line(&l) { return false; }
    l.contains("warning") || l.contains(" cmake warning") || l.starts_with("warning")
}

fn classify_line(line: &str) -> Sev {
    if is_error_line(line) { Sev::Err }
    else if is_warning_line(line) { Sev::Warn }
    else { Sev::Info }
}

// --- metrics: print only once per process ------------------------------------

static METRICS_PRINTED_ONCE: AtomicBool = AtomicBool::new(false);

// ---- main runner ------------------------------------------------------------

pub fn run_streamed_with_env(
    exe: &str,
    args: &[String],
    env_overlay: Option<&HashMap<String,String>>,
    cwd: Option<&Path>
) -> RunResult {
    let workdir: PathBuf = match cwd {
        Some(d) => d.to_path_buf(),
        None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    };

    enable_ansi();

    // Metrics logging + optional seeding (via build_metrics.rs)
    let (mut metrics, metrics_file, seed_src) = BuildMetrics::load_or_seed(&workdir);
    if !METRICS_PRINTED_ONCE.swap(true, Ordering::SeqCst) {
        println!("[RUNNER] metrics={}", metrics_file.display());
        if let Some(src) = seed_src { println!("[RUNNER] metrics-seeded-from={}", src.display()); }
    }

    let mut cmd = Command::new(exe);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(d) = cwd { cmd.current_dir(d); }
    if let Some(envmap) = env_overlay { for (k,v) in envmap.iter() { cmd.env(k, v); } }

    let phase_sig = detect_phase_and_sig(exe, args);
    out_info("RUN", &format!("exe=\"{}\" phase={} sig={}", exe, phase_sig.phase, phase_sig.sig));

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => { out_err("RUN", &format!("spawn failed exe={} err={}", exe, e)); return RunResult { code: 1 }; }
    };

    let mut out_reader = match child.stdout.take() {
        Some(s) => BufReader::new(s), None => { out_err("RUN", "failed to take stdout"); return RunResult { code: 1 }; }
    };
    let mut err_reader = match child.stderr.take() {
        Some(s) => BufReader::new(s), None => { out_err("RUN", "failed to take stderr"); return RunResult { code: 1 }; }
    };

    // Predicted duration for current phase from metrics
    let predicted_ms = metrics.get_last_ms(&phase_sig.sig, &phase_sig.phase).unwrap_or(0);

    // Progress state
    let start = Instant::now();
    let mut last_tick = Instant::now();
    let mut last_snippet = String::new();
    let mut last_emitted_ephemeral = false;
    let mut best_builder_pct: Option<u32> = None;
    let spinner = ['|', '/', '-', '\\'];
    let mut spin_idx: usize = 0;

    let mut out_buf = String::new();
    let mut err_buf = String::new();

    let tag = if exe.eq_ignore_ascii_case("cmake") {
        if phase_sig.phase == "build" { "CMAKE/BUILD" } else { "CMAKE" }
    } else { "PROC" };

    loop {
        let mut progressed = false;

        out_buf.clear();
        if let Ok(n) = out_reader.read_line(&mut out_buf) {
            if n > 0 {
                if last_emitted_ephemeral { print!("\r{:>120}\r", ""); let _ = std::io::stdout().flush(); last_emitted_ephemeral = false; }
                if let Some(p) = parse_percent(&out_buf) { best_builder_pct = Some(best_builder_pct.map(|b| b.max(p)).unwrap_or(p)); }

                // Trim CR/LF to avoid double-blank lines, then route through tag printers
                let trimmed = out_buf.trim_end_matches(&['\r','\n'][..]);
                match classify_line(trimmed) {
                    Sev::Err  => out_err (tag, trimmed),
                    Sev::Warn => out_warn(tag, trimmed),
                    Sev::Info => out_info(tag, trimmed),
                }

                last_snippet = last_nonempty_snippet(trimmed, 80);
                progressed = true;
            }
        }

        err_buf.clear();
        if let Ok(n) = err_reader.read_line(&mut err_buf) {
            if n > 0 {
                if last_emitted_ephemeral { print!("\r{:>120}\r", ""); let _ = std::io::stdout().flush(); last_emitted_ephemeral = false; }
                if let Some(p) = parse_percent(&err_buf) { best_builder_pct = Some(best_builder_pct.map(|b| b.max(p)).unwrap_or(p)); }

                // Trim CR/LF to avoid double-blank lines, then route through tag printers
                let trimmed = err_buf.trim_end_matches(&['\r','\n'][..]);
                match classify_line(trimmed) {
                    Sev::Err  => out_err (tag, trimmed),
                    Sev::Warn => out_warn(tag, trimmed),
                    Sev::Info => out_info(tag, trimmed),
                }

                let snip = last_nonempty_snippet(trimmed, 80);
                if !snip.is_empty() { last_snippet = snip; }
                progressed = true;
            }
        }

        if !progressed {
            if progress_enabled() && last_tick.elapsed() >= Duration::from_secs(1) {
                last_tick = Instant::now();
                let elapsed_ms = start.elapsed().as_millis() as u128;

                let time_pct = if predicted_ms > 0 {
                    let mut p = ((elapsed_ms as f64 / predicted_ms as f64) * 100.0).floor() as u32;
                    if p > 99 { p = 99; }
                    Some(p)
                } else { None };

                let pct = match (best_builder_pct, time_pct) {
                    (Some(b), Some(t)) => Some(b.max(t)),
                    (Some(b), None)    => Some(b),
                    (None,    Some(t)) => Some(t),
                    (None,    None)    => None,
                };

                let elapsed = fmt_hms((elapsed_ms / 1000) as u64);
                let pct_str = pct.map(|v| format!("{:>3}%", v)).unwrap_or("---%".into());
                let spin = spinner[spin_idx % spinner.len()]; spin_idx = (spin_idx + 1) % spinner.len();
                let snippet = if last_snippet.is_empty() { "" } else { &last_snippet };

                print!("\r[PROG]  [{}] {}{} {}  pred={}s  {}  {}{}",
                    phase_sig.phase.to_ascii_uppercase(),
                    CYA, spin, elapsed,
                    if predicted_ms > 0 { (predicted_ms/1000) as u64 } else { 0 },
                    pct_str, snippet, RESET
                );
                let _ = std::io::stdout().flush();
                last_emitted_ephemeral = true;
            }

            match child.try_wait() {
                Ok(Some(st)) => {
                    if last_emitted_ephemeral { print!("\r{:>120}\r", ""); let _ = std::io::stdout().flush();
                        println!("[INFO]  [RUN] phase={} done (elapsed={}s)", phase_sig.phase, start.elapsed().as_secs());
                    }
                    let elapsed_ms = start.elapsed().as_millis() as u128;
                    metrics.upsert_phase_ms(&phase_sig.sig, &phase_sig.phase, elapsed_ms);
                    let _ = metrics.save(&workdir);
                    return RunResult { code: st.code().unwrap_or(1) };
                }
                Ok(None) => { std::thread::sleep(Duration::from_millis(10)); }
                Err(e) => { out_err("RUN", &format!("wait failed: {}", e)); return RunResult { code: 1 }; }
            }
        }
    }
}

// Beibehaltener Name für Abwärtskompatibilität (falls extern benutzt)
#[allow(dead_code)]
pub fn run_streamed(exe: &str, args: &[String]) -> RunResult {
    run_streamed_with_env(exe, args, None, None)
}
