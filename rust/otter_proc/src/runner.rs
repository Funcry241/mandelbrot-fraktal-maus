///// Otter: Prozessstart, Streaming-Logs & tickende Progressbar (Spinner, %/ETA, EMA) – ANSI-Farbe.
// ///// Schneefuchs: Fehlercodes sauber; CWD/ENV optional; Metrics via build_metrics.rs (atomar, Seeding).
///// Maus: 1 Hz Ticker; ephemer bei Idle, persistent bei Output; keine externen Crates.
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
const GRN:   &str = "\x1b[32m";
const YEL:   &str = "\x1b[33m";
const BLU:   &str = "\x1b[34m";
const MAG:   &str = "\x1b[35m";
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

// Pick a color for the source tag, e.g. [CMAKE], [PROC], [RUN], [PS]
fn color_for_source(src: &str) -> &'static str {
    let up = src.to_ascii_uppercase();
    if up.starts_with("CMAKE") {
        MAG
    } else if up == "RUN" {
        CYA
    } else if up == "PROC" {
        GRN
    } else if up == "PS" {
        BLU
    } else {
        CYA
    }
}

// --- simple tagged output helpers (with color for tag + source) ---------------
pub fn out_info(source: &str, msg: &str) {
    let c = color_for_source(source);
    println!("[INFO]  [{}{}{}] {}", c, source, RESET, msg);
}
pub fn out_warn(source: &str, msg: &str) {
    let c = color_for_source(source);
    println!("{}[WARN]{}  [{}{}{}] {}", YEL, RESET, c, source, RESET, msg);
}
pub fn out_err(source: &str, msg: &str) {
    let c = color_for_source(source);
    eprintln!("{}[ERR]{}   [{}{}{}] {}", RED, RESET, c, source, RESET, msg);
}

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

// ASCII progress bar
fn progress_bar(pct: u32, width: usize) -> String {
    let width = width.max(3);
    let filled = ((pct as usize) * width) / 100;
    let filled = filled.min(width);
    let mut s = String::with_capacity(width + 2);
    s.push('[');
    for _ in 0..filled { s.push('#'); }
    for _ in filled..width { s.push('-'); }
    s.push(']');
    s
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

    // Predicted duration for current phase from metrics (log once per phase start)
    let predicted_ms = metrics.get_last_ms(&phase_sig.sig, &phase_sig.phase).unwrap_or(0);
    if predicted_ms > 0 {
        println!("[RUNNER] predicted_total={}s phase={}", (predicted_ms / 1000) as u64, phase_sig.phase);
    }

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

    // Progress state
    const TICK: Duration = Duration::from_secs(1);
    const RECENT_OUTPUT: Duration = Duration::from_millis(900);
    const BAR_W: usize = 26;
    const EMA_A: f32 = 0.25; // smoothing for pct/ETA

    let start = Instant::now();
    let mut last_tick = Instant::now();
    let mut last_output = Instant::now().checked_sub(Duration::from_secs(10)).unwrap_or_else(Instant::now);
    let mut last_snippet = String::new();
    let mut last_emitted_ephemeral = false;
    let mut best_builder_pct: Option<u32> = None;
    let mut smooth_pct: Option<f32> = None;
    let spinner = ['|', '/', '-', '\\'];
    let mut spin_idx: usize = 0;

    let mut out_buf = String::new();
    let mut err_buf = String::new();

    let tag = if exe.eq_ignore_ascii_case("cmake") {
        if phase_sig.phase == "build" { "CMAKE/BUILD" } else { "CMAKE" }
    } else { "PROC" };

    loop {
        let mut progressed = false;

        // --- STDOUT ---
        out_buf.clear();
        if let Ok(n) = out_reader.read_line(&mut out_buf) {
            if n > 0 {
                if last_emitted_ephemeral {
                    print!("\r{:>120}\r", ""); let _ = std::io::stdout().flush();
                    last_emitted_ephemeral = false;
                }
                if let Some(p) = parse_percent(&out_buf) {
                    best_builder_pct = Some(best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
                }
                let trimmed = out_buf.trim_end_matches(&['\r','\n'][..]);
                match classify_line(trimmed) {
                    Sev::Err  => out_err (tag, trimmed),
                    Sev::Warn => out_warn(tag, trimmed),
                    Sev::Info => out_info(tag, trimmed),
                }
                last_snippet = last_nonempty_snippet(trimmed, 80);
                progressed = true;
                last_output = Instant::now();
            }
        }

        // --- STDERR ---
        err_buf.clear();
        if let Ok(n) = err_reader.read_line(&mut err_buf) {
            if n > 0 {
                if last_emitted_ephemeral {
                    print!("\r{:>120}\r", ""); let _ = std::io::stdout().flush();
                    last_emitted_ephemeral = false;
                }
                if let Some(p) = parse_percent(&err_buf) {
                    best_builder_pct = Some(best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
                }
                let trimmed = err_buf.trim_end_matches(&['\r','\n'][..]);
                match classify_line(trimmed) {
                    Sev::Err  => out_err (tag, trimmed),
                    Sev::Warn => out_warn(tag, trimmed),
                    Sev::Info => out_info(tag, trimmed),
                }
                let snip = last_nonempty_snippet(trimmed, 80);
                if !snip.is_empty() { last_snippet = snip; }
                progressed = true;
                last_output = Instant::now();
            }
        }

        // --- PROGRESS TICK (always once per second) ---
        let now = Instant::now();
        if progress_enabled() && now.duration_since(last_tick) >= TICK {
            last_tick = now;
            let elapsed_ms = start.elapsed().as_millis() as u128;

            // time-based pct from metrics if available
            let time_pct = if predicted_ms > 0 {
                let mut p = ((elapsed_ms as f64 / predicted_ms as f64) * 100.0).floor() as u32;
                if p > 99 { p = 99; } // never claim 100 until done
                Some(p)
            } else { None };

            // raw combined pct
            let raw_pct_opt = match (best_builder_pct, time_pct) {
                (Some(b), Some(t)) => Some(b.max(t)),
                (Some(b), None)    => Some(b),
                (None,    Some(t)) => Some(t),
                (None,    None)    => None,
            };

            // EMA smoothing
            if let Some(raw) = raw_pct_opt {
                let r = raw as f32;
                smooth_pct = Some(match smooth_pct {
                    None => r,
                    Some(prev) => prev + EMA_A * (r - prev),
                }.clamp(0.0, 99.0));
            }

            let shown_pct_u32 = smooth_pct.map(|v| v.floor() as u32).or(raw_pct_opt);
            let pct_str = shown_pct_u32.map(|v| format!("{:>3}%", v)).unwrap_or("---%".into());
            let bar = progress_bar(shown_pct_u32.unwrap_or(0), BAR_W);

            let elapsed = fmt_hms((elapsed_ms / 1000) as u64);
            let spin = spinner[spin_idx % spinner.len()]; spin_idx = (spin_idx + 1) % spinner.len();
            let snippet = if last_snippet.is_empty() { "" } else { &last_snippet };
            let pred_secs = if predicted_ms > 0 { (predicted_ms/1000) as u64 } else { 0 };

            // If output was very recent, emit a *persistent* PROG line (newline),
            // so motion is visible even while logs are flowing. Otherwise, ephemeral (\r…).
            if progressed || now.duration_since(last_output) <= RECENT_OUTPUT {
                if last_emitted_ephemeral {
                    print!("\r{:>120}\r", ""); let _ = std::io::stdout().flush();
                    last_emitted_ephemeral = false;
                }
                println!("[PROG]  [{}] {}{} {}  pred={}s  {}  {}  {}{}",
                    phase_sig.phase.to_ascii_uppercase(),
                    CYA, spin, elapsed,
                    pred_secs,
                    pct_str, bar, snippet, RESET
                );
            } else {
                print!("\r[PROG]  [{}] {}{} {}  pred={}s  {}  {}  {}{}",
                    phase_sig.phase.to_ascii_uppercase(),
                    CYA, spin, elapsed,
                    pred_secs,
                    pct_str, bar, snippet, RESET
                );
                let _ = std::io::stdout().flush();
                last_emitted_ephemeral = true;
            }
        }

        // --- Process completion / wait ---
        if !progressed {
            match child.try_wait() {
                Ok(Some(st)) => {
                    // finalize ephemeral line
                    if last_emitted_ephemeral {
                        print!("\r{:>120}\r", ""); let _ = std::io::stdout().flush();
                        println!("[INFO]  [RUN] phase={} done (elapsed={}s)", phase_sig.phase, start.elapsed().as_secs());
                    }
                    // update metrics with measured duration
                    let elapsed_ms = start.elapsed().as_millis() as u128;
                    metrics.upsert_phase_ms(&phase_sig.sig, &phase_sig.phase, elapsed_ms);
                    let _ = metrics.save(&workdir);

                    return RunResult { code: st.code().unwrap_or(1) };
                }
                Ok(None) => { std::thread::sleep(Duration::from_millis(10)); }
                Err(e) => {
                    out_err("RUN", &format!("wait failed: {}", e));
                    return RunResult { code: 1 };
                }
            }
        }
    }
}

// Beibehaltener Name für Abwärtskompatibilität (falls extern benutzt)
#[allow(dead_code)]
pub fn run_streamed(exe: &str, args: &[String]) -> RunResult {
    run_streamed_with_env(exe, args, None, None)
}
