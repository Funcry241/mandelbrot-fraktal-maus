///// Otter: Progress UI — spinner, ETA, percent merge (builder vs time), ASCII bar.
///// Schneefuchs: 200 ms throttle; width auto via env (OTTER_TERM_COLS/COLUMNS) with sane fallback.
///// Maus: ASCII-only; trims/snips messages; ratio parser “[n/m]” and plain “68%”.
///// Datei: rust/otter_proc/src/runner/runner_progress.rs

use std::time::{Duration, Instant};

use crate::runner::runner_term::print_ephemeral;
use crate::runner::runner_term::term_cols;

// Public state carried by runner.rs
pub struct ProgressState {
    pub start: Instant,
    pub last_tick: Instant,
    pub last_print_len: usize,
    pub runtime_phase: String,
    pub last_snippet: String,
    pub best_builder_pct: Option<u32>,
    spinner_idx: usize,
}

impl ProgressState {
    pub fn new(phase: &str) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_tick: now.checked_sub(Duration::from_millis(1000)).unwrap_or(now),
            last_print_len: 0,
            runtime_phase: phase.to_string(),
            last_snippet: String::new(),
            best_builder_pct: None,
            spinner_idx: 0,
        }
    }
}

// --- helpers ------------------------------------------------------------------

pub fn progress_enabled() -> bool {
    // default on; disable with OTTER_NO_PROGRESS=1 / true
    match std::env::var("OTTER_NO_PROGRESS") {
        Ok(v) => {
            let s = v.to_ascii_lowercase();
            !(s == "1" || s == "true" || s == "yes")
        }
        Err(_) => true,
    }
}

pub fn due(p: &ProgressState) -> bool {
    p.last_tick.elapsed() >= Duration::from_millis(200)
}

pub fn last_nonempty_snippet(s: &str, max_len: usize) -> String {
    let t = s.trim();
    if t.is_empty() { return String::new(); }
    let mut snip = t.replace('\t', " ");
    snip = snip.split_whitespace().collect::<Vec<_>>().join(" ");
    if snip.len() > max_len { snip.truncate(max_len); }
    snip
}

pub fn parse_percent(line: &str) -> Option<u32> {
    // find "... 68% ..." anywhere
    if let Some(pos) = line.find('%') {
        let left = &line[..pos];
        let digits: String = left.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() { return None; }
        let rev = digits.chars().rev().collect::<String>();
        if let Ok(v) = rev.parse::<u32>() { if v <= 100 { return Some(v); } }
    }
    None
}

pub fn parse_ratio_percent(line: &str) -> Option<u32> {
    // matches e.g. "[17/45]" or "17/45" (loose)
    let bytes = line.as_bytes();
    let mut num = String::new();
    let mut den = String::new();
    let mut in_num = false;
    let mut in_den = false;
    for &b in bytes {
        let c = b as char;
        if !in_num && c.is_ascii_digit() {
            in_num = true;
            num.push(c);
        } else if in_num && !in_den && c.is_ascii_digit() {
            num.push(c);
        } else if in_num && !in_den && c == '/' {
            in_den = true;
        } else if in_den && c.is_ascii_digit() {
            den.push(c);
        } else if in_den {
            break;
        }
    }
    if num.is_empty() || den.is_empty() { return None; }
    let n = num.parse::<u32>().ok()?;
    let d = den.parse::<u32>().ok()?;
    if d == 0 { return None; }
    let pct = ((n as f64 / d as f64) * 100.0).floor() as u32;
    if pct <= 100 { Some(pct) } else { Some(100) }
}

fn fmt_hms(mut secs: u64) -> String {
    let h = secs / 3600; secs %= 3600;
    let m = secs / 60;   let s = secs % 60;
    if h > 0 { format!("{:02}:{:02}:{:02}", h, m, s) } else { format!("{:02}:{:02}", m, s) }
}

fn bar_string(cols: usize, pct: Option<u32>) -> String {
    let width = cols.min(40).max(10);
    let p = pct.unwrap_or(0).min(100);
    let filled = ((p as usize * width) / 100).min(width);
    let mut s = String::with_capacity(width + 2);
    s.push('[');
    for i in 0..width {
        if i < filled { s.push('='); } else { s.push('-'); }
    }
    s.push(']');
    s
}

fn spinner_char(idx: usize) -> char {
    ['|','/','-','\\'][idx % 4]
}

pub fn render_and_print(p: &mut ProgressState, predicted_ms: u128) {
    p.last_tick = Instant::now();

    let elapsed_ms = p.start.elapsed().as_millis() as u128;
    let time_pct = if predicted_ms > 0 {
        let mut v = ((elapsed_ms as f64 / predicted_ms as f64) * 100.0).floor() as u32;
        if v > 99 { v = 99; } // never claim 100% before done
        Some(v)
    } else { None };

    let pct = match (p.best_builder_pct, time_pct) {
        (Some(b), Some(t)) => Some(b.max(t)),
        (Some(b), None)    => Some(b),
        (None,    Some(t)) => Some(t),
        (None,    None)    => None,
    };

    let cols = term_cols();
    let spin = spinner_char(p.spinner_idx);
    p.spinner_idx = (p.spinner_idx + 1) % 4;

    let elapsed = fmt_hms((elapsed_ms / 1000) as u64);
    let eta = if predicted_ms > 0 && elapsed_ms < predicted_ms {
        let remain = ((predicted_ms - elapsed_ms) / 1000) as u64;
        fmt_hms(remain)
    } else { "--:--".into() };

    let pct_str = pct.map(|v| format!("{:>3}%", v)).unwrap_or("---%".into());
    let bar = bar_string((cols / 3).max(10), pct);
    let pred_s = if predicted_ms > 0 { (predicted_ms/1000) as u64 } else { 0 };

    // Compose line
    let left = format!("[PROG] [{}] {} {}  eta={}  pred={}s  {}", p.runtime_phase.to_ascii_uppercase(), spin, elapsed, eta, pred_s, pct_str);
    let mid  = format!(" {}", bar);
    let mut room = cols.saturating_sub(left.len() + mid.len() + 1);
    if room < 8 { room = 8; }
    let mut snip = p.last_snippet.clone();
    if snip.len() > room { snip.truncate(room); }

    let line = format!("{}{} {}", left, mid, snip);
    p.last_print_len = print_ephemeral(&line, p.last_print_len);
}
