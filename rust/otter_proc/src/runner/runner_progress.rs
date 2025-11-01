///// Otter: Progress state + pretty bar (ratio+[time] merge), responsive to term width.
///// Schneefuchs: 200ms tick; bar fill colored; safe truncation; ASCII-only.
/// ///// Maus: Parses [n/m] and “68%”; snippet carried into tail if space permits.
///// Datei: rust/otter_proc/src/runner/runner_progress.rs

use std::time::{Duration, Instant};

use crate::runner::runner_term::{term_cols, print_ephemeral, CYA, GRN, RESET};

pub struct ProgressState {
    pub start: Instant,
    pub last_tick: Instant,
    pub spinner_idx: usize,
    pub last_snippet: String,
    pub best_builder_pct: Option<u32>,
}
impl ProgressState {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            last_tick: Instant::now(),
            spinner_idx: 0,
            last_snippet: String::new(),
            best_builder_pct: None,
        }
    }
}

pub fn progress_enabled() -> bool {
    // Default: enabled. Disable with OTTER_NO_PROGRESS=1/true.
    !matches!(
        std::env::var("OTTER_NO_PROGRESS").map(|v| v.to_ascii_lowercase()).as_deref(),
        Ok("1") | Ok("true")
    )
}

pub fn fmt_hms(mut secs: u64) -> String {
    let h = secs / 3600; secs %= 3600;
    let m = secs / 60;   let s = secs % 60;
    if h > 0 { format!("{:02}:{:02}:{:02}", h, m, s) } else { format!("{:02}:{:02}", m, s) }
}

pub fn last_nonempty_snippet(buf: &str, max_len: usize) -> String {
    let s = buf.trim_end_matches(&['\r','\n'][..]).trim();
    if s.is_empty() { String::new() } else {
        let mut t = s.replace('\t', " ");
        if t.len() > max_len { t.truncate(max_len); }
        t
    }
}

pub fn parse_percent(line: &str) -> Option<u32> {
    if let Some(pos) = line.find('%') {
        let left = &line[..pos];
        let digits: String = left.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() { return None; }
        let rev = digits.chars().rev().collect::<String>();
        if let Ok(v) = rev.parse::<u32>() { if v <= 100 { return Some(v); } }
    }
    None
}

/// Parse a ratio like “[17/45]” anywhere in the line → percent.
pub fn parse_ratio_percent(line: &str) -> Option<u32> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i + 3 < bytes.len() {
        if bytes[i] == b'[' {
            // read numerator
            let mut j = i + 1;
            let mut num: u64 = 0;
            let mut had_num = false;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                had_num = true;
                num = num * 10 + (bytes[j] - b'0') as u64;
                j += 1;
            }
            if had_num && j < bytes.len() && bytes[j] == b'/' {
                j += 1;
                // read denominator
                let mut den: u64 = 0;
                let mut had_den = false;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    had_den = true;
                    den = den * 10 + (bytes[j] - b'0') as u64;
                    j += 1;
                }
                if had_den && den > 0 {
                    let pct = ((num as f64 / den as f64) * 100.0).floor() as i64;
                    let pct = pct.clamp(0, 100) as u32;
                    return Some(pct);
                }
            }
        }
        i += 1;
    }
    None
}

/// Compose the progress line and print it ephemerally.
pub fn render_and_print(state: &mut ProgressState, phase: &str, predicted_ms: u128) {
    let elapsed_ms = state.start.elapsed().as_millis() as u128;

    // time-based pct (capped to 99% until finish to avoid “stuck at 100” while still running)
    let time_pct = if predicted_ms > 0 {
        let mut p = ((elapsed_ms as f64 / predicted_ms as f64) * 100.0).floor() as u32;
        if p > 99 { p = 99; }
        Some(p)
    } else { None };

    let raw_pct = match (state.best_builder_pct, time_pct) {
        (Some(b), Some(t)) => Some(b.max(t)),
        (Some(b), None)    => Some(b),
        (None,    Some(t)) => Some(t),
        (None,    None)    => None,
    };

    let cols = term_cols().max(40);
    let spinner = ['|','/','-','\\'];
    let spin = spinner[state.spinner_idx % spinner.len()];
    state.spinner_idx = (state.spinner_idx + 1) % spinner.len();

    let elapsed = fmt_hms((elapsed_ms/1000) as u64);
    let pred_s  = if predicted_ms > 0 { (predicted_ms/1000) as u64 } else { 0 };

    let pct_str = raw_pct.map(|v| format!("{:>3}%", v)).unwrap_or_else(|| "---%".into());

    // Prefix without bar.
    let prefix = format!("[PROG]  [{}] {}{}{} {}  pred={}s  {} ",
        phase.to_ascii_uppercase(),
        CYA, spin, RESET,
        elapsed, pred_s, pct_str
    );

    // Bar budget.
    let mut bar_cols = cols.saturating_sub(prefix.len() + 2);
    if bar_cols < 10 { bar_cols = 10; }
    if bar_cols > 60 { bar_cols = 60; }

    let fill = raw_pct.unwrap_or(0);
    let filled = (fill as usize * bar_cols) / 100;
    let mut s = String::with_capacity(bar_cols + 2);
    s.push('[');
    if filled > 0 {
        s.push_str(GRN);
        s.push_str(&"=".repeat(filled));
        s.push_str(RESET);
    }
    if filled < bar_cols {
        s.push_str(&"-".repeat(bar_cols - filled));
    }
    s.push(']');

    // Optional snippet tail if space remains.
    let mut tail = String::new();
    if !state.last_snippet.is_empty() {
        let used = prefix.len() + s.len() + 1;
        if cols > used {
            let max_tail = cols - used;
            let mut snip = state.last_snippet.clone();
            if snip.len() > max_tail { snip.truncate(max_tail); }
            tail.push(' ');
            tail.push_str(&snip);
        }
    }

    let line = format!("{}{}{}", prefix, s, tail);
    print_ephemeral(&line);

    state.last_tick = Instant::now();
}

/// 200 ms throttle reached?
pub fn due(state: &ProgressState) -> bool {
    state.last_tick.elapsed() >= Duration::from_millis(200)
}
