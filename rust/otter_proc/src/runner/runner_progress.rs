///// Otter: Fortschrittsbalken/ETA-Renderer (ASCII), dynamische Breite, Spinner.
/// ///// Schneefuchs: Keine externen Crates; robust bei fehlenden Metrics (pred=—).
/// ///// Maus: 1 Hz-Takt gedacht; liefert fertige Zeile als String.
/// ///// Datei: rust/otter_proc/src/runner/runner_progress.rs

use std::env;
use std::time::Instant;

use crate::runner::runner_term::{CYA, RESET};

pub struct ProgressState {
    pub start: Instant,
    pub last_tick: Instant,
    pub spin_idx: usize,
}

impl ProgressState {
    pub fn new() -> Self {
        let now = Instant::now();
        Self { start: now, last_tick: now, spin_idx: 0 }
    }
}

fn term_cols() -> usize {
    if let Ok(v) = env::var("COLUMNS") {
        if let Ok(n) = v.parse::<usize>() {
            return n.clamp(60, 160);
        }
    }
    100
}

fn fmt_hms(mut secs: u64) -> String {
    let h = secs / 3600; secs %= 3600;
    let m = secs / 60;   let s = secs % 60;
    if h > 0 { format!("{:02}:{:02}:{:02}", h, m, s) } else { format!("{:02}:{:02}", m, s) }
}

pub fn render_progress_line(
    phase: &str,
    pct_opt: Option<u32>,
    elapsed_ms: u128,
    predicted_ms: u128,
    last_snippet: &str,
    spin_idx: usize,
) -> String {
    let cols = term_cols();

    let spinner = ['|', '/', '-', '\\'];
    let spin = spinner[spin_idx % spinner.len()];

    let elapsed = fmt_hms((elapsed_ms / 1000) as u64);
    let pred_s  = if predicted_ms > 0 { (predicted_ms / 1000) as u64 } else { 0 };
    let pct_str = pct_opt.map(|v| format!("{:>3}%", v)).unwrap_or_else(|| "---%".into());

    let prefix = format!("[PROG]  [{}] {}{}{} {}", phase.to_ascii_uppercase(), CYA, spin, RESET, elapsed);
    let mid    = if pred_s > 0 { format!(" pred={}s  {}", pred_s, pct_str) }
                 else           { format!(" pred=—   {}", pct_str) };

    let mut bar_cols = cols.saturating_sub(prefix.len() + mid.len() + 2 /*spaces*/ + 2 /*[]*/ + 1 /*space*/);
    bar_cols = bar_cols.clamp(10, 60);

    let filled = pct_opt.map(|v| ((v as usize) * bar_cols) / 100).unwrap_or(0);
    let mut bar = String::with_capacity(bar_cols + 2);
    bar.push('[');
    for i in 0..bar_cols {
        if i < filled { bar.push('='); } else { bar.push('-'); }
    }
    bar.push(']');

    let snip = {
        let s = last_snippet.trim();
        if s.is_empty() { "".to_string() }
        else {
            // Platz übrig? Wir kappen, um nicht umzubrechen.
            let mut max = cols.saturating_sub(prefix.len() + mid.len() + bar.len() + 6);
            if max < 0 { max = 0; }
            let mut ss = s.replace('\t', " ");
            if ss.len() > max { ss.truncate(max); }
            ss
        }
    };

    if snip.is_empty() {
        format!("{} {} {}", prefix, mid, bar)
    } else {
        format!("{} {} {}  {}", prefix, mid, bar, snip)
    }
}
