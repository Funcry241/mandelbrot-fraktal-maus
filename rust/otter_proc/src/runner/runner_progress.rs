///// Otter: Fortschrittsanzeige – responsiver Balken, Spinner, ETA; nutzt Metrics-Prognose & Builder-% (max).
///// Schneefuchs: 1 Hz-Throttle per State; ANSI minimal lokal; keine externen Abhängigkeiten.
///// Maus: Einfache API: State + render_progress_line(...); Bar passt sich an Terminalbreite an.
///// Datei: rust/otter_proc/src/runner/runner_progress.rs

use std::time::{Duration, Instant};

// lokale, minimale Farben (bewusst dupliziert – unabhängig vom Terminalmodul)
const CYA:   &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

pub struct ProgressState {
    pub last_tick: Instant,
    pub last_emitted_ephemeral: bool,
    pub best_builder_pct: Option<u32>,
    pub last_snippet: String,
    pub spin_idx: usize,
}

impl ProgressState {
    pub fn new() -> Self {
        Self {
            last_tick: Instant::now(),
            last_emitted_ephemeral: false,
            best_builder_pct: None,
            last_snippet: String::new(),
            spin_idx: 0,
        }
    }

    #[inline]
    pub fn tick_due(&mut self) -> bool {
        if self.last_tick.elapsed() >= Duration::from_secs(1) {
            self.last_tick = Instant::now();
            true
        } else { false }
    }
}

// Default via new() – Instant hat kein Default
impl Default for ProgressState {
    fn default() -> Self { Self::new() }
}

// ---------- helpers -----------------------------------------------------------

fn fmt_hms(mut secs: u64) -> String {
    let h = secs / 3600; secs %= 3600;
    let m = secs / 60;   let s = secs % 60;
    if h > 0 { format!("{:02}:{:02}:{:02}", h, m, s) } else { format!("{:02}:{:02}", m, s) }
}

fn terminal_cols() -> usize {
    // Best effort: env COLUMNS → sonst 100
    if let Ok(c) = std::env::var("COLUMNS") {
        if let Ok(v) = c.parse::<usize>() { return v.max(40).min(240); }
    }
    100
}

fn clamp_u32(v: i32, lo: i32, hi: i32) -> u32 {
    v.max(lo).min(hi) as u32
}

// baut den Text der Progresszeile komplett zusammen
pub fn render_progress_line(
    st: &mut ProgressState,
    elapsed_ms: u128,
    predicted_ms: u128,
    phase: &str,
) -> String {
    // Prozent aus Zeitprognose (Metrics)
    let time_pct = if predicted_ms > 0 {
        let p = ((elapsed_ms as f64 / predicted_ms as f64) * 100.0).floor() as i32;
        Some(clamp_u32(p, 0, 99)) // nie 100% vor Abschluss
    } else { None };

    // Mergen: Builder% vs Zeit%
    let pct = match (st.best_builder_pct, time_pct) {
        (Some(b), Some(t)) => Some(b.max(t)),
        (Some(b), None)    => Some(b),
        (None,    Some(t)) => Some(t),
        (None,    None)    => None,
    };

    // Layout berechnen
    let cols = terminal_cols();
    // feste Teile kalkulieren
    let prefix = "[PROG]";
    let phase_str = phase.to_ascii_uppercase();
    let elapsed = fmt_hms((elapsed_ms / 1000) as u64);
    let pred_s  = if predicted_ms > 0 { (predicted_ms / 1000) as u64 } else { 0 };
    let pct_str = pct.map(|v| format!("{:>3}%", v)).unwrap_or_else(|| "---%".into());

    // Balkenbreite
    let reserved = prefix.len() + 2
        + 1 + phase_str.len() + 1
        + 1 + 5 + elapsed.len()
        + 2 + 7 + 1 + 5
        + 2 + pct_str.len() + 2;

    let bar_cols = if cols > reserved + 20 {
        (cols - reserved - 10).min(40).max(20)
    } else { 20 };

    // Balken rendern
    let fill_cols = if let Some(p) = pct {
        ((p as usize) * bar_cols) / 100
    } else { 0 };
    let mut bar = String::with_capacity(bar_cols + 2);
    bar.push('[');
    for i in 0..bar_cols {
        bar.push(if i < fill_cols { '=' } else { '-' });
    }
    bar.push(']');

    // Spinner
    let spinner = ['|', '/', '-', '\\'];
    let spin = spinner[st.spin_idx % spinner.len()];
    st.spin_idx = (st.spin_idx + 1) % spinner.len();

    // Snippet
    let snippet = if st.last_snippet.is_empty() { "" } else { &st.last_snippet };

    // Finale Zeile
    format!(
        "[PROG]  [{}] {}{}{}  {}  pred={}s  {}  {}  {}",
        phase_str,
        CYA, spin, RESET,
        elapsed,
        pred_s,
        pct_str,
        bar,
        snippet
    )
}
