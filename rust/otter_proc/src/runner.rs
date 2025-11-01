///// Otter: Prozessstart mit Streaming + hübscher Progress-Zeile & farbigen Herkunfts-Tags.
/// ///// Schneefuchs: Keine Borrow-Fallen; stabile Signaturen; Metrics genutzt wenn vorhanden.
/// ///// Maus: Spinner/ETA alle 200 ms; nach jeder PS-Zeile Ephemeral sofort wieder herstellen.
/// ///// Datei: rust/otter_proc/src/runner.rs

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::build_metrics::BuildMetrics;
mod runner_term;
mod runner_progress;
mod runner_classify;
mod runner_phase;

use runner_term::{enable_ansi, out_err, out_info, out_warn, clear_ephemeral_line, print_ephemeral, end_ephemeral};
use runner_progress::{ProgressState, render_progress_line};
use runner_classify::{classify_line, Sev};

#[derive(Default)]
pub struct RunResult { pub code: i32 }

fn progress_enabled() -> bool {
    // Default: enabled. Disable with OTTER_NO_PROGRESS=1
    std::env::var("OTTER_NO_PROGRESS").map(|v| v == "0" || v.to_ascii_lowercase() == "false").unwrap_or(true)
}

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

struct PhaseDetect { phase: String, sig: String }

fn detect_phase_and_sig(exe: &str, args: &[String]) -> PhaseDetect {
    let mut phase = "proc".to_string();
    if args.iter().any(|a| a == "--build") { phase = "build".to_string(); }
    let sig = if exe.eq_ignore_ascii_case("cmake") {
        let mut cfg: Option<&str> = None;
        let mut preset: Option<&str> = None;
        for i in 0..args.len() {
            if args[i] == "--config" && i+1 < args.len() { cfg = Some(&args[i+1]); }
            if args[i] == "--preset" && i+1 < args.len() { preset = Some(&args[i+1]); }
        }
        format!("cmake:{}:{}", preset.unwrap_or("-"), cfg.unwrap_or("-"))
    } else {
        if phase == "build" { "cmd:build".to_string() } else { "cmd:proc".to_string() }
    };
    PhaseDetect { phase, sig }
}

fn last_nonempty_snippet(s: &str, max_len: usize) -> String {
    let t = s.trim_end_matches(&['\r','\n'][..]).trim();
    if t.is_empty() { String::new() } else {
        let mut x = t.replace('\t', " ");
        if x.len() > max_len { x.truncate(max_len); }
        x
    }
}

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

    // Metrics laden/seed
    let (mut metrics, metrics_file, seed_src) = BuildMetrics::load_or_seed(&workdir);
    out_info("RUST", &format!("metrics={}", metrics_file.display()));
    if let Some(src) = seed_src { out_info("RUST", &format!("metrics-seeded-from={}", src.display())); }

    let phase_sig = detect_phase_and_sig(exe, args);
    out_info("RUST", &format!("RUN exe=\"{}\" phase={} sig={}", exe, &phase_sig.phase, &phase_sig.sig));

    let mut cmd = Command::new(exe);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(d) = cwd { cmd.current_dir(d); }
    if let Some(envmap) = env_overlay { for (k,v) in envmap.iter() { cmd.env(k, v); } }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => { out_err("RUST", &format!("spawn failed exe={} err={}", exe, e)); return RunResult { code: 1 }; }
    };

    let mut out_reader = match child.stdout.take() {
        Some(s) => BufReader::new(s), None => { out_err("RUST", "failed to take stdout"); return RunResult { code: 1 }; }
    };
    let mut err_reader = match child.stderr.take() {
        Some(s) => BufReader::new(s), None => { out_err("RUST", "failed to take stderr"); return RunResult { code: 1 }; }
    };

    // Predicted Dauer für Phase aus Metrics (fallback 0)
    let predicted_ms = metrics.get_last_ms(&phase_sig.sig, &phase_sig.phase).unwrap_or(0);

    let start = Instant::now();
    let mut state = ProgressState::new();
    let mut best_builder_pct: Option<u32> = None;
    let mut last_snippet = String::new();
    let mut ephemeral_visible = false;
    let mut tick = Instant::now();
    let tick_interval = Duration::from_millis(200);

    let mut out_buf = String::new();
    let mut err_buf = String::new();

    loop {
        let mut progressed = false;

        // --- STDOUT
        out_buf.clear();
        if let Ok(n) = out_reader.read_line(&mut out_buf) {
            if n > 0 {
                if ephemeral_visible { clear_ephemeral_line(); ephemeral_visible = false; }
                if let Some(p) = parse_percent(&out_buf) {
                    best_builder_pct = Some(best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
                }
                match classify_line(&out_buf) {
                    Sev::Err  => out_err ("PS", &out_buf),
                    Sev::Warn => out_warn("PS", &out_buf),
                    Sev::Info => out_info("PS", &out_buf),
                }
                last_snippet = last_nonempty_snippet(&out_buf, 120);
                progressed = true;
            }
        }

        // --- STDERR
        err_buf.clear();
        if let Ok(n) = err_reader.read_line(&mut err_buf) {
            if n > 0 {
                if ephemeral_visible { clear_ephemeral_line(); ephemeral_visible = false; }
                if let Some(p) = parse_percent(&err_buf) {
                    best_builder_pct = Some(best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
                }
                match classify_line(&err_buf) {
                    Sev::Err  => out_err ("PS", &err_buf),
                    Sev::Warn => out_warn("PS", &err_buf),
                    Sev::Info => out_info("PS", &err_buf),
                }
                let snip = last_nonempty_snippet(&err_buf, 120);
                if !snip.is_empty() { last_snippet = snip; }
                progressed = true;
            }
        }

        // Nach realem Output die Ephemeral-Zeile sofort wieder hinstellen:
        if progressed && progress_enabled() {
            let elapsed_ms = start.elapsed().as_millis();
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

            let line = render_progress_line(&phase_sig.phase, pct, elapsed_ms, predicted_ms, &last_snippet, state.spin_idx);
            print_ephemeral(&line);
            ephemeral_visible = true;
        }

        // Tick-getriebene Aktualisierung (Animation), auch wenn keine Zeile kam:
        if progress_enabled() && tick.elapsed() >= tick_interval {
            tick = Instant::now();
            state.spin_idx = (state.spin_idx + 1) % 4;

            let elapsed_ms = start.elapsed().as_millis();
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

            let line = render_progress_line(&phase_sig.phase, pct, elapsed_ms, predicted_ms, &last_snippet, state.spin_idx);
            print_ephemeral(&line);
            ephemeral_visible = true;
        }

        // Prozessende?
        match child.try_wait() {
            Ok(Some(st)) => {
                if ephemeral_visible {
                    clear_ephemeral_line();
                    out_info("RUST", &format!("RUN phase={} done (elapsed={}s)", &phase_sig.phase, start.elapsed().as_secs()));
                    end_ephemeral();
                }
                let elapsed_ms = start.elapsed().as_millis() as u128;
                metrics.upsert_phase_ms(&phase_sig.sig, &phase_sig.phase, elapsed_ms);
                let _ = metrics.save(&workdir);
                return RunResult { code: st.code().unwrap_or(1) };
            }
            Ok(None) => { std::thread::sleep(Duration::from_millis(10)); }
            Err(e) => { out_err("RUST", &format!("wait failed: {}", e)); return RunResult { code: 1 }; }
        }
    }
}

#[allow(dead_code)]
pub fn run_streamed(exe: &str, args: &[String]) -> RunResult {
    run_streamed_with_env(exe, args, None, None)
}
