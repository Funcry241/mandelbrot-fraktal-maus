///// Otter: Prozessstart, Streaming-Logs & ephemere Progress-Zeile (Spinner, ETA, % aus Metrics & Builder); ANSI-Farbe aktiv.
///// Schneefuchs: Fehlercodes sauber weiterreichen; CWD optional; ENV-Overlay; .build_metrics zentral über build_metrics.rs (atomar, Seeding).
///// Maus: Ruhig bei Output-Dauerfeuer, 1 Hz-Throttle; keine externen Crates; Metrics-Log nur 1× pro Prozess; Warn-/Error-Triage farbig.
///// Datei: rust/otter_proc/src/runner.rs

mod runner_term;
mod runner_phase;
mod runner_classify;
mod runner_progress;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::build_metrics::BuildMetrics;
use runner_classify::classify_line;
use runner_progress::{ProgressState, render_progress_line};
use runner_term::{enable_ansi, out_err, out_info, out_warn, clear_ephemeral_line};

// ---- tiny helpers -----------------------------------------------------------

fn last_nonempty_snippet(buf: &str, max_len: usize) -> String {
    let s = buf.trim_end_matches(&['\r', '\n'][..]).trim();
    if s.is_empty() { String::new() } else {
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

fn progress_enabled() -> bool {
    // Default: enabled. Disable with OTTER_NO_PROGRESS=1
    std::env::var("OTTER_NO_PROGRESS")
        .map(|v| v == "0" || v.to_ascii_lowercase() == "false")
        .unwrap_or(true)
}

struct PhaseDetect { phase: String, sig: String }

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

// --- metrics: print only once per process ------------------------------------

static METRICS_PRINTED_ONCE: AtomicBool = AtomicBool::new(false);

// ---- main runner ------------------------------------------------------------

#[derive(Default)]
pub struct RunResult { pub code: i32 }

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

    // Metrics via BuildMetrics (inkl. optionalem Seed aus Vorgängerordner)
    let (mut metrics, metrics_file, seed_src) = BuildMetrics::load_or_seed(&workdir);
    if !METRICS_PRINTED_ONCE.swap(true, Ordering::SeqCst) {
        println!("[RUNNER] metrics={}", metrics_file.display());
        if let Some(src) = seed_src { println!("[RUNNER] metrics-seeded-from={}", src.display()); }
    }

    // Command vorbereiten
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
        Some(s) => BufReader::new(s),
        None => { out_err("RUN", "failed to take stdout"); return RunResult { code: 1 }; }
    };
    let mut err_reader = match child.stderr.take() {
        Some(s) => BufReader::new(s),
        None => { out_err("RUN", "failed to take stderr"); return RunResult { code: 1 }; }
    };

    // Vorhersage aus Metrics (für diese Phase)
    let predicted_ms = metrics.get_last_ms(&phase_sig.sig, &phase_sig.phase).unwrap_or(0);

    // Progress-Status in eigenem Struct (verhindert Borrow-Probleme)
    let mut pstate = ProgressState::new();

    let start = Instant::now();
    let mut out_buf = String::new();
    let mut err_buf = String::new();

    // Tag (farbiges Herkunftslabel liegt in runner_classify / runner_term)
    let tag = if exe.eq_ignore_ascii_case("cmake") {
        if phase_sig.phase == "build" { "CMAKE/BUILD" } else { "CMAKE" }
    } else { "PROC" };

    loop {
        let mut progressed = false;

        out_buf.clear();
        if let Ok(n) = out_reader.read_line(&mut out_buf) {
            if n > 0 {
                if pstate.last_emitted_ephemeral { clear_ephemeral_line(); pstate.last_emitted_ephemeral = false; }
                if let Some(p) = parse_percent(&out_buf) {
                    pstate.best_builder_pct = Some(pstate.best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
                }
                match classify_line(&out_buf) {
                    runner_classify::Sev::Err  => out_err (tag, &out_buf),
                    runner_classify::Sev::Warn => out_warn(tag, &out_buf),
                    runner_classify::Sev::Info => out_info(tag, &out_buf),
                }
                pstate.last_snippet = last_nonempty_snippet(&out_buf, 120);
                progressed = true;
            }
        }

        err_buf.clear();
        if let Ok(n) = err_reader.read_line(&mut err_buf) {
            if n > 0 {
                if pstate.last_emitted_ephemeral { clear_ephemeral_line(); pstate.last_emitted_ephemeral = false; }
                if let Some(p) = parse_percent(&err_buf) {
                    pstate.best_builder_pct = Some(pstate.best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
                }
                match classify_line(&err_buf) {
                    runner_classify::Sev::Err  => out_err (tag, &err_buf),
                    runner_classify::Sev::Warn => out_warn(tag, &err_buf),
                    runner_classify::Sev::Info => out_info(tag, &err_buf),
                }
                let snip = last_nonempty_snippet(&err_buf, 120);
                if !snip.is_empty() { pstate.last_snippet = snip; }
                progressed = true;
            }
        }

        if !progressed {
            if progress_enabled() && pstate.tick_due() {
                let elapsed_ms = start.elapsed().as_millis() as u128;
                let line = render_progress_line(&mut pstate, elapsed_ms, predicted_ms, &phase_sig.phase);
                print!("\r{}", line);
                let _ = std::io::stdout().flush();
                pstate.last_emitted_ephemeral = true;
            }

            match child.try_wait() {
                Ok(Some(st)) => {
                    if pstate.last_emitted_ephemeral {
                        clear_ephemeral_line();
                        println!("[INFO]  [RUN] phase={} done (elapsed={}s)", phase_sig.phase, start.elapsed().as_secs());
                    }
                    let elapsed_ms = start.elapsed().as_millis() as u128;
                    metrics.upsert_phase_ms(&phase_sig.sig, &phase_sig.phase, elapsed_ms);
                    let _ = metrics.save(&workdir);
                    return RunResult { code: st.code().unwrap_or(1) };
                }
                Ok(None) => { std::thread::sleep(std::time::Duration::from_millis(10)); }
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
