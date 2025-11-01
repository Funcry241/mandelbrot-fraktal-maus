///// Otter: Process runner with live progress (ratio+time), metrics-seeded ETA, colored tags.
/// ///// Schneefuchs: No external crates; Windows+POSIX; prints durable logs & smooth 200ms animation.
/// ///// Maus: Parses both “68%” and “[17/45]”; merges with predicted; ASCII-only; safe saves to .build_metrics.
///// Datei: rust/otter_proc/src/runner.rs

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::build_metrics::BuildMetrics;

mod runner_term;
mod runner_progress;
mod runner_classify;
mod runner_phase; // placeholder module (keeps reorg noise low)

use runner_classify::{classify_line, Sev};
use runner_progress::{ProgressState, render_and_print, parse_percent, parse_ratio_percent, last_nonempty_snippet, progress_enabled, due};
use runner_term::{enable_ansi, out_err, out_info, out_warn, print_ephemeral, end_ephemeral};

#[derive(Default)]
pub struct RunResult { pub code: i32 }

static METRICS_PRINTED_ONCE: AtomicBool = AtomicBool::new(false);

struct PhaseDetect {
    phase: String,
    sig: String,
}

fn detect_phase_and_sig(exe: &str, args: &[String]) -> PhaseDetect {
    let mut phase = "proc".to_string();

    // We accept a few conventional “cmd glue” signatures from our PS runner.
    let mut is_build = false;
    for a in args {
        if a.contains("VsDevCmd.bat") && a.contains("-arch=") {
            // Our PS wrapper: first call → env/cmake (proc), second call with “build” marker → build.
            if let Some(x) = args.iter().find(|s| s.as_str().eq_ignore_ascii_case("build")) {
                let _ = x; // hint for intent, not used
            }
        }
        if a.eq_ignore_ascii_case("build") || a.contains("cmake --build") {
            is_build = true;
        }
    }
    if exe.eq_ignore_ascii_case("cmake") && !is_build { phase = "configure".to_string(); }
    if is_build { phase = "build".to_string(); }

    // Signal used for metrics table key (stable & short)
    let sig = if exe.eq_ignore_ascii_case("cmake") {
        format!("cmake:{}", phase)
    } else if exe.eq_ignore_ascii_case("cmd") {
        format!("cmd:{}", phase)
    } else {
        let mut short = String::new();
        for a in args.iter().take(4) { if !short.is_empty() { short.push(' '); } short.push_str(a); }
        format!("{}:{}", exe, short)
    };

    PhaseDetect { phase, sig }
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

    // Metrics: load or seed, log only once per process
    let (mut metrics, metrics_file, seed_src) = BuildMetrics::load_or_seed(&workdir);
    if !METRICS_PRINTED_ONCE.swap(true, Ordering::SeqCst) {
        out_info("RUST", &format!("metrics={}", metrics_file.display()));
        if let Some(src) = seed_src { out_info("RUST", &format!("metrics-seeded-from={}", src.display())); }
    }

    // Spawn
    let mut cmd = Command::new(exe);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(d) = cwd { cmd.current_dir(d); }
    if let Some(envmap) = env_overlay { for (k,v) in envmap.iter() { cmd.env(k, v); } }

    let phase_sig = detect_phase_and_sig(exe, args);
    out_info("RUST", &format!("RUN exe=\"{}\" phase={} sig={}", exe, phase_sig.phase, phase_sig.sig));

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => { out_err("RUST", &format!("spawn failed exe={} err={}", exe, e)); return RunResult { code: 1 }; }
    };

    let mut out_reader = match child.stdout.take() {
        Some(s) => BufReader::new(s),
        None => { out_err("RUST", "failed to take stdout"); return RunResult { code: 1 }; }
    };
    let mut err_reader = match child.stderr.take() {
        Some(s) => BufReader::new(s),
        None => { out_err("RUST", "failed to take stderr"); return RunResult { code: 1 }; }
    };

    let predicted_ms = metrics.get_last_ms(&phase_sig.sig, &phase_sig.phase).unwrap_or(0);

    // Progress
    let mut pstate = ProgressState::new();

    // Tag for child streams in logs
    let tag = if exe.eq_ignore_ascii_case("cmd") { "PS" } else { "PROC" };

    let mut out_buf = String::new();
    let mut err_buf = String::new();

    loop {
        let mut progressed = false;

        out_buf.clear();
        if let Ok(n) = out_reader.read_line(&mut out_buf) {
            if n > 0 {
                // Make durable: end ephemeral first
                end_ephemeral();

                // Update progress knowledge from the line
                if let Some(p) = parse_percent(&out_buf).or_else(|| parse_ratio_percent(&out_buf)) {
                    pstate.best_builder_pct = Some(pstate.best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
                }
                let snip = last_nonempty_snippet(&out_buf, 120);
                if !snip.is_empty() { pstate.last_snippet = snip; }

                match classify_line(&out_buf) {
                    Sev::Err  => out_err (tag, &out_buf),
                    Sev::Warn => out_warn(tag, &out_buf),
                    Sev::Info => out_info(tag, &out_buf),
                }
                progressed = true;
            }
        }

        err_buf.clear();
        if let Ok(n) = err_reader.read_line(&mut err_buf) {
            if n > 0 {
                end_ephemeral();

                if let Some(p) = parse_percent(&err_buf).or_else(|| parse_ratio_percent(&err_buf)) {
                    pstate.best_builder_pct = Some(pstate.best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
                }
                let snip = last_nonempty_snippet(&err_buf, 120);
                if !snip.is_empty() { pstate.last_snippet = snip; }

                match classify_line(&err_buf) {
                    Sev::Err  => out_err (tag, &err_buf),
                    Sev::Warn => out_warn(tag, &err_buf),
                    Sev::Info => out_info(tag, &err_buf),
                }
                progressed = true;
            }
        }

        if !progressed {
            if progress_enabled() && due(&pstate) {
                render_and_print(&mut pstate, &phase_sig.phase, predicted_ms);
            }

            match child.try_wait() {
                Ok(Some(st)) => {
                    // finish: clear ephemeral, print done line, save metrics
                    end_ephemeral();
                    out_info("RUST", &format!("RUN phase={} done (elapsed={}s)", phase_sig.phase, pstate.start.elapsed().as_secs()));
                    let elapsed_ms = pstate.start.elapsed().as_millis() as u128;
                    metrics.upsert_phase_ms(&phase_sig.sig, &phase_sig.phase, elapsed_ms);
                    let _ = metrics.save(&workdir);
                    return RunResult { code: st.code().unwrap_or(1) };
                }
                Ok(None) => { std::thread::sleep(std::time::Duration::from_millis(10)); }
                Err(e) => { end_ephemeral(); out_err("RUST", &format!("wait failed: {}", e)); return RunResult { code: 1 }; }
            }
        }
    }
}

// Legacy name kept for back-compat (if externally used)
#[allow(dead_code)]
pub fn run_streamed(exe: &str, args: &[String]) -> RunResult {
    run_streamed_with_env(exe, args, None, None)
}
