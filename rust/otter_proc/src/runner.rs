///// Otter: Process runner with live progress (spinner, ETA, %, ratio merge); pretty tags.
///// Schneefuchs: No external crates; trims blank lines; smooth 200 ms animation; saves metrics safely.
///// Maus: Colors for source ([RUST]/[PS]); ASCII bar; parses both “68%” and “[17/45]”; Windows+POSIX.
///// Datei: rust/otter_proc/src/runner.rs

use std::collections::HashMap;
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use crate::build_metrics::BuildMetrics;

mod runner_term;
mod runner_progress;
mod runner_classify;

use runner_classify::{classify_line, Sev};
use runner_progress::{
    ProgressState, render_and_print, parse_percent, parse_ratio_percent,
    last_nonempty_snippet, progress_enabled, due
};
use runner_term::{
    enable_ansi, out_err, out_info, out_warn, end_ephemeral, sanitize_line,
    out_trailer_min, print_ephemeral,
};

#[derive(Default)]
pub struct RunResult { pub code: i32 }

static METRICS_PRINTED_ONCE: AtomicBool = AtomicBool::new(false);

struct PhaseDetect {
    phase: String,
    sig: String,
}

fn detect_phase_and_sig(exe: &str, args: &[String]) -> PhaseDetect {
    let mut phase = "proc".to_string();

    // Heuristic: try to infer build vs configure
    let mut is_build = false;
    for a in args {
        if a.eq_ignore_ascii_case("build") || a.contains("cmake --build") {
            is_build = true;
            break;
        }
    }
    if exe.eq_ignore_ascii_case("cmake") && !is_build { phase = "configure".to_string(); }
    if is_build { phase = "build".to_string(); }

    // Signal used for metrics key
    let sig = if exe.eq_ignore_ascii_case("cmake") {
        format!("cmake:{}", phase)
    } else if exe.eq_ignore_ascii_case("cmd") {
        format!("cmd:{}", phase)
    } else {
        let mut short = String::new();
        for a in args.iter().take(4) {
            if !short.is_empty() { short.push(' '); }
            short.push_str(a);
        }
        format!("{}:{}", exe, short)
    };

    PhaseDetect { phase, sig }
}

/// Trailer standardmäßig **an**.
/// Nur wenn OTTER_TRAILER=0|off gesetzt ist, wird er unterdrückt.
fn trailer_enabled() -> bool {
    match env::var("OTTER_TRAILER") {
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !(t == "0" || t == "off" || t == "no")
        }
        Err(_) => true, // Default: an
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

    // Sofortiger Start-Heartbeat, damit der Beginn nie „stuck“ wirkt.
    print_ephemeral("[proc] starting...");

    // Metrics: load or seed, log only once per process
    let (mut metrics, metrics_file, seed_src) = BuildMetrics::load_or_seed(&workdir);
    if !METRICS_PRINTED_ONCE.swap(true, Ordering::SeqCst) {
        out_info("RUST", &format!("metrics={}", metrics_file.display()));
        if let Some(src) = seed_src {
            out_info("RUST", &format!("metrics-seeded-from={}", src.display()));
        }
    }

    // Spawn child
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(d) = cwd { cmd.current_dir(d); }
    if let Some(envmap) = env_overlay {
        for (k,v) in envmap.iter() { cmd.env(k, v); }
    }

    let phase_sig = detect_phase_and_sig(exe, args);
    out_info("RUST", &format!("RUN exe=\"{}\" phase={} sig={}", exe, phase_sig.phase, phase_sig.sig));

    // Spawn-Latenz messen (z. B. Smartscreen/AV)
    let t_spawn0 = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => { out_err("RUST", &format!("spawn failed exe={} err={}", exe, e)); return RunResult { code: 1 }; }
    };
    let spawn_ms = t_spawn0.elapsed().as_millis();
    if spawn_ms > 400 {
        let _ = end_ephemeral();
        out_info("RUST", &format!("spawn-latency={}ms", spawn_ms));
    }

    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => { out_err("RUST", "failed to take stdout"); return RunResult { code: 1 }; }
    };
    let stderr = match child.stderr.take() {
        Some(s) => s,
        None => { out_err("RUST", "failed to take stderr"); return RunResult { code: 1 }; }
    };

    let predicted_ms = metrics.get_last_ms(&phase_sig.sig, &phase_sig.phase).unwrap_or(0);

    // Progress
    let mut pstate = ProgressState::new(&phase_sig.phase);

    // Tag for child streams in logs
    let tag = if exe.eq_ignore_ascii_case("cmd") { "PS" } else { "PROC" };

    // Non-blocking design: two reader threads feed a channel; main loop ticks UI every 200 ms.
    let (tx, rx) = mpsc::channel::<String>();

    // stdout reader
    {
        let tx = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => { let _ = tx.send(l); }
                    Err(_) => break,
                }
            }
        });
    }

    // stderr reader
    {
        let tx = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(l) => { let _ = tx.send(l); }
                    Err(_) => break,
                }
            }
        });
    }
    drop(tx); // main thread keeps only rx

    // Helper: processes one cleaned line (update progress + durable log)
    fn handle_line(pstate: &mut ProgressState, cleaned: &str, tag: &str) {
        if cleaned.is_empty() { return; }

        let ratio = parse_ratio_percent(cleaned);
        let pct = parse_percent(cleaned).or(ratio);

        if let Some(p) = pct {
            pstate.best_builder_pct = Some(pstate.best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
        }
        if ratio.is_some() {
            pstate.runtime_phase = "build".into();
        }

        let snip = last_nonempty_snippet(cleaned, 120);
        if !snip.is_empty() { pstate.last_snippet = snip; }

        let _ = end_ephemeral();
        match classify_line(cleaned) {
            Sev::Err  => out_err (tag, cleaned),
            Sev::Warn => out_warn(tag, cleaned),
            Sev::Info => out_info(tag, cleaned),
        }
    }

    let mut readers_done = false;
    let mut exit_code: Option<i32> = None;

    // Initial ephemeral or start line
    if progress_enabled() {
        render_and_print(&mut pstate, predicted_ms);
    } else {
        let _ = end_ephemeral();
        out_info("RUST", &format!("RUN phase={} started", pstate.runtime_phase));
    }

    loop {
        // Drain currently available lines
        let mut drained_any = false;
        loop {
            match rx.try_recv() {
                Ok(raw) => {
                    let cleaned = sanitize_line(&raw);
                    if !cleaned.is_empty() {
                        handle_line(&mut pstate, &cleaned, tag);
                    }
                    drained_any = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    readers_done = true;
                    break;
                }
            }
        }

        // Keep animation alive
        if progress_enabled() && due(&pstate) {
            render_and_print(&mut pstate, predicted_ms);
        }

        // Poll child exit
        match child.try_wait() {
            Ok(Some(st)) => { exit_code = Some(st.code().unwrap_or(1)); }
            Ok(None) => {}
            Err(e) => {
                let _ = end_ephemeral();
                out_err("RUST", &format!("wait failed: {}", e));
                return RunResult { code: 1 };
            }
        }

        // Finish condition: child exited AND all readers done
        if let Some(code) = exit_code {
            if readers_done {
                let _ = end_ephemeral();

                // Dauer erfassen & persistieren
                let elapsed_ms = pstate.start.elapsed().as_millis() as u128;
                metrics.upsert_phase_ms(&phase_sig.sig, &pstate.runtime_phase, elapsed_ms);
                let _ = metrics.save(&workdir);

                // Hübsches Ende: Trailer ODER (falls deaktiviert) die alte „RUN done“-Zeile
                if trailer_enabled() {
                    let secs = (elapsed_ms as f32) / 1000.0;
                    let ok = code == 0;
                    out_trailer_min(ok, code, secs, None);
                } else {
                    out_info("RUST", &format!(
                        "RUN phase={} done (elapsed={}s)",
                        pstate.runtime_phase,
                        pstate.start.elapsed().as_secs()
                    ));
                }

                return RunResult { code };
            }
        }

        if !drained_any {
            thread::sleep(Duration::from_millis(20));
        }
    }
}

// Legacy name kept for back-compat (if externally used)
#[allow(dead_code)]
pub fn run_streamed(exe: &str, args: &[String]) -> RunResult {
    run_streamed_with_env(exe, args, None, None)
}
