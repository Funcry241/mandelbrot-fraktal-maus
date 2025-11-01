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

/// Aggregiert Artefakt- & Git-Infos aus Kindprozess-Logs für den hübschen Trailer.
#[derive(Default)]
struct TrailerAgg {
    artifact: Option<String>,
    git_remote: Option<String>,
    git_branch: Option<String>,
    git_commit_short: Option<String>,
    git_pushed_ok: bool,
    git_rules_bypassed: bool,
}
impl TrailerAgg {
    fn feed(&mut self, line: &str) -> bool /* suppress printing? */ {
        let l = line.trim();

        // Artefakte
        if let Some(idx) = l.find("[RUNNER] artifact:") {
            // Format: "[RUNNER] artifact: C:\...\mandelbrot_otterdream.exe"
            if let Some(path) = l.get(idx + 19..).map(|s| s.trim()) {
                if !path.is_empty() { self.artifact = Some(path.to_string()); }
            }
            return true; // leise sammeln, nicht doppelt ausgeben
        }
        if l.contains("[RUNNER] artifact-candidate:") {
            return true; // Rauschen unterdrücken
        }

        // AUTOGIT Start/Kommandos-Rauschen
        if l.starts_with("[AUTOGIT] start")
            || l.starts_with("[AUTOGIT][RUN] git")
            || l.starts_with("Enumerating objects:")
            || l.starts_with("Counting objects:")
            || l.starts_with("Delta compression")
            || l.starts_with("Compressing objects:")
            || l.starts_with("Writing objects:")
            || l.starts_with("Total ")
            || l.starts_with("remote: Resolving deltas:")
        {
            return true;
        }

        // Bypassed/Protected-Branch-Hinweise merken, aber nicht spammen
        if l.starts_with("remote: Bypassed rule violations")
            || l.contains("Cannot update this protected ref")
        {
            self.git_rules_bypassed = true;
            return true;
        }

        // Push-Ziel / Remote (z. B. "To https://...  main -> main")
        if l.starts_with("To ") {
            self.git_pushed_ok = true;
            // branch heuristisch ziehen
            if let Some(pos) = l.rfind("->") {
                let tail = &l[pos+2..].trim();
                if !tail.is_empty() { self.git_branch = Some(tail.to_string()); }
            }
            // remote
            if let Some(space) = l.find(' ') {
                self.git_remote = Some(l[3..space].trim().to_string());
            }
            return true;
        }

        // Commit-Zeile: "[main ba51fed] chore: update"
        if l.starts_with("[main ") && l.contains(']') {
            // Branch + short SHA
            if let Some(end) = l.find(']') {
                let body = &l[1..end]; // main ba51fed
                let mut it = body.split_whitespace();
                self.git_branch = it.next().map(|s| s.to_string());
                self.git_commit_short = it.next().map(|s| s.to_string());
            }
            return false; // darf sichtbar bleiben, ist oft nützlich
        }

        // Abschluss von AUTOGIT
        if l.starts_with("[AUTOGIT] done status=OK") {
            self.git_pushed_ok = true;
            return true;
        }

        // branch 'main' set up to track 'origin/main'.
        if l.starts_with("branch '") && l.contains(" set up to track ") {
            // branch extrahieren
            let name = l.trim_start_matches("branch '")
                .split('\'').next().unwrap_or("").trim();
            if !name.is_empty() { self.git_branch = Some(name.to_string()); }
            return true;
        }

        false
    }

    fn build_extra(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();

        if let Some(p) = &self.artifact {
            let base = Path::new(p).file_name()
                .and_then(|o| o.to_str()).unwrap_or(p);
            parts.push(format!("artifact={}", base));
        }

        if self.git_pushed_ok {
            let mut s = String::from("git: pushed ✓");
            if let Some(b) = &self.git_branch {
                s.push(' ');
                s.push_str(b);
            }
            if let Some(c) = &self.git_commit_short {
                s.push_str(" @");
                s.push_str(c);
            }
            if self.git_rules_bypassed {
                s.push_str(" (rules)");
            }
            parts.push(s);
        }

        if parts.is_empty() { None } else { Some(parts.join(" • ")) }
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

    // Trailer-Aggregator
    let mut trailer = TrailerAgg::default();

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
    fn handle_line(pstate: &mut ProgressState, cleaned: &str, tag: &str, trailer: &mut TrailerAgg) {
        if cleaned.is_empty() { return; }

        // Trailers sammeln / Rauschen ggf. unterdrücken
        if trailer.feed(cleaned) {
            return; // nichts ausgeben
        }

        // Progress aus den Inhalten schätzen
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
                        handle_line(&mut pstate, &cleaned, tag, &mut trailer);
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

                // Hübsches Ende: farbiger Trailer + kompakte Extras
                if trailer_enabled() {
                    let secs = (elapsed_ms as f32) / 1000.0;
                    let ok = code == 0;
                    let extra = trailer.build_extra();
                    out_trailer_min(ok, code, secs, extra.as_deref());
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
