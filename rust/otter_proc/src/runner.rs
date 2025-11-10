///// Otter: Runner – robustes Streaming, Trailer, **Fail-Report mit Log-Tail (300 Zeilen)**.
///// Schneefuchs: dedupliziert Rauschen, Trailer-Aggregat, ANSI sicher; /WX-safe.
/// // Maus: Heartbeat wenn kein Output; CMake/Ninja-Signature für Metriken; Diagnose-Hook bei Exit≠0.
/// ///// Datei: rust/otter_proc/src/runner.rs

use std::collections::{HashMap, VecDeque};
use std::env;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use crate::build_metrics::BuildMetrics;

pub mod runner_term;
mod runner_progress;
mod runner_classify;
mod runner_diagnose; // Diagnose-Report bei Exit≠0

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

struct PhaseDetect { phase: String, sig: String }

fn detect_phase_and_sig(exe: &str, args: &[String]) -> PhaseDetect {
    let mut phase = "proc".to_string();
    let mut is_build = false;
    for a in args {
        if a.eq_ignore_ascii_case("build") || a.contains("cmake --build") {
            is_build = true; break;
        }
    }
    if exe.eq_ignore_ascii_case("cmake") && !is_build { phase = "configure".into(); }
    if is_build { phase = "build".into(); }

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

fn trailer_enabled() -> bool {
    match env::var("OTTER_TRAILER") {
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !(t == "0" || t == "off" || t == "no")
        }
        Err(_) => true,
    }
}

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
    fn feed(&mut self, line: &str) -> bool {
        let l = line.trim();

        if let Some(idx) = l.find("[RUNNER] artifact:") {
            if let Some(path) = l.get(idx + 19..).map(|s| s.trim()) {
                if !path.is_empty() { self.artifact = Some(path.to_string()); }
            }
            return true;
        }
        if l.contains("[RUNNER] artifact-candidate:") { return true; }

        if l.starts_with("[AUTOGIT] start")
            || l.starts_with("[AUTOGIT][RUN] git")
            || l.starts_with("Enumerating objects:")
            || l.starts_with("Counting objects:")
            || l.starts_with("Delta compression")
            || l.starts_with("Compressing objects:")
            || l.starts_with("Writing objects:")
            || l.starts_with("Total ")
            || l.starts_with("remote: Resolving deltas:")
        { return true; }

        if l.starts_with("remote: Bypassed rule violations")
            || l.contains("Cannot update this protected ref")
        { self.git_rules_bypassed = true; return true; }

        if l.starts_with("To ") {
            self.git_pushed_ok = true;
            if let Some(pos) = l.rfind("->") {
                let tail = &l[pos+2..].trim();
                if !tail.is_empty() { self.git_branch = Some(tail.to_string()); }
            }
            if let Some(space) = l.find(' ') {
                self.git_remote = Some(l[3..space].trim().to_string());
            }
            return true;
        }

        if l.starts_with("[main ") && l.contains(']') {
            if let Some(end) = l.find(']') {
                let body = &l[1..end];
                let mut it = body.split_whitespace();
                self.git_branch = it.next().map(|s| s.to_string());
                self.git_commit_short = it.next().map(|s| s.to_string());
            }
            return false;
        }

        if l.starts_with("[AUTOGIT] done status=OK") { self.git_pushed_ok = true; return true; }
        if l.starts_with("branch '") && l.contains(" set up to track ") {
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
            let base = Path::new(p).file_name().and_then(|o| o.to_str()).unwrap_or(p);
            parts.push(format!("artifact={}", base));
        }

        if self.git_pushed_ok {
            let mut s = String::from("git: pushed ✓");
            if let Some(b) = &self.git_branch    { s.push(' '); s.push_str(b); }
            if let Some(c) = &self.git_commit_short { s.push_str(" @"); s.push_str(c); }
            if self.git_rules_bypassed { s.push_str(" (rules)"); }
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
    print_ephemeral("[proc] starting...");

    let (mut metrics, metrics_file, seed_src) = BuildMetrics::load_or_seed(&workdir);
    if !METRICS_PRINTED_ONCE.swap(true, Ordering::SeqCst) {
        out_info("RUST", &format!("metrics={}", metrics_file.display()));
        if let Some(src) = seed_src { out_info("RUST", &format!("metrics-seeded-from={}", src.display())); }
    }

    let mut cmd = Command::new(exe);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Some(d) = cwd { cmd.current_dir(d); }
    if let Some(envmap) = env_overlay { for (k,v) in envmap.iter() { cmd.env(k, v); } }

    let phase_sig = detect_phase_and_sig(exe, args);
    out_info("RUST", &format!("RUN exe=\"{}\" phase={} sig={}", exe, phase_sig.phase, phase_sig.sig));

    let t_spawn0 = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => { end_ephemeral().ok(); out_err("RUST", &format!("spawn failed exe={} err={}", exe, e)); return RunResult { code: 1 }; }
    };
    let spawn_ms = t_spawn0.elapsed().as_millis();
    if spawn_ms > 400 { end_ephemeral().ok(); out_info("RUST", &format!("spawn-latency={}ms", spawn_ms)); }

    let stdout = match child.stdout.take() { Some(s) => s, None => { out_err("RUST", "failed to take stdout"); return RunResult { code: 1 }; } };
    let stderr = match child.stderr.take() { Some(s) => s, None => { out_err("RUST", "failed to take stderr"); return RunResult { code: 1 }; } };

    let predicted_ms = metrics.get_last_ms(&phase_sig.sig, &phase_sig.phase).unwrap_or(0);

    let mut pstate = ProgressState::new(&phase_sig.phase);
    let mut trailer = TrailerAgg::default();

    // --- Tail-Puffer für Diagnose ------------------------------------------------
    const TAIL_KEEP: usize = 300;
    let mut tail: VecDeque<String> = VecDeque::with_capacity(TAIL_KEEP);
    // ----------------------------------------------------------------------------

    let tag = if exe.eq_ignore_ascii_case("cmd") { "PS" } else { "PROC" };
    let (tx, rx) = mpsc::channel::<String>();

    {
        let tx = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(l) = line { let _ = tx.send(l); } else { break; }
            }
        });
    }
    {
        let tx = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(l) = line { let _ = tx.send(l); } else { break; }
            }
        });
    }
    drop(tx);

    let heartbeat_enabled = !progress_enabled();
    let spinner: [char; 4] = ['-', '\\', '|', '/'];
    let mut hb_idx: usize = 0;
    let mut hb_last = Instant::now();
    let mut saw_any_child_output = false;

    let mut readers_done = false;
    let mut exit_code: Option<i32> = None;

    if progress_enabled() {
        render_and_print(&mut pstate, predicted_ms);
    } else {
        end_ephemeral().ok();
        out_info("RUST", &format!("RUN phase={} started", pstate.runtime_phase));
    }

    // Hilfsfunktion statt Closure – vermeidet Borrow-Konflikte
    fn handle_line(
        pstate: &mut ProgressState,
        trailer: &mut TrailerAgg,
        tail: &mut VecDeque<String>,
        cleaned: &str,
        tag: &str,
    ) {
        if cleaned.is_empty() { return; }

        if trailer.feed(cleaned) { return; }

        if let Some(r) = parse_ratio_percent(cleaned) {
            pstate.runtime_phase = "build".into();
            pstate.best_builder_pct = Some(pstate.best_builder_pct.map(|b| b.max(r)).unwrap_or(r));
        } else if let Some(p) = parse_percent(cleaned) {
            pstate.best_builder_pct = Some(pstate.best_builder_pct.map(|b| b.max(p)).unwrap_or(p));
        }

        let snip = last_nonempty_snippet(cleaned, 120);
        if !snip.is_empty() { pstate.last_snippet = snip; }

        // Tail auffüllen
        tail.push_back(cleaned.to_string());
        if tail.len() > TAIL_KEEP { tail.pop_front(); }

        let _ = end_ephemeral();
        match classify_line(cleaned) {
            Sev::Err  => out_err (tag, cleaned),
            Sev::Warn => out_warn(tag, cleaned),
            Sev::Info => out_info(tag, cleaned),
        }
    }

    loop {
        let mut drained_any = false;
        loop {
            match rx.try_recv() {
                Ok(raw) => {
                    let cleaned = sanitize_line(&raw);
                    if !cleaned.is_empty() {
                        saw_any_child_output = true;
                        handle_line(&mut pstate, &mut trailer, &mut tail, &cleaned, tag);
                    }
                    drained_any = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => { readers_done = true; break; }
            }
        }

        if progress_enabled() && due(&pstate) {
            render_and_print(&mut pstate, predicted_ms);
        }
        if heartbeat_enabled && exit_code.is_none() && !saw_any_child_output && hb_last.elapsed() >= Duration::from_millis(120) {
            hb_idx = (hb_idx + 1) & 3;
            let secs = pstate.start.elapsed().as_secs_f32();
            let spin = spinner[hb_idx];
            print_ephemeral(&format!("[{tag}] waiting for output… {spin} t={secs:.1}s"));
            hb_last = Instant::now();
        }

        match child.try_wait() {
            Ok(Some(st)) => { exit_code = Some(st.code().unwrap_or(1)); }
            Ok(None) => {}
            Err(e) => { end_ephemeral().ok(); out_err("RUST", &format!("wait failed: {}", e)); return RunResult { code: 1 }; }
        }

        if let Some(code) = exit_code {
            if readers_done {
                end_ephemeral().ok();

                // Diagnose-Hook: Fail-Report vor Metriken/Trailer schreiben
                if code != 0 {
                    let snapshot: Vec<String> = tail.iter().cloned().collect();
                    let _ = runner_diagnose::try_write_on_failure(
                        &workdir,
                        &pstate.runtime_phase,
                        &phase_sig.sig,
                        code,
                        Some(&snapshot),
                    );
                }

                // Dauer persistieren
                let elapsed_ms = pstate.start.elapsed().as_millis() as u128;
                metrics.upsert_phase_ms(&phase_sig.sig, &pstate.runtime_phase, elapsed_ms);
                let _ = metrics.save(&workdir);

                // Trailer
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

#[allow(dead_code)]
pub fn run_streamed(exe: &str, args: &[String]) -> RunResult {
    run_streamed_with_env(exe, args, None, None)
}
