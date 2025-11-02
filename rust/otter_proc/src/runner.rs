///// Otter: Process runner with live progress (spinner, ETA, %, ratio merge); pretty trailer & dist bundling.
/// //// Schneefuchs: No external crates; trims noise; smooth 200 ms animation; safe metrics persistence.
/// //// Maus: Colors for tags ([RUST]/[PS]/[PROC]); ASCII bar; parses “68%” & “[17/45]”; Windows+POSIX.
/// //// Datei: rust/otter_proc/src/runner.rs

use std::collections::HashMap;
use std::env;
use std::fs;
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
/// Nur wenn OTTER_TRAILER=0|off|no gesetzt ist, wird er unterdrückt.
fn trailer_enabled() -> bool {
    match env::var("OTTER_TRAILER") {
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !(t == "0" || t == "off" || t == "no")
        }
        Err(_) => true, // Default: an
    }
}

/// Triplet aus CMake-Cache lesen (falls vorhanden)
fn detect_vcpkg_triplet(root: &Path) -> Option<String> {
    let cache = root.join("build").join("CMakeCache.txt");
    let Ok(text) = fs::read_to_string(&cache) else { return None; };
    for line in text.lines() {
        if line.contains("VCPKG_TARGET_TRIPLET") {
            if let Some(eq) = line.find('=') {
                let trip = line[eq+1..].trim();
                if !trip.is_empty() { return Some(trip.to_string()); }
            }
        }
    }
    None
}

fn name_eq_ci(name: &str, pat: &str) -> bool { name.eq_ignore_ascii_case(pat) }
fn name_has_ci(name: &str, needle: &str) -> bool {
    name.to_ascii_lowercase().contains(&needle.to_ascii_lowercase())
}

#[cfg(windows)]
fn gather_vcpkg_dlls(root: &Path, triplet_opt: Option<String>) -> Vec<PathBuf> {
    let triplet = triplet_opt.or_else(|| env::var("OTTER_VCPKG_TRIPLET").ok())
        .unwrap_or_else(|| "x64-windows".to_string());
    let base = root.join("vcpkg").join("installed").join(&triplet);
    let candidates = [ base.join("bin"), base.join("debug").join("bin") ];

    let mut out = Vec::new();
    for dir in candidates.iter() {
        let Ok(rd) = fs::read_dir(dir) else { continue; };
        for e in rd {
            if let Ok(ent) = e {
                let p = ent.path();
                if p.extension().and_then(|s| s.to_str())
                    .map(|s| s.eq_ignore_ascii_case("dll")).unwrap_or(false)
                {
                    let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    // gezielt minimal halten: glew/glfw
                    if name_eq_ci(fname, "glew32.dll") || name_has_ci(fname, "glfw") {
                        out.push(p);
                    }
                }
            }
        }
    }
    out
}
#[cfg(not(windows))]
fn gather_vcpkg_dlls(_root: &Path, _triplet_opt: Option<String>) -> Vec<PathBuf> { Vec::new() }

/// Finde das Artefakt (EXE) auch ohne Log-Sniffing.
/// Kandidaten: build/{cfg}/mandelbrot_otterdream.exe, build/bin/{cfg}/..., build/bin/..., build/...
fn find_artifact_exe(root: &Path) -> Option<PathBuf> {
    let build = root.join("build");
    let cfgs = ["RelWithDebInfo", "Release", "Debug", "MinSizeRel"];

    // harte Kandidaten
    for cfg in &cfgs {
        let c1 = build.join(cfg).join("mandelbrot_otterdream.exe");
        if c1.exists() { return Some(c1); }
        let c2 = build.join("bin").join(cfg).join("mandelbrot_otterdream.exe");
        if c2.exists() { return Some(c2); }
    }
    let c3 = build.join("bin").join("mandelbrot_otterdream.exe");
    if c3.exists() { return Some(c3); }
    let c4 = build.join("mandelbrot_otterdream.exe");
    if c4.exists() { return Some(c4); }

    // weiche Suche (flach, keine teure Rekursion)
    if let Ok(rd) = fs::read_dir(&build) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_file() {
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    if name_has_ci(name, "mandelbrot") && name_has_ci(name, ".exe") { return Some(p); }
                    if name_has_ci(name, "otterdream") && name_has_ci(name, ".exe") { return Some(p); }
                }
            }
        }
    }
    None
}

struct DistResult {
    exe_name: String,
    copied_dlls: Vec<String>,
}

/// Kopiert EXE → dist und sammelt passende DLLs (Windows)
fn copy_to_dist(artifact: &Path, root: &Path) -> std::io::Result<DistResult> {
    let dist = root.join("dist");
    fs::create_dir_all(&dist)?;

    // EXE kopieren
    let exe_name = artifact.file_name().and_then(|s| s.to_str()).unwrap_or("app.exe").to_string();
    let dest_exe = dist.join(&exe_name);
    fs::copy(artifact, &dest_exe)?;

    // DLLs sammeln (Windows) und kopieren
    let mut copied: Vec<String> = Vec::new();
    #[cfg(windows)]
    {
        let triplet = detect_vcpkg_triplet(root);
        let dlls = gather_vcpkg_dlls(root, triplet);
        for dll in dlls {
            if let Some(fname) = dll.file_name().and_then(|s| s.to_str()) {
                let _ = fs::copy(&dll, dist.join(fname))?;
                copied.push(fname.to_string());
            }
        }
    }

    Ok(DistResult { exe_name, copied_dlls: copied })
}

/// Aggregiert Git-Infos aus Kindprozess-Logs für den Trailer (Artefakt wird unabhängig gesucht).
#[derive(Default)]
struct TrailerAgg {
    git_remote: Option<String>,
    git_branch: Option<String>,
    git_commit_short: Option<String>,
    git_pushed_ok: bool,
    git_rules_bypassed: bool,
}
impl TrailerAgg {
    fn feed(&mut self, line: &str) -> bool /* suppress printing? */ {
        let l = line.trim();

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
            if let Some(pos) = l.rfind("->") {
                let tail = &l[pos+2..].trim();
                if !tail.is_empty() { self.git_branch = Some(tail.to_string()); }
            }
            if let Some(space) = l.find(' ') {
                self.git_remote = Some(l[3..space].trim().to_string());
            }
            return true;
        }

        // Commit-Zeile: "[main fe52e68] chore: update"
        if l.starts_with("[main ") && l.contains(']') {
            if let Some(end) = l.find(']') {
                let body = &l[1..end]; // main fe52e68
                let mut it = body.split_whitespace();
                self.git_branch = it.next().map(|s| s.to_string());
                self.git_commit_short = it.next().map(|s| s.to_string());
            }
            return false; // nützlich, darf sichtbar bleiben
        }

        if l.starts_with("[AUTOGIT] done status=OK") {
            self.git_pushed_ok = true;
            return true;
        }

        if l.starts_with("branch '") && l.contains(" set up to track ") {
            let name = l.trim_start_matches("branch '")
                .split('\'').next().unwrap_or("").trim();
            if !name.is_empty() { self.git_branch = Some(name.to_string()); }
            return true;
        }

        false
    }

    fn build_extra(&self, dist_part: Option<&str>) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();

        if let Some(dp) = dist_part {
            if !dp.is_empty() { parts.push(dp.to_string()); }
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
        Err(e) => { let _ = end_ephemeral(); out_err("RUST", &format!("spawn failed exe={} err={}", exe, e)); return RunResult { code: 1 }; }
    };
    let spawn_ms = t_spawn0.elapsed().as_millis();
    if spawn_ms > 400 {
        let _ = end_ephemeral();
        out_info("RUST", &format!("spawn-latency={}ms", spawn_ms));
    }

    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => { let _ = end_ephemeral(); out_err("RUST", "failed to take stdout"); return RunResult { code: 1 }; }
    };
    let stderr = match child.stderr.take() {
        Some(s) => s,
        None => { let _ = end_ephemeral(); out_err("RUST", "failed to take stderr"); return RunResult { code: 1 }; }
    };

    let predicted_ms = metrics.get_last_ms(&phase_sig.sig, &phase_sig.phase).unwrap_or(0);

    // Progress
    let mut pstate = ProgressState::new(&phase_sig.phase);
    // Trailer-Aggregator (nur Git/Meta)
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

        // Git/Meta sammeln / Rauschen ggf. unterdrücken
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

                // Dist bundling (unabhängig vom Log: EXE direkt suchen)
                let mut dist_part: Option<String> = None;
                if code == 0 {
                    if let Some(art) = find_artifact_exe(&workdir) {
                        match copy_to_dist(&art, &workdir) {
                            Ok(dr) => { dist_part = Some(format!("dist={} (+{} DLLs)", dr.exe_name, dr.copied_dlls.len())); }
                            Err(e) => { dist_part = Some(format!("dist=ERR({})", e)); }
                        }
                    }
                }

                // Hübsches Ende: farbiger Trailer + kompakte Extras
                if trailer_enabled() {
                    let secs = (elapsed_ms as f32) / 1000.0;
                    let ok = code == 0;
                    let extra = trailer.build_extra(dist_part.as_deref());
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
