///// Otter: Zentraler Metrics-Helper — Profil/Plan/Events + History/Baseline; nicht-fatal, ASCII-only.
/// ///// Schneefuchs: Kein serde; einfache JSON-Strings; robuste Datei-I/O; UTF-8 lossy; umkehr via Vec<String>. 
/// ///// Maus: Legt Verzeichnisse lazy an; Dateien: build_profile.json, progress_plan.json, progress.jsonl, *_history.txt, *_baseline_*.txt.
/// ///// Datei: rust/otter_proc/src/metrics.rs
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const PROFILE_FILE: &str = "build_profile.json";
const PLAN_FILE: &str = "progress_plan.json";
const EVENTS_FILE: &str = "progress.jsonl";

// -------- public structs ------------------------------------------------------

#[derive(Default, Debug, Clone, Copy)]
pub struct PhaseDurations {
    pub probe: u64,
    pub configure: u64,
    pub build: u64,
    pub stage: u64,
}

impl PhaseDurations {
    pub fn set(&mut self, name: &str, ms: u64) {
        match name {
            "probe" => self.probe = ms,
            "configure" => self.configure = ms,
            "build" => self.build = ms,
            "stage" => self.stage = ms,
            _ => {}
        }
    }
    pub fn total(&self) -> u64 { self.probe + self.configure + self.build + self.stage }
}

// -------- public utils --------------------------------------------------------

pub fn ensure_dir(dir: &Path) -> io::Result<()> {
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }
    Ok(())
}

// -------- profile & plan ------------------------------------------------------

pub fn read_last_profile(metrics_dir: &Path) -> io::Result<Option<PhaseDurations>> {
    let p = metrics_dir.join(PROFILE_FILE);
    if !p.exists() {
        return Ok(None);
    }
    let mut s = String::new();
    File::open(&p)?.read_to_string(&mut s)?;
    let mut out = PhaseDurations::default();
    out.probe     = find_json_u64(&s, r#""probe":"#).unwrap_or(0);
    out.configure = find_json_u64(&s, r#""configure":"#).unwrap_or(0);
    out.build     = find_json_u64(&s, r#""build":"#).unwrap_or(0);
    out.stage     = find_json_u64(&s, r#""stage":"#).unwrap_or(0);
    if out.total() == 0 { Ok(None) } else { Ok(Some(out)) }
}

pub fn write_build_profile(metrics_dir: &Path, p: &PhaseDurations) -> io::Result<()> {
    ensure_dir(metrics_dir)?;
    let path = metrics_dir.join(PROFILE_FILE);
    let txt = format!(
        concat!(
            r#"{{"version":1,"recorded_at":{},"phases_ms":{{"#,
            r#""probe":{},"#,
            r#""configure":{},"#,
            r#""build":{},"#,
            r#""stage":{}"#,
            r#"}}}}"#
        ),
        now_ms(), p.probe, p.configure, p.build, p.stage
    );
    write_text_file(&path, &txt)
}

pub fn write_progress_plan_measured(metrics_dir: &Path, p: &PhaseDurations) -> io::Result<()> {
    ensure_dir(metrics_dir)?;
    let path = metrics_dir.join(PLAN_FILE);
    let total = p.total();
    let txt = format!(
        concat!(
            r#"{{"version":1,"mode":"measured","predicted_total_ms":{},"phases":["#,
            r#"{{"name":"probe","ms":{}}},"#,
            r#"{{"name":"configure","ms":{}}},"#,
            r#"{{"name":"build","ms":{}}},"#,
            r#"{{"name":"stage","ms":{}}}"#,
            r#"]}}"#
        ),
        total, p.probe, p.configure, p.build, p.stage
    );
    write_text_file(&path, &txt)
}

pub fn write_progress_plan_indeterminate(metrics_dir: &Path, passes: &[String]) -> io::Result<()> {
    ensure_dir(metrics_dir)?;
    let path = metrics_dir.join(PLAN_FILE);
    let items = passes.iter().map(|s| format!(r#""{}""#, s)).collect::<Vec<_>>().join(",");
    let txt = format!(r#"{{"version":1,"mode":"indeterminate","phases":[{}]}}"#, items);
    write_text_file(&path, &txt)
}

// -------- progress events (jsonl) --------------------------------------------

pub fn init_events_file(metrics_dir: &Path) -> io::Result<()> {
    ensure_dir(metrics_dir)?;
    let path = metrics_dir.join(EVENTS_FILE);
    if !path.exists() {
        File::create(path)?;
    }
    Ok(())
}

pub fn write_phase_event(metrics_dir: &Path, name: &str, state: &str) -> io::Result<()> {
    let path = metrics_dir.join(EVENTS_FILE);
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let line = format!(r#"{{"ts":{},"kind":"phase","name":"{}","state":"{}"}}"#, now_ms(), name, state);
    writeln!(f, "{}", line)?;
    Ok(())
}

pub fn write_phase_event_with_elapsed(metrics_dir: &Path, name: &str, state: &str, elapsed_ms: u64) -> io::Result<()> {
    let path = metrics_dir.join(EVENTS_FILE);
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let line = format!(r#"{{"ts":{},"kind":"phase","name":"{}","state":"{}","elapsed_ms":{}}}"#, now_ms(), name, state, elapsed_ms);
    writeln!(f, "{}", line)?;
    Ok(())
}

// -------- history/baseline (aus deinem Snippet) ------------------------------

fn history_file(dir: &Path, generator: &str) -> PathBuf {
    dir.join(format!("{}_history.txt", generator))
}

fn baseline_file(dir: &Path, config: &str, generator: &str) -> PathBuf {
    dir.join(format!("{}_baseline_{}.txt", generator, config))
}

/// Append a raw line to the generator’s history file (creates if missing).
pub fn append_history_line(dir: &Path, generator: &str, line: &str) -> io::Result<()> {
    ensure_dir(dir)?;
    let mut f = OpenOptions::new().create(true).append(true).open(history_file(dir, generator))?;
    writeln!(f, "{}", line)?;
    Ok(())
}

/// Return up to `limit` last lines from the generator’s history (newest-first).
pub fn tail_history(dir: &Path, generator: &str, limit: usize) -> io::Result<Vec<String>> {
    ensure_dir(dir)?;
    let p = history_file(dir, generator);
    if !p.exists() { return Ok(Vec::new()); }
    let rdr = BufReader::new(File::open(p)?);

    // Kein DoubleEndedIterator über std::io::Lines — erst sammeln, dann rückwärts iterieren.
    let lines: Vec<String> = rdr.lines().filter_map(|r| r.ok()).collect();
    let mut out = Vec::new();
    for s in lines.iter().rev().take(limit) {
        out.push(s.to_string());
    }
    Ok(out)
}

/// Rebuild a tiny “baseline” file from the most recent history lines.
/// Strategy: pick the newest non-empty line and store it as the baseline snapshot.
pub fn rebuild_baseline(dir: &Path, config: &str, generator: &str, history_limit: usize) -> io::Result<()> {
    ensure_dir(dir)?;
    let lines = tail_history(dir, generator, history_limit)?;
    let snapshot = lines.into_iter().find(|s| !s.trim().is_empty()).unwrap_or_else(|| "# empty".to_string());

    let mut f = File::create(baseline_file(dir, config, generator))?;
    writeln!(f, "{}", snapshot)?;
    Ok(())
}

// -------- internals ----------------------------------------------------------

fn write_text_file(path: &Path, txt: &str) -> io::Result<()> {
    if let Some(p) = path.parent() { ensure_dir(p)?; }
    let mut f = File::create(path)?;
    f.write_all(txt.as_bytes())
}

fn find_json_u64(hay: &str, needle: &str) -> Option<u64> {
    if let Some(i) = hay.find(needle) {
        let j = i + needle.len();
        let tail = &hay[j..];
        let mut num = String::new();
        for ch in tail.chars() {
            if ch.is_ascii_digit() { num.push(ch); } else { break; }
        }
        if !num.is_empty() { return num.parse::<u64>().ok(); }
    }
    None
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
}
