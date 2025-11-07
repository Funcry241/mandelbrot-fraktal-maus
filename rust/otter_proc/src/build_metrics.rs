///// Otter: BuildMetrics – Phasen-Metriken (HashMap) + runs.jsonl; atomische Writes; panic-sicheres Seeding.
///// Schneefuchs: ASCII-JSON-Writer, keine Fremd-Deps; deterministische Pfade; Windows-safe.
///// Maus: Klare öffentliche API (load_or_seed, save, snapshot, write_extended_run).
///// Datei: rust/otter_proc/src/build_metrics.rs

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Default, Clone)]
pub struct BuildMetrics {
    phases_ms: HashMap<String, u128>,
}

impl BuildMetrics {
    #[inline]
    fn key(sig: &str, phase: &str) -> String {
        let mut s = String::with_capacity(sig.len() + phase.len() + 1);
        s.push_str(sig);
        s.push('|');
        s.push_str(phase);
        s
    }

    /// Returns (metrics, path_to_metrics_json, seeded_from_opt)
    pub fn load_or_seed(workdir: &Path) -> (BuildMetrics, PathBuf, Option<PathBuf>) {
        let dir = metrics_dir(workdir);
        let file = metrics_file_path(workdir);

        // Hygiene: alte .build_metrics.tmp* im Root entfernen (Bug-Altlast)
        cleanup_root_tmp_files(workdir);

        let _ = fs::create_dir_all(&dir);

        if file.exists() {
            let m = load_json_map(&file).unwrap_or_default();
            return (BuildMetrics { phases_ms: m }, file, None);
        }

        if let Some(seed_json) = find_latest_sibling_metrics_json(workdir) {
            let m = load_json_map(&seed_json).unwrap_or_default();
            let _ = save_json_map_atomically(&file, &m);
            return (BuildMetrics { phases_ms: m }, file, Some(seed_json));
        }

        let m = BuildMetrics::default();
        let _ = save_json_map_atomically(&file, &m.phases_ms);
        (m, file, None)
    }

    pub fn get_last_ms(&self, sig: &str, phase: &str) -> Option<u128> {
        self.phases_ms.get(&Self::key(sig, phase)).copied()
    }

    pub fn upsert_phase_ms(&mut self, sig: &str, phase: &str, ms: u128) {
        self.phases_ms.insert(Self::key(sig, phase), ms);
    }

    /// Atomisch nur die Phasen-Map (kompakt) nach metrics.json schreiben.
    pub fn save(&self, workdir: &Path) -> std::io::Result<()> {
        let file = metrics_file_path(workdir);
        if let Some(parent) = file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        save_json_map_atomically(&file, &self.phases_ms)
    }

    /// Schnappschuss der Phasen-Map (Kopie für Summary/History).
    pub fn snapshot(&self) -> HashMap<String, u128> {
        self.phases_ms.clone()
    }
}

// ---------- History-API (runs.jsonl) ----------

pub struct RunSummary<'a> {
    pub op: &'a str,                // "branch" | "build"
    pub channel: &'a str,           // "dev" | "stable"
    pub ts_ms: u128,                // Start
    pub elapsed_ms: u128,           // Dauer
    pub success: bool,
    pub exit_code: i32,
    pub root: &'a Path,             // Projekt-Root
    pub branch: &'a str,            // Branch-Name
    pub commit: Option<&'a str>,    // short sha
    pub cfg: &'a str,               // z.B. "RelWithDebInfo"
    pub preset_cfg: &'a str,        // z.B. "windows-msvc"
    pub preset_build: &'a str,      // z.B. "windows-build"
    pub artifact: Option<&'a Path>, // finaler Artefaktpfad (falls vorhanden)
}

/// Hängt einen JSON-Eintrag an .build_metrics/runs.jsonl an.
/// metrics.json bleibt ein reines Phasen-Dokument (kompakt, atomisch).
pub fn write_extended_run(
    workdir: &Path,
    run: RunSummary<'_>,
    phases: &HashMap<String, u128>,
    counters_opt: Option<&HashMap<String, u128>>,
) -> std::io::Result<()> {
    let dir = metrics_dir(workdir);
    let _ = fs::create_dir_all(&dir);

    let mut s = String::with_capacity(512);
    s.push('{');

    kv_str(&mut s, "op", run.op, true);
    kv_str(&mut s, "channel", run.channel, false);
    kv_u128(&mut s, "ts_ms", run.ts_ms, false);
    kv_u128(&mut s, "elapsed_ms", run.elapsed_ms, false);
    kv_bool(&mut s, "success", run.success, false);
    kv_i32(&mut s, "exit_code", run.exit_code, false);
    kv_str(&mut s, "root", run.root.to_string_lossy().as_ref(), false);
    kv_str(&mut s, "branch", run.branch, false);
    if let Some(c) = run.commit {
        kv_str(&mut s, "commit", c, false);
    }
    kv_str(&mut s, "cfg", run.cfg, false);
    kv_str(&mut s, "preset_cfg", run.preset_cfg, false);
    kv_str(&mut s, "preset_build", run.preset_build, false);
    if let Some(a) = run.artifact {
        kv_str(&mut s, "artifact", a.to_string_lossy().as_ref(), false);
    }

    // phases-Objekt
    s.push_str(",\"phases\":{");
    let mut first = true;
    for (k, v) in phases {
        if !first { s.push(','); }
        first = false;
        s.push('\"'); s.push_str(&escape_json_str(k)); s.push_str("\":");
        s.push_str(&v.to_string());
    }
    s.push('}');

    // optionale Counter
    if let Some(cn) = counters_opt {
        s.push_str(",\"counters\":{");
        let mut first = true;
        for (k, v) in cn {
            if !first { s.push(','); }
            first = false;
            s.push('\"'); s.push_str(&escape_json_str(k)); s.push_str("\":");
            s.push_str(&v.to_string());
        }
        s.push('}');
    }

    s.push_str("}\n");

    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(runs_jsonl_path(workdir))?;
    f.write_all(s.as_bytes())?;
    f.flush()?;
    Ok(())
}

// ---------- helpers: Dateipfade & JSON-Minimalschreiber ----------

fn metrics_dir(workdir: &Path) -> PathBuf {
    workdir.join(".build_metrics")
}

fn metrics_file_path(workdir: &Path) -> PathBuf {
    metrics_dir(workdir).join("metrics.json")
}

fn runs_jsonl_path(workdir: &Path) -> PathBuf {
    metrics_dir(workdir).join("runs.jsonl")
}

/// Very small JSON writer: {"phases":{"k":123,"x|y":456}}
fn save_json_map_atomically(path: &Path, map: &HashMap<String, u128>) -> std::io::Result<()> {
    let tmp = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("metrics.json.tmp{}", now_millis()));

    // Compose JSON in-memory -> atomisch
    let mut out = String::with_capacity(64 + map.len() * 32);
    out.push_str("{\"phases\":{");
    let mut first = true;
    for (k, v) in map {
        if !first { out.push(','); }
        first = false;
        out.push('\"');
        out.push_str(&escape_json_str(k));
        out.push_str("\":");
        out.push_str(&v.to_string());
    }
    out.push_str("}}\n");

    {
        let mut f = File::create(&tmp)?;
        f.write_all(out.as_bytes())?;
        f.sync_all()?;
    }
    if path.exists() {
        let _ = fs::remove_file(path);
    }
    fs::rename(&tmp, path)
}

/// Very small JSON reader for {"phases":{ "key": number, ... }}
fn load_json_map(path: &Path) -> Option<HashMap<String, u128>> {
    let mut s = String::new();
    let mut f = File::open(path).ok()?;
    f.read_to_string(&mut s).ok()?;

    // "phases" finden
    let ph_idx = s.find("\"phases\"")?;
    let brace = s[ph_idx..].find('{')?;
    let start = ph_idx + brace + 1;

    // key:number-Paare scannen bis '}' der phases-Map
    let mut depth = 1usize;
    let bytes = s.as_bytes();
    let mut i = start;
    let mut map = HashMap::new();

    while i < bytes.len() && depth > 0 {
        while i < bytes.len()
            && (((bytes[i] as char).is_ascii_whitespace()) || bytes[i] == b',')
        { i += 1; }
        if i >= bytes.len() { break; }
        if bytes[i] == b'}' {
            depth -= 1; i += 1; continue;
        }
        if bytes[i] != b'"' { break; }

        // key
        i += 1;
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'"' { i += 1; }
        if i >= bytes.len() { break; }
        let key = &s[key_start..i];
        i += 1;

        // :
        while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() { i += 1; }
        if i >= bytes.len() || bytes[i] != b':' { break; }
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() { i += 1; }

        // number
        let num_start = i;
        while i < bytes.len() && (bytes[i] as char).is_ascii_digit() { i += 1; }
        if num_start == i { break; }
        let num_str = &s[num_start..i];
        if let Ok(val) = num_str.parse::<u128>() {
            map.insert(key.to_string(), val);
        }
    }

    Some(map)
}

/// Neueste Geschwistermappe (lexikographisch) mit .build_metrics/metrics.json finden.
fn find_latest_sibling_metrics_json(workdir: &Path) -> Option<PathBuf> {
    let parent = workdir.parent()?;
    let current_name = workdir.file_name()?.to_string_lossy().to_string();

    let mut names_paths: Vec<(String, PathBuf)> = Vec::new();
    if let Ok(rd) = fs::read_dir(parent) {
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() { continue; }
            let name = e.file_name().to_string_lossy().to_string();
            if name == current_name { continue; }
            let candidate = p.join(".build_metrics").join("metrics.json");
            if candidate.is_file() {
                names_paths.push((name, candidate));
            }
        }
    }
    names_paths.sort_by(|a, b| b.0.cmp(&a.0));
    names_paths.into_iter().map(|(_, p)| p).next()
}

/// Obsolete Root-Tempfiles löschen (.build_metrics.tmp*)
fn cleanup_root_tmp_files(workdir: &Path) {
    if let Ok(rd) = fs::read_dir(workdir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_file() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with(".build_metrics.tmp") {
                        let _ = fs::remove_file(&p);
                    }
                }
            }
        }
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

// ----- kleine JSON-Helfer (ASCII-only) -----

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for b in s.as_bytes() {
        match *b {
            b'\\' => { out.push('\\'); out.push('\\'); }
            b'"'  => { out.push('\\'); out.push('"'); }
            b'\n' => { out.push('\\'); out.push('n'); }
            b'\r' => { out.push('\\'); out.push('r'); }
            b'\t' => { out.push('\\'); out.push('t'); }
            _ if *b < 0x20 => { out.push_str(&format!("\\u{:04x}", *b as u16)); }
            _ => out.push(*b as char),
        }
    }
    out
}

fn kv_sep(s: &mut String) { if !s.ends_with('{') { s.push(','); } }

fn kv_str(s: &mut String, k: &str, v: &str, first: bool) {
    if !first { kv_sep(s); }
    s.push('\"'); s.push_str(k); s.push_str("\":\"");
    s.push_str(&escape_json_str(v));
    s.push('\"');
}

fn kv_u128(s: &mut String, k: &str, v: u128, first: bool) {
    if !first { kv_sep(s); }
    s.push('\"'); s.push_str(k); s.push_str("\":");
    s.push_str(&v.to_string());
}

fn kv_i32(s: &mut String, k: &str, v: i32, first: bool) {
    if !first { kv_sep(s); }
    s.push('\"'); s.push_str(k); s.push_str("\":");
    s.push_str(&v.to_string());
}

fn kv_bool(s: &mut String, k: &str, v: bool, first: bool) {
    if !first { kv_sep(s); }
    s.push('\"'); s.push_str(k); s.push_str("\":");
    s.push_str(if v { "true" } else { "false" });
}
