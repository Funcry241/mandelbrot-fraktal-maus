///// Otter: Build metrics stored under .build_metrics/metrics.json with atomic saves.
///// Schneefuchs: No external crates; Windows+POSIX; migrates from latest sibling; cleans root tmp files.
///// Maus: Simple JSON (phases -> ms), key = "<sig>|<phase>"; logs path for visibility.
///// Datei: rust/otter_proc/src/build_metrics.rs

use std::collections::HashMap;
use std::fs::{self, File};
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

        // Hygiene: remove old root-level .build_metrics.tmp* files from a prior bug
        cleanup_root_tmp_files(workdir);

        // Ensure directory exists
        if let Err(_e) = fs::create_dir_all(&dir) {
            // Even if mkdir fails, proceed – subsequent open will err and caller will see code=1
        }

        // Case 1: current metrics file exists → load
        if file.exists() {
            let m = load_json_map(&file).unwrap_or_default();
            return (BuildMetrics { phases_ms: m }, file, None);
        }

        // Case 2: try to seed from latest sibling directory (same parent)
        if let Some(seed_json) = find_latest_sibling_metrics_json(workdir) {
            // Load seed
            let m = load_json_map(&seed_json).unwrap_or_default();
            // Save immediately to establish local file
            let _ = save_json_map_atomically(&file, &m);
            return (BuildMetrics { phases_ms: m }, file, Some(seed_json));
        }

        // Case 3: nothing found → start empty
        let m = BuildMetrics::default();
        // Persist an empty file to avoid repeated seeding attempts
        let _ = save_json_map_atomically(&file, &m.phases_ms);
        (m, file, None)
    }

    pub fn get_last_ms(&self, sig: &str, phase: &str) -> Option<u128> {
        self.phases_ms.get(&Self::key(sig, phase)).copied()
    }

    pub fn upsert_phase_ms(&mut self, sig: &str, phase: &str, ms: u128) {
        self.phases_ms.insert(Self::key(sig, phase), ms);
    }

    pub fn save(&self, workdir: &Path) -> std::io::Result<()> {
        let file = metrics_file_path(workdir);
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent).ok();
        }
        save_json_map_atomically(&file, &self.phases_ms)
    }
}

// ---------- helpers ----------

fn metrics_dir(workdir: &Path) -> PathBuf {
    workdir.join(".build_metrics")
}

fn metrics_file_path(workdir: &Path) -> PathBuf {
    metrics_dir(workdir).join("metrics.json")
}

/// Very small JSON writer: {"phases":{"k":123,"x|y":456}}
fn save_json_map_atomically(path: &Path, map: &HashMap<String, u128>) -> std::io::Result<()> {
    let tmp = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            "metrics.json.tmp{}",
            now_millis()
        ));

    // Compose JSON into memory to keep write atomic
    let mut out = String::with_capacity(64 + map.len() * 32);
    out.push_str("{\"phases\":{");
    let mut first = true;
    for (k, v) in map {
        if !first { out.push(','); }
        first = false;
        // escape minimal (we only ever use ASCII sig/phase), so just wrap in quotes
        out.push('\"');
        out.push_str(k);
        out.push_str("\":");
        out.push_str(&v.to_string());
    }
    out.push_str("}}\n");

    // Write tmp
    {
        let mut f = File::create(&tmp)?;
        f.write_all(out.as_bytes())?;
        f.sync_all()?;
    }

    // Replace destination
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

    // Find the "phases" object
    let ph_idx = s.find("\"phases\"")?;
    let brace = s[ph_idx..].find('{')?;
    let start = ph_idx + brace + 1;

    // Scan key:value pairs until matching closing brace of the phases object
    let mut depth = 1usize;
    let bytes = s.as_bytes();
    let mut i = start;
    let mut map = HashMap::new();

    // Tiny state machine: "key" : number
    while i < bytes.len() && depth > 0 {
        // skip whitespace and commas
        while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() || (i < bytes.len() && bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() { break; }
        if bytes[i] == b'}' {
            depth -= 1;
            i += 1;
            continue;
        }
        if bytes[i] != b'"' { break; }
        // read key
        i += 1;
        let key_start = i;
        while i < bytes.len() && bytes[i] != b'"' { i += 1; }
        if i >= bytes.len() { break; }
        let key = &s[key_start..i];
        i += 1;

        // skip spaces and colon
        while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() { i += 1; }
        if i >= bytes.len() || bytes[i] != b':' { break; }
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() { i += 1; }
        // read number
        let num_start = i;
        while i < bytes.len() && (bytes[i] as char).is_ascii_digit() { i += 1; }
        if num_start == i { break; }
        let num_str = &s[num_start..i];
        if let Ok(val) = num_str.parse::<u128>() {
            map.insert(key.to_string(), val);
        }
        // loop continues; next iteration will skip comma/space
    }

    Some(map)
}

/// Find newest sibling folder (lexicographically) that has .build_metrics/metrics.json
fn find_latest_sibling_metrics_json(workdir: &Path) -> Option<PathBuf> {
    let parent = workdir.parent()?;
    let current_name = workdir.file_name()?.to_string_lossy().to_string();

    // Collect candidates
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
    // Sort by name desc (timestamps like 20251101_09 sort well)
    names_paths.sort_by(|a, b| b.0.cmp(&a.0));
    names_paths.into_iter().map(|(_, p)| p).next()
}

/// Remove obsolete root-level .build_metrics.tmp* files created by an older buggy implementation.
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
