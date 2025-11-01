///// Otter: ASCII-Metrics V1 – Phasenlaufzeiten (.build_metrics) lesen/schreiben; atomisches Update; Seeding aus Geschwistern.
///// Schneefuchs: Key=(sig,phase); einfache API (load/get_last_ms/upsert/save/load_or_seed); kein Serde.
///// Maus: Robuste Format-Toleranz; keine lauten Logs im Modul; nur Rückgabewerte für den Aufrufer.
/// ///// Datei: rust/otter_proc/src/build_metrics.rs

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn metrics_path(root: &Path) -> PathBuf { root.join(".build_metrics") }

fn now_epoch_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
}

#[derive(Debug, Clone)]
pub struct BuildMetrics {
    // key = (sig, phase)
    phase_last_ms: HashMap<(String, String), u128>,
    version: String,
}

impl BuildMetrics {
    pub fn new() -> Self {
        Self { phase_last_ms: HashMap::new(), version: "V1".into() }
    }

    pub fn load(root: &Path) -> Self {
        let p = metrics_path(root);
        let Ok(text) = fs::read_to_string(&p) else { return Self::new() };
        let mut db = Self::new();
        let mut first = true;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            if first {
                first = false;
                // accept any first line as header
                continue;
            }
            // expected: phase|sig=...|name=...|last_ms=1234
            if !line.starts_with("phase|") { continue; }
            let mut sig: Option<String> = None;
            let mut name: Option<String> = None;
            let mut last: Option<u128> = None;
            for part in line.split('|').skip(1) {
                if let Some(v) = part.strip_prefix("sig=") { sig = Some(v.to_string()); }
                else if let Some(v) = part.strip_prefix("name=") { name = Some(v.to_string()); }
                else if let Some(v) = part.strip_prefix("last_ms=") { last = v.parse::<u128>().ok(); }
            }
            if let (Some(s), Some(n), Some(ms)) = (sig, name, last) {
                db.phase_last_ms.insert((s, n), ms);
            }
        }
        db
    }

    pub fn get_last_ms(&self, sig: &str, phase: &str) -> Option<u128> {
        self.phase_last_ms.get(&(sig.to_string(), phase.to_string())).copied()
    }

    pub fn upsert_phase_ms(&mut self, sig: &str, phase: &str, ms: u128) {
        self.phase_last_ms.insert((sig.to_string(), phase.to_string()), ms);
    }

    pub fn save(&self, root: &Path) -> std::io::Result<()> {
        let p = metrics_path(root);
        let tmp = p.with_extension(format!("tmp{}", std::process::id()));
        let mut out = String::new();
        out.push_str("V1\n");
        for ((sig, phase), ms) in self.phase_last_ms.iter() {
            out.push_str(&format!("phase|sig={}|name={}|last_ms={}\n", sig, phase, ms));
        }
        fs::write(&tmp, out.as_bytes())?;
        if p.exists() { fs::remove_file(&p).ok(); }
        fs::rename(tmp, p)
    }

    /// Lädt Metrics oder seeded sie vorher aus der "jüngsten" Geschwisterdatei im Elternordner.
    /// Rückgabe: (DB, ziel_pfad, optional: seed_source_pfad)
    pub fn load_or_seed(root: &Path) -> (Self, PathBuf, Option<PathBuf>) {
        let dst = metrics_path(root);
        if dst.exists() {
            return (Self::load(root), dst, None);
        }

        // try to find newest sibling's metrics in parent
        let parent = match root.parent() { Some(p) => p, None => return (Self::new(), dst, None) };
        let mut best_time: Option<SystemTime> = None;
        let mut best_src: Option<PathBuf> = None;

        let entries = match fs::read_dir(parent) {
            Ok(e) => e,
            Err(_) => return (Self::new(), dst, None),
        };

        for ent in entries {
            let Ok(ent) = ent else { continue };
            let p = ent.path();
            if !p.is_dir() { continue; }
            if p == root { continue; }
            let candidate = p.join(".build_metrics");
            if !candidate.exists() { continue; }
            let mtime = fs::metadata(&candidate).and_then(|m| m.modified()).unwrap_or(UNIX_EPOCH);
            match best_time {
                None => { best_time = Some(mtime); best_src = Some(candidate); }
                Some(bt) => if mtime > bt { best_time = Some(mtime); best_src = Some(candidate); },
            }
        }

        if let Some(src) = best_src {
            if let Ok(bytes) = fs::read(&src) {
                let tmp = dst.with_extension(format!("seed{}.tmp{}", now_epoch_ms(), std::process::id()));
                if fs::write(&tmp, &bytes).is_ok() {
                    if fs::rename(&tmp, &dst).is_err() {
                        let _ = fs::remove_file(&dst);
                        let _ = fs::rename(&tmp, &dst);
                    }
                    // even wenn rename scheitert, versuchen wir trotzdem zu laden (dst evtl. existiert)
                }
            }
            return (Self::load(root), dst, Some(src));
        }

        (Self::new(), dst, None)
    }
}
