///// Otter: ASCII-Metrics V1 – Phasenlaufzeiten (.build_metrics) lesen/schreiben; atomisches Update; tolerant beim Parsen.
///// Schneefuchs: Key=(sig,phase); einfache API (load/get_last_ms/upsert/save); kein Serde; optionales Seeding aus Nachbarordner.
///// Maus: Robuste Format-Toleranz; keine Kommentare in der Datei; Header nutzt 'version'.
///// Datei: rust/otter_proc/src/build_metrics.rs

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

fn metrics_path(root: &Path) -> PathBuf { root.join(".build_metrics") }

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
                first = false; // accept any first line as header/version tag
                continue;
            }
            if !line.starts_with("phase|") { continue; }
            let mut sig: Option<String> = None;
            let mut name: Option<String> = None;
            let mut last: Option<u128> = None;
            for part in line.split('|').skip(1) {
                if let Some(v) = part.strip_prefix("sig=")     { sig = Some(v.to_string()); }
                else if let Some(v) = part.strip_prefix("name=")    { name = Some(v.to_string()); }
                else if let Some(v) = part.strip_prefix("last_ms=") { last = v.parse::<u128>().ok(); }
            }
            if let (Some(s), Some(n), Some(ms)) = (sig, name, last) {
                db.phase_last_ms.insert((s, n), ms);
            }
        }
        db
    }

    /// Lädt aus `root`, seedet optional aus dem jüngsten Geschwisterordner (falls vorhanden),
    /// gibt zusätzlich den Pfad der Metrics-Datei und die Seed-Quelle zurück.
    pub fn load_or_seed(root: &Path) -> (Self, PathBuf, Option<PathBuf>) {
        let mut db = Self::load(root);
        let me = metrics_path(root);

        // Seed-Quelle finden: Parent scannen, jüngsten Nachbar-Ordner mit '.build_metrics' wählen
        let mut seed_from: Option<PathBuf> = None;
        if let Some(parent) = root.parent() {
            if let Ok(read) = fs::read_dir(parent) {
                let mut candidates: Vec<PathBuf> = read.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.is_dir())
                    .filter(|p| p.join(".build_metrics").exists())
                    .collect();
                candidates.sort(); // lexikographisch: unsere Ordner sind datumsartig → reicht
                if let Some(last) = candidates.into_iter().rev().find(|p| p != root) {
                    seed_from = Some(last.join(".build_metrics"));
                }
            }
        }
        if db.phase_last_ms.is_empty() {
            if let Some(seed) = seed_from.as_ref() {
                if let Ok(text) = fs::read_to_string(seed) {
                    let mut seeded = Self::new();
                    let mut first = true;
                    for line in text.lines() {
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        if first { first = false; continue; }
                        if !line.starts_with("phase|") { continue; }
                        let mut sig: Option<String> = None;
                        let mut name: Option<String> = None;
                        let mut last: Option<u128> = None;
                        for part in line.split('|').skip(1) {
                            if let Some(v) = part.strip_prefix("sig=")     { sig = Some(v.to_string()); }
                            else if let Some(v) = part.strip_prefix("name=")    { name = Some(v.to_string()); }
                            else if let Some(v) = part.strip_prefix("last_ms=") { last = v.parse::<u128>().ok(); }
                        }
                        if let (Some(s), Some(n), Some(ms)) = (sig, name, last) {
                            seeded.phase_last_ms.insert((s, n), ms);
                        }
                    }
                    if !seeded.phase_last_ms.is_empty() { db = seeded; }
                }
            }
        }
        (db, me, seed_from)
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
        out.push_str(&format!("{}\n", self.version)); // <- version-Feld wird gelesen → Warnung weg
        for ((sig, phase), ms) in self.phase_last_ms.iter() {
            out.push_str(&format!("phase|sig={}|name={}|last_ms={}\n", sig, phase, ms));
        }
        fs::write(&tmp, out.as_bytes())?;
        if p.exists() { let _ = fs::remove_file(&p); }
        fs::rename(tmp, p)
    }
}
