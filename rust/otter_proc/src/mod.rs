///// Otter: Modulbaum - Orchestrator (full), Cache-Wächter, Windows-Env, Pack + Export integriert.
///// Schneefuchs: Reihenfolge kompatibel; keine Fremd-Abhängigkeiten nach außen.
///// Maus: Minimale öffentliche Oberfläche.
///// Datei: rust/otter_proc/src/commands/mod.rs

pub mod autogit;
pub mod clean;
pub mod detect;
pub mod envkit;

pub mod cacheguard;
pub mod winenv;
pub mod full;

pub mod export; // für artifacts/zip (CLI: export)
pub mod pack;   // low-level packer (von export genutzt)
