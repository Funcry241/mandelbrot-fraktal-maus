///// Otter: Modulbaum gestrafft — Orchestrator (full), Cache-Wächter, Windows-Env isoliert.
/// /// Schneefuchs: Reihung kompatibel zu bestehendem Code; keine Fremd-Abhängigkeiten.
/// /// Maus: Minimale öffentliche Oberfläche; zukünftige Erweiterungen ohne Bruch möglich.
/// /// Datei: rust/otter_proc/src/commands/mod.rs
pub mod autogit;
pub mod clean;
pub mod detect;
pub mod envkit;

pub mod cacheguard;
pub mod winenv;
pub mod full;
