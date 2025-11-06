///// Otter: Modulbaum gestrafft — Orchestrator (full), Cache-Wächter, Windows-Env, Pack (ZIP, Rust).
///// Schneefuchs: Reihung kompatibel zu bestehendem Code; keine Fremd-Abhängigkeiten.
///// Maus: Minimale öffentliche Oberfläche; zukünftige Erweiterungen ohne Bruch möglich.
///// Datei: rust/otter_proc/src/commands/mod.rs

pub mod autogit;
pub mod clean;

// Optionale Tools nur bei Feature "win-probe", damit der Default-Build warnungsfrei bleibt.
#[cfg(feature = "win-probe")]
pub mod detect;
#[cfg(feature = "win-probe")]
pub mod envkit;

pub mod cacheguard;
pub mod winenv;
pub mod full;
pub mod pack;
pub mod export; // eigenes Export-Modul

// Keine Reexports mehr; main.rs ruft commands::export::run bzw. commands::pack::run direkt.
