///// Otter: Commands-Registry; `detect` nur in Tests/mit Feature aktiv.
/// //// Schneefuchs: Gating verhindert Dead-Code-Warnungen im Produktionsbuild.
/// //// Maus: Nichts entfernt, nur sauber gruppiert.
/// //// Datei: rust/otter_proc/src/commands/mod.rs
#[cfg(any(test, feature = "win-probe"))]
pub mod detect;

pub mod autogit;
pub mod cacheguard;
pub mod clean;
pub mod envkit;
pub mod full;
pub mod winenv;
