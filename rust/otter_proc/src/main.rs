///// Otter: Mini-Entrypoint – delegiert an (full, clean, autogit) und committet nach erfolgreichem Full-Build.
///// Schneefuchs: Clap-Parser; PS-5.1-kompatible CMake-Aufrufe; ASCII-Logs.
///// Maus: Kein Over-Engineering; nur benötigte Imports.
///// Datei: rust/otter_proc/src/main.rs

mod utils;         // minimales Helfer-Modul (epoch_ms)
mod prockit;       // Runner-Helpers (Proc/Git/Guards)
mod cli;
mod commands;
mod build_metrics; // Zentral: .build_metrics (ASCII), Seeding & atomisches Speichern
mod runner;        // <<— NEU: für crate::runner in winenv.rs
mod summary;       // <<— NEU: ASCII-Endblock-Formatter

use clap::Parser;
use cli::{Cli, Commands};
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_artifact(root: &Path) -> Option<PathBuf> {
    // Kandidaten wie im aktuellen Log geprüft
    let candidates = [
        "build/RelWithDebInfo/mandelbrot_otterdream.exe",
        "build/bin/RelWithDebInfo/mandelbrot_otterdream.exe",
        "build/bin/mandelbrot_otterdream.exe",
        "build/mandelbrot_otterdream.exe",
    ];
    for rel in candidates {
        let p = root.join(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn git_short_hash(root: &Path) -> Option<String> {
    let root_s = root.to_str()?;
    let out = Command::new("git")
        .args(["-C", root_s, "rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn main() {
    let cli = Cli::parse();
    let root = cli.root.unwrap_or_else(|| std::env::current_dir().unwrap());

    // Vor dem konsumierenden match merken, ob es ein Full-Aufruf ist (verhindert E0382).
    let auto_commit_after = matches!(&cli.command, Commands::Full { .. });

    // Startzeit – deterministisch im Log
    let start_ms = utils::epoch_ms();
    println!("[RUNNER] ts_ms={} root={}", start_ms, prockit::display_path(&root));

    // Jetzt cli.command konsumieren – danach nicht mehr verwenden.
    let res = match cli.command {
        Commands::Full { cfg, configure_preset, build_preset, parallel } => {
            commands::full::run(
                &root,
                &cfg,
                configure_preset.as_deref(),
                build_preset.as_deref(),
                parallel,
            )
        }
        Commands::Clean { dry_run, hard, extra } =>
            commands::clean::run(&root, dry_run, hard, &extra),
        Commands::Autogit { message, allow_empty, remote, branch, auto_https_fallback } =>
            commands::autogit::run(&root, message, allow_empty, &remote, branch.as_deref(), auto_https_fallback),
    };

    if let Err(e) = res {
        eprintln!("[ERROR] {}", e);
        std::process::exit(1);
    }

    // Erfolgreicher Durchlauf – optionaler Autogit bei Full
    let mut autogit_ok = true;
    if auto_commit_after {
        autogit_ok = commands::autogit::run(&root, None, false, "origin", Some("main"), true).is_ok();
    }

    // Fakten für die neue Abschluss-Zusammenfassung sammeln
    let end_ms = utils::epoch_ms();
    let elapsed_ms = end_ms.saturating_sub(start_ms);
    let artifact = find_artifact(&root);
    let git_hash = git_short_hash(&root);

    // Hübscher, stabiler ASCII-Endblock
    summary::print_end_summary(summary::EndSummary {
        success: true,
        exit_code: 0,
        started_ms: start_ms,
        elapsed_ms,
        artifact_path: artifact.map(|p| p.to_string_lossy().to_string()),
        commit_short: git_hash,
        commit_branch: Some("origin/main".to_string()),
        autogit_pushed: autogit_ok,
        notes: Vec::new(),
    });
}
