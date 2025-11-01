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

use clap::Parser;
use cli::{Cli, Commands};

fn main() {
    let cli = Cli::parse();
    let root = cli.root.unwrap_or_else(|| std::env::current_dir().unwrap());

    // Vor dem konsumierenden match merken, ob es ein Full-Aufruf ist (verhindert E0382).
    let auto_commit_after = matches!(&cli.command, Commands::Full { .. });

    let ts_ms = utils::epoch_ms();
    println!("[RUNNER] ts_ms={} root={}", ts_ms, prockit::display_path(&root));

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

    // Nach erfolgreichem Full-Build automatisch commit/pushen.
    if auto_commit_after {
        let _ = commands::autogit::run(&root, None, false, "origin", Some("main"), true);
    }
}
