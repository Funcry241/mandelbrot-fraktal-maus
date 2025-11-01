///// Otter: Mini entrypoint – delegiert an Module (full, clean, autogit); ASCII-only.
///// Schneefuchs: Nach erfolgreichem FULL immer Autogit (ohne Flags/Env) → origin/main.
///// Maus: prockit::display_path, utils::epoch_ms; robuste Exit-Logs.
///// Datei: rust/otter_proc/src/main.rs

mod utils;     // minimales Helfer-Modul (epoch_ms)
mod prockit;   // Runner-Helpers (Proc/Git/Guards)
mod cli;
mod commands;

use clap::Parser;
use cli::{Cli, Commands};
use std::path::Path;

fn main() {
    let cli  = Cli::parse();
    let root = cli.root.unwrap_or_else(|| std::env::current_dir().unwrap());
    let ts_ms = utils::epoch_ms();
    println!("[RUNNER] ts_ms={} root={}", ts_ms, prockit::display_path(&root));

    // Merken, ob FULL erfolgreich war (Exitcode 0)
    let mut ran_full_ok = false;

    let res = match cli.command {
        Commands::Full { cfg, configure_preset, build_preset, parallel } => {
            let r = commands::full::run(&root, &cfg, configure_preset.as_deref(), build_preset.as_deref(), parallel);
            if let Ok(code) = r {
                if code == 0 { ran_full_ok = true; }
            }
            r
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

    // Immer automatisch commit/push nach erfolgreichem FULL (ohne Schalter)
    if ran_full_ok {
        if let Err(e) = commands::autogit::run(&root, None, false, "origin", None, true) {
            eprintln!("[AUTOGIT][ERR] {}", e);
            // kein exit – Build bleibt erfolgreich, Autogit ist Poststep
        }
    }
}
