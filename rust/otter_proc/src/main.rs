///// Otter: Mini-Entrypoint – delegiert an (full, clean, autogit) und committet nach erfolgreichem Full-Build.
///// Schneefuchs: Clap-Parser; PS-5.1-kompatible CMake-Aufrufe; ASCII-Logs.
///// Maus: Kein Over-Engineering; nur benötigte Imports.
///// Datei: rust/otter_proc/src/main.rs

mod utils;         // minimales Helfer-Modul (epoch_ms)
mod prockit;       // Runner-Helpers (Proc/Git/Guards)
mod cli;
mod commands;
mod build_metrics; // Zentral: .build_metrics (ASCII), Seeding & atomisches Speichern
mod runner;        // für crate::runner::runner_term::{enable_ansi,color_enabled,out_info}
mod summary;       // ASCII/ANSI Endblock-Formatter

use clap::Parser;
use cli::{Cli, Commands};
use std::path::{Path, PathBuf};
use std::process::Command;

fn fmt_exists(b: bool) -> String {
    if crate::runner::runner_term::color_enabled() {
        if b { "\x1b[32myes\x1b[0m".to_string() } else { "\x1b[33mno\x1b[0m".to_string() }
    } else {
        if b { "yes".to_string() } else { "no".to_string() }
    }
}

fn log_candidate(root: &Path, rel: &str) -> (PathBuf, bool) {
    let p = root.join(rel);
    let exists = p.is_file();
    crate::runner::runner_term::out_info(
        "RUNNER",
        &format!("artifact-candidate: {} exists={}", p.display(), fmt_exists(exists)),
    );
    (p, exists)
}

fn find_artifact(root: &Path) -> Option<PathBuf> {
    let candidates = [
        "build/RelWithDebInfo/mandelbrot_otterdream.exe",
        "build/bin/RelWithDebInfo/mandelbrot_otterdream.exe",
        "build/bin/mandelbrot_otterdream.exe",
        "build/mandelbrot_otterdream.exe",
    ];
    for rel in candidates {
        let (p, exists) = log_candidate(root, rel);
        if exists {
            crate::runner::runner_term::out_info("RUNNER", &format!("artifact: {}", p.display()));
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
    // ANSI/VT einschalten (zentral, ohne doppeltes FFI). Fällt still zurück, falls nicht möglich.
    crate::runner::runner_term::enable_ansi();

    let cli = Cli::parse();
    let root = cli.root.unwrap_or_else(|| std::env::current_dir().unwrap());

    // Vor dem konsumierenden match merken, ob es ein Full-Aufruf ist (verhindert E0382).
    let auto_commit_after = matches!(&cli.command, Commands::Full { .. });

    // Startzeit – deterministisch im Log
    let start_ms = utils::epoch_ms();
    crate::runner::runner_term::out_info("RUNNER", &format!("ts_ms={} root={}", start_ms, prockit::display_path(&root)));

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

    // Fakten für die Abschlusszusammenfassung sammeln
    let end_ms = utils::epoch_ms();
    let elapsed_ms = end_ms.saturating_sub(start_ms);
    let artifact = find_artifact(&root);
    let git_hash = git_short_hash(&root);

    // Hübscher, stabiler ASCII/ANSI-Endblock
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
