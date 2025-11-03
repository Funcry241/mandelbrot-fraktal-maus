///// Otter: Mini-Entrypoint – delegiert (full/clean/autogit) + ENV-Zweimodus OTTER_OP=branch|build.
///// Schneefuchs: /branch committet & pusht jetzt (autogit) und baut anschließend; “wupp” Default-Branch.
///// Maus: Null-Magie, deterministisch; remote=origin; /build pusht nach erfolgreichem Full; ASCII-Logs.
///// Datei: rust/otter_proc/src/main.rs

mod utils;         // minimales Helfer-Modul (epoch_ms)
mod prockit;       // Runner-Helpers (Proc/Git/Guards)
mod cli;
mod commands;
mod build_metrics; // Zentral: .build_metrics (ASCII), Seeding & atomisches Speichern
mod runner;        // für crate::runner::runner_term::{enable_ansi,color_enabled,out_info}
mod summary;       // ASCII/ANSI Endblock-Formatter

// neu
mod vcs;           // Git-Utilities (quiet checkout etc.)
mod artifact;      // Artefakt-Suche (build/… -> exe)
mod ops_branch;    // /branch-Orchestrierung (Checkout→Build→Pack→Push→Summary)

use clap::Parser;
use cli::{Cli, Commands};
use std::path::PathBuf;

use crate::artifact::find_artifact;
use crate::runner::runner_term;
use crate::vcs::{git_current_branch, git_short_hash};

fn main() {
    // ANSI/VT einschalten (zentral). Fällt still zurück, falls nicht möglich.
    runner::runner_term::enable_ansi();

    // Root früh bestimmen (ENV bevorzugt, damit /branch minimal bleibt)
    let root: PathBuf = std::env::var_os("OTTER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // --- ENV-Pfad: OTTER_OP=branch → Branch+Build (Dev/unstable) via ops_branch ---
    if let Ok(op) = std::env::var("OTTER_OP") {
        if op.eq_ignore_ascii_case("branch") {
            std::process::exit(ops_branch::exec(&root));
        }
    }

    // --- Normale CLI (full/clean/autogit) -------------------------------------
    let cli = Cli::parse();
    // Wenn CLI root gesetzt hat, nutze das, sonst ENV/aktuelles Verzeichnis.
    let root = cli.root.unwrap_or(root);

    // Vor dem konsumierenden match merken, ob es ein Full-Aufruf ist (verhindert E0382).
    let auto_commit_after = matches!(&cli.command, Commands::Full { .. });

    // Startzeit – deterministisch im Log
    let start_ms = utils::epoch_ms();
    runner_term::out_info(
        "RUNNER",
        &format!("ts_ms={} root={}", start_ms, prockit::display_path(&root)),
    );

    // Jetzt cli.command konsumieren – danach nicht mehr verwenden.
    let res = match cli.command {
        Commands::Full { cfg, configure_preset, build_preset, parallel } => {
            let rc = commands::full::run(
                &root,
                &cfg,
                configure_preset.as_deref(),
                build_preset.as_deref(),
                parallel,
            );

            // Nach erfolgreichem Full: Metrics-Run schreiben (Stable)
            if let Ok(code) = rc {
                let end_ms = utils::epoch_ms();
                let artifact = find_artifact(&root);
                let branch = git_current_branch(&root)
                    .or_else(|| std::env::var("OTTER_BRANCH").ok())
                    .unwrap_or_else(|| "wupp".to_string());
                let hash = git_short_hash(&root);
                let success = code == 0;

                // Metrics laden (panic-sicher) und erweiterten Run schreiben
                if let Ok((m, _, _)) =
                    std::panic::catch_unwind(|| build_metrics::BuildMetrics::load_or_seed(&root))
                {
                    let _ = build_metrics::write_extended_run(
                        &root,
                        build_metrics::RunSummary {
                            op: "build",
                            channel: "stable",
                            ts_ms: start_ms,
                            elapsed_ms: end_ms.saturating_sub(start_ms),
                            success,
                            exit_code: code,
                            root: &root,
                            branch: &branch,
                            commit: hash.as_deref(),
                            cfg: &cfg,
                            preset_cfg: configure_preset.as_deref().unwrap_or("windows-msvc"),
                            preset_build: build_preset.as_deref().unwrap_or("windows-build"),
                            artifact: artifact.as_deref(),
                        },
                        // Phasen aus Metrics
                        &m.snapshot(),
                        None,
                    );
                }
            }

            rc
        }
        Commands::Clean { dry_run, hard, extra } =>
            commands::clean::run(&root, dry_run, hard, &extra),
        Commands::Autogit { message, allow_empty, remote, branch, auto_https_fallback } =>
            commands::autogit::run(&root, message, allow_empty, &remote, branch.as_deref(), auto_https_fallback),
    };

    // Fehlerfall: sofort beenden
    let (run_ok, run_code) = match res {
        Ok(code) => (code == 0, code),
        Err(e) => {
            eprintln!("[ERROR] {}", e);
            std::process::exit(1);
        }
    };

    // Erfolgreicher Durchlauf – optionaler Autogit bei Full und nur bei Erfolg.
    let mut autogit_ok = false;
    let curr_branch = git_current_branch(&root)
        .or_else(|| std::env::var("OTTER_BRANCH").ok())
        .unwrap_or_else(|| "wupp".to_string());

    if auto_commit_after && run_ok {
        autogit_ok = commands::autogit::run(&root, None, false, "origin", Some(&curr_branch), true).is_ok();
    }

    // Fakten für die Abschlusszusammenfassung sammeln
    let end_ms = utils::epoch_ms();
    let elapsed_ms = end_ms.saturating_sub(start_ms);
    let artifact = find_artifact(&root);
    let git_hash = git_short_hash(&root);

    // Hübscher, stabiler ASCII/ANSI-Endblock (zeigt realen Exitcode/Status)
    summary::print_end_summary(summary::EndSummary {
        success: run_ok,
        exit_code: run_code,
        started_ms: start_ms,
        elapsed_ms,
        artifact_path: artifact.map(|p| p.to_string_lossy().to_string()),
        commit_short: git_hash,
        commit_branch: Some(format!("origin/{}", curr_branch)),
        autogit_pushed: autogit_ok,
        notes: Vec::new(),
    });
}
