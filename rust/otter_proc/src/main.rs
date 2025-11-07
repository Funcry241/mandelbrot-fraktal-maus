///// Otter: Mini-Entrypoint – delegiert (full/clean/autogit/export) + ENV-Zweimodus OTTER_OP=branch|build.
///// Schneefuchs: /branch committet & pusht → dann Build; “wupp” Default-Branch.
///// Maus: Null-Magie, deterministisch; remote=origin; /build pusht nach Full; ASCII-Logs.
///// Datei: rust/otter_proc/src/main.rs

mod utils;         // minimales Helfer-Modul (epoch_ms)
mod prockit;       // Runner-Helpers (Proc/Git/Guards)
mod cli;
mod commands;
mod build_metrics; // Zentral: .build_metrics (ASCII), Seeding & atomisches Speichern
mod runner;        // für crate::runner::runner_term::{enable_ansi,color_enabled,out_info}
mod summary;       // ASCII/ANSI Endblock-Formatter

// optional / vorhanden
mod vcs;           // Git-Utilities (quiet checkout etc.)
mod artifact;      // Artefakt-Suche (build/… -> exe)
// vorhanden, aber hier nicht mehr genutzt (keine "Magie"):
mod ops_branch;    // historisch: /branch-Orchestrierung

use clap::Parser;
use cli::{Cli, Commands};
use std::path::PathBuf;

use crate::artifact::find_artifact;
use crate::runner::runner_term;
use crate::vcs::{git_current_branch, git_short_hash};

fn env_truthy(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .map(|s| {
            let s = s.to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn main() {
    runner_term::enable_ansi();

    let root: PathBuf = std::env::var_os("OTTER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // ----------------------------- ENV-Zweimodus ------------------------------
    let mut branch_mode = false;
    let mut forced_cmd: Option<Commands> = None;
    let mut autogit_after_full = false;

    if let Ok(op) = std::env::var("OTTER_OP") {
        if op.eq_ignore_ascii_case("branch") {
            branch_mode = true;
            forced_cmd = Some(Commands::Full {
                cfg: "RelWithDebInfo".to_string(),
                configure_preset: None,
                build_preset: None,
                parallel: None,
            });
            autogit_after_full = false;
        } else if op.eq_ignore_ascii_case("build") {
            forced_cmd = Some(Commands::Full {
                cfg: "RelWithDebInfo".to_string(),
                configure_preset: None,
                build_preset: None,
                parallel: None,
            });
            autogit_after_full = true;
        }
    }

    // ----------------------------- CLI-Parsen --------------------------------
    let cli = if forced_cmd.is_none() { Some(Cli::parse()) } else { None };
    let root = cli.as_ref().and_then(|c| c.root.clone()).unwrap_or(root);

    let had_forced = forced_cmd.is_some();
    let mut command = if let Some(cmd) = forced_cmd { cmd } else { cli.unwrap().command };

    if !had_forced {
        autogit_after_full = matches!(command, Commands::Full { .. }) && env_truthy("OTTER_UPLOAD");
    }

    // Startzeit – deterministisch im Log
    let start_ms = utils::epoch_ms();
    runner_term::out_info("RUNNER", &format!("ts_ms={} root={}", start_ms, prockit::display_path(&root)));

    // -------------------------------- /branch --------------------------------
    if branch_mode {
        let branch = std::env::var("OTTER_BRANCH").unwrap_or_else(|_| "wupp".to_string());
        let pre_rc: anyhow::Result<i32> =
            commands::autogit::run(&root, None, false, "origin", Some(&branch), true)
                .map(|_| 0)
                .map_err(Into::into);

        if let Err(e) = pre_rc {
            let end_ms = utils::epoch_ms();
            let elapsed_ms = end_ms.saturating_sub(start_ms);
            let git_hash = git_short_hash(&root);
            summary::print_end_summary(summary::EndSummary {
                success: false,
                exit_code: 1,
                started_ms: start_ms,
                elapsed_ms,
                artifact_path: None,
                commit_short: git_hash,
                commit_branch: Some(format!("origin/{}", branch)),
                autogit_pushed: false,
                notes: vec![format!("autogit before build failed: {}", e)],
            });
            std::process::exit(1);
        }

        command = Commands::Full {
            cfg: "RelWithDebInfo".to_string(),
            configure_preset: None,
            build_preset: None,
            parallel: None,
        };
        autogit_after_full = false;
    }

    // ------------------------------ Ausführung --------------------------------
    let res_code: anyhow::Result<i32> = match command {
        Commands::Export { out_dir, max_keep, dry_run } => {
            commands::export::run(&root, out_dir.as_deref(), max_keep, dry_run)
                .map(|_p| 0)
                .map_err(Into::into)
        }
        Commands::Full { cfg, configure_preset, build_preset, parallel } => {
            let rc = commands::full::run(
                &root,
                &cfg,
                configure_preset.as_deref(),
                build_preset.as_deref(),
                parallel,
            );

            // Nach erfolgreichem Build ggf. Export anschieben:
            // branch_mode ⇒ Default **AN**, via OTTER_EXPORT_ON_BRANCH (1/true/on) steuerbar (default: true)
            // sonst ⇒ via OTTER_EXPORT_AFTER_BUILD steuerbar (default: false)
            if let Ok(code) = rc {
                let success = code == 0;

                // Build-Metriken erfassen
                let end_ms = utils::epoch_ms();
                let artifact = find_artifact(&root);
                let branch = git_current_branch(&root)
                    .or_else(|| std::env::var("OTTER_BRANCH").ok())
                    .unwrap_or_else(|| "wupp".to_string());
                let hash = git_short_hash(&root);

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
                        &m.snapshot(),
                        None,
                    );
                }

                // Export-Entscheidung
                let export_after_build = if branch_mode {
                    std::env::var("OTTER_EXPORT_ON_BRANCH")
                        .ok()
                        .map(|s| {
                            let s = s.to_ascii_lowercase();
                            // default: true (wenn gesetzt leer → false; wenn nicht gesetzt → None)
                            matches!(s.as_str(), "1" | "true" | "yes" | "on")
                        })
                        .unwrap_or(true)
                } else {
                    env_truthy("OTTER_EXPORT_AFTER_BUILD")
                };

                if success && export_after_build {
                    let _ = commands::export::run(
                        &root,
                        Some(&root.join("out").join("exports")),
                        5,
                        false,
                    ).map(|p| runner_term::out_info("EXPORT", &format!("wrote {}", p.display())));
                }
            }

            rc.map_err(Into::into)
        }
        Commands::Clean { dry_run, hard, extra } => {
            commands::clean::run(&root, dry_run, hard, &extra)
                .map(|_| 0)
                .map_err(Into::into)
        }
        Commands::Autogit { message, allow_empty, remote, branch, auto_https_fallback } => {
            commands::autogit::run(&root, message, allow_empty, &remote, branch.as_deref(), auto_https_fallback)
                .map(|_| 0)
                .map_err(Into::into)
        }
    };

    // -------------------------- Nachlauf + Summary ----------------------------
    let (mut run_ok, mut run_code, mut notes) = match res_code {
        Ok(code) => (code == 0, code, Vec::<String>::new()),
        Err(e) => (false, 1, vec![format!("command failed: {}", e)]),
    };

    let mut autogit_ok = false;
    if run_ok && autogit_after_full {
        let curr_branch = git_current_branch(&root)
            .or_else(|| std::env::var("OTTER_BRANCH").ok())
            .unwrap_or_else(|| "wupp".to_string());

        match commands::autogit::run(&root, None, false, "origin", Some(&curr_branch), true) {
            Ok(_) => autogit_ok = true,
            Err(e) => {
                autogit_ok = false;
                notes.push(format!("autogit after build failed: {}", e));
                run_ok = false;
                run_code = 1;
            }
        }
    }

    let end_ms = utils::epoch_ms();
    let elapsed_ms = end_ms.saturating_sub(start_ms);
    let artifact = find_artifact(&root);
    let git_hash = git_short_hash(&root);
    let curr_branch = git_current_branch(&root)
        .or_else(|| std::env::var("OTTER_BRANCH").ok())
        .unwrap_or_else(|| "wupp".to_string());

    summary::print_end_summary(summary::EndSummary {
        success: run_ok,
        exit_code: run_code,
        started_ms: start_ms,
        elapsed_ms,
        artifact_path: artifact.map(|p| p.to_string_lossy().to_string()),
        commit_short: git_hash,
        commit_branch: Some(format!("origin/{}", curr_branch)),
        autogit_pushed: autogit_ok || branch_mode,
        notes,
    });

    std::process::exit(run_code);
}
