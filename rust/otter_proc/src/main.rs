///// Otter: Mini-Entrypoint – delegiert (full/clean/autogit/export) + ENV-Zweimodus OTTER_OP=branch|build.
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
    // ANSI/VT einschalten (zentral). Fällt still zurück, falls nicht möglich.
    runner_term::enable_ansi();

    // Root früh bestimmen (ENV bevorzugt, damit /branch minimal bleibt)
    let root: PathBuf = std::env::var_os("OTTER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // ----------------------------- ENV-Zweimodus ------------------------------
    // OTTER_OP=branch → 1) autogit (origin/wupp)  2) danach Full-Build
    // OTTER_OP=build  → Full-Build, danach autogit (origin/<HEAD or wupp>)
    let mut branch_mode = false;
    let mut forced_cmd: Option<Commands> = None;
    let mut autogit_after_full = false; // wird ggf. durch /build gesetzt

    if let Ok(op) = std::env::var("OTTER_OP") {
        if op.eq_ignore_ascii_case("branch") {
            branch_mode = true;
            // nach dem Autogit bauen wir standardmäßig RelWithDebInfo
            forced_cmd = Some(Commands::Full {
                cfg: "RelWithDebInfo".to_string(),
                configure_preset: None,
                build_preset: None,
                parallel: None,
            });
            autogit_after_full = false; // wichtig: bei /branch NICHT nochmal nach dem Build pushen
        } else if op.eq_ignore_ascii_case("build") {
            forced_cmd = Some(Commands::Full {
                cfg: "RelWithDebInfo".to_string(),
                configure_preset: None,
                build_preset: None,
                parallel: None,
            });
            autogit_after_full = true; // /build pusht nach erfolgreichem Full
        }
    }

    // ----------------------------- CLI-Parsen --------------------------------
    // CLI nur parsen, wenn kein ENV-Command erzwungen ist
    let cli = if forced_cmd.is_none() { Some(Cli::parse()) } else { None };
    // Root-Pfad: CLI > ENV > cwd
    let root = cli.as_ref().and_then(|c| c.root.clone()).unwrap_or(root);

    // Endgültigen Command bestimmen (ENV erzwingt, sonst CLI)
    // Wichtig: nicht `forced_cmd` nach `unwrap_or_else` bewegen und danach erneut verwenden.
    // Wir merken uns vorher, ob ENV forciert hat:
    let had_forced = forced_cmd.is_some();
    let mut command = if let Some(cmd) = forced_cmd {
        cmd
    } else {
        // safe: wenn kein forced_cmd, gibt es ein CLI
        cli.unwrap().command
    };

    // Falls kein ENV-Flag, entscheidet OTTER_UPLOAD (0/1) über Autogit nach Full:
    if !had_forced {
        autogit_after_full = matches!(command, Commands::Full { .. }) && env_truthy("OTTER_UPLOAD");
    }

    // Startzeit – deterministisch im Log
    let start_ms = utils::epoch_ms();
    runner_term::out_info(
        "RUNNER",
        &format!("ts_ms={} root={}", start_ms, prockit::display_path(&root)),
    );

    // -------------------------------- /branch --------------------------------
    // Gemäß Vorgabe: zuerst commit/push, DANN bauen.
    if branch_mode {
        let branch = std::env::var("OTTER_BRANCH").unwrap_or_else(|_| "wupp".to_string());
        // commit+push jetzt (remote fix: origin)
        let pre_rc: anyhow::Result<i32> =
            commands::autogit::run(&root, None, false, "origin", Some(&branch), true)
                .map(|_| 0)
                .map_err(Into::into);

        if let Err(e) = pre_rc {
            // Frühfehler: wir bauen NICHT, aber geben sauberen Endblock aus.
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

        // Nach erfolgreichem Push bauen wir weiter:
        command = Commands::Full {
            cfg: "RelWithDebInfo".to_string(),
            configure_preset: None,
            build_preset: None,
            parallel: None,
        };
        // und NACH dem Build NICHT nochmal autogitten
        autogit_after_full = false;
    }

    // ------------------------------ Ausführung --------------------------------
    // Vereinheitlichtes Resultat: i32 Exitcode in anyhow::Result
    let res_code: anyhow::Result<i32> = match command {
        Commands::Export { out_dir, max_keep, dry_run } => {
            commands::export::run(&root, out_dir.as_deref(), max_keep, dry_run)
                .map(|_| 0)
                .map_err(Into::into)
        }
        Commands::Full { cfg, configure_preset, build_preset, parallel } => {
            commands::full::run(
                &root,
                &cfg,
                configure_preset.as_deref(),
                build_preset.as_deref(),
                parallel,
            )
            .map_err(Into::into)
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
    // Nicht sofort beenden – wir möchten IMMER den Endblock ausgeben.
    let (mut run_ok, mut run_code, mut notes) = match res_code {
        Ok(code) => (code == 0, code, Vec::<String>::new()),
        Err(e) => (false, 1, vec![format!("command failed: {}", e)]),
    };

    // Erfolgreicher Durchlauf – optionaler Autogit NUR wenn gewünscht.
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
                // Build war ok; Push-Fehler soll Exitcode ≠0 signalisieren:
                run_ok = false;
                run_code = 1;
            }
        }
    }

    // Fakten für die Abschlusszusammenfassung sammeln
    let end_ms = utils::epoch_ms();
    let elapsed_ms = end_ms.saturating_sub(start_ms);
    let artifact = find_artifact(&root);
    let git_hash = git_short_hash(&root);
    let curr_branch = git_current_branch(&root)
        .or_else(|| std::env::var("OTTER_BRANCH").ok())
        .unwrap_or_else(|| "wupp".to_string());

    // Hübscher, stabiler ASCII/ANSI-Endblock (zeigt realen Exitcode/Status)
    summary::print_end_summary(summary::EndSummary {
        success: run_ok,
        exit_code: run_code,
        started_ms: start_ms,
        elapsed_ms,
        artifact_path: artifact.map(|p| p.to_string_lossy().to_string()),
        commit_short: git_hash,
        commit_branch: Some(format!("origin/{}", curr_branch)),
        autogit_pushed: autogit_ok || branch_mode, // bei /branch wurde vorher gepusht
        notes,
    });

    // realer Exit
    std::process::exit(run_code);
}
