///// Otter: Mini-Entrypoint – delegiert (full/clean/autogit) + ENV-Zweimodus OTTER_OP=branch|build.
///// Schneefuchs: /branch committet & pusht jetzt (autogit) – “wupp” als Default-Branch; ASCII-Logs.
///// Maus: Null-Magie, deterministisch; remote=origin; Build-Mode pusht nach erfolgreichem Full.
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

// ------------------------------- helpers --------------------------------------

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

fn git_current_branch(root: &Path) -> Option<String> {
    let root_s = root.to_str()?;
    let out = Command::new("git")
        .args(["-C", root_s, "rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s == "HEAD" || s.is_empty() { None } else { Some(s) }
}

fn git_is_repo(root: &Path) -> bool {
    let root_s = match root.to_str() { Some(s) => s, None => return false };
    Command::new("git")
        .args(["-C", root_s, "rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn git_local_branch_exists(root: &Path, name: &str) -> bool {
    let root_s = match root.to_str() { Some(s) => s, None => return false };
    Command::new("git")
        .args(["-C", root_s, "show-ref", "--verify", "--quiet", &format!("refs/heads/{}", name)])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn git_remote_branch_exists(root: &Path, remote: &str, name: &str) -> bool {
    let root_s = match root.to_str() { Some(s) => s, None => return false };
    Command::new("git")
        .args(["-C", root_s, "ls-remote", "--exit-code", "--heads", remote, name])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn git_checkout_new_local(root: &Path, name: &str) -> anyhow::Result<()> {
    let root_s = root.to_str().ok_or_else(|| anyhow::anyhow!("bad path"))?;
    let st = Command::new("git").args(["-C", root_s, "checkout", "-b", name]).status()?;
    if !st.success() {
        anyhow::bail!("git checkout -b {} failed", name);
    }
    Ok(())
}

fn git_checkout_from_remote(root: &Path, remote: &str, name: &str) -> anyhow::Result<()> {
    let root_s = root.to_str().ok_or_else(|| anyhow::anyhow!("bad path"))?;
    let st = Command::new("git").args(["-C", root_s, "checkout", "-b", name, &format!("{}/{}", remote, name)]).status()?;
    if !st.success() {
        anyhow::bail!("git checkout -b {} {}/{} failed", name, remote, name);
    }
    Ok(())
}

fn git_checkout_existing(root: &Path, name: &str) -> anyhow::Result<()> {
    let root_s = root.to_str().ok_or_else(|| anyhow::anyhow!("bad path"))?;
    let st = Command::new("git").args(["-C", root_s, "checkout", name]).status()?;
    if !st.success() {
        anyhow::bail!("git checkout {} failed", name);
    }
    Ok(())
}

// ----------------------------- branch op (ENV) --------------------------------
// Neu: Nach Branch-Checkout führt der Runner ein Autogit (add/commit/push -u) aus,
// damit „/branch“ IMMER hochlädt – wie gewünscht.
fn branch_mode_run(root: &Path) -> anyhow::Result<bool> {
    use crate::runner::runner_term::out_info;

    let remote = "origin";
    let name = std::env::var("OTTER_BRANCH").unwrap_or_else(|_| "wupp".to_string());

    if !git_is_repo(root) {
        anyhow::bail!("Not a Git repository: {}", prockit::display_path(root));
    }

    out_info("BRANCH", &format!("target='{}' remote='{}'", name, remote));

    if git_local_branch_exists(root, &name) {
        out_info("BRANCH", &format!("checkout local {}", name));
        git_checkout_existing(root, &name)?;
    } else if git_remote_branch_exists(root, remote, &name) {
        out_info("BRANCH", &format!("create from {}/{}", remote, name));
        git_checkout_from_remote(root, remote, &name)?;
    } else {
        out_info("BRANCH", &format!("create new {}", name));
        git_checkout_new_local(root, &name)?;
    }

    // Autogit erledigt add/commit und (falls kein Upstream) push -u.
    let pushed_ok = commands::autogit::run(root, None, false, remote, Some(&name), true).unwrap_or(1) == 0;
    out_info("BRANCH", if pushed_ok { "upload done" } else { "upload had issues" });
    Ok(pushed_ok)
}

// ---------------------------------- main --------------------------------------

fn main() {
    // ANSI/VT einschalten (zentral, ohne doppeltes FFI). Fällt still zurück, falls nicht möglich.
    crate::runner::runner_term::enable_ansi();

    // Root früh bestimmen (ENV bevorzugt, damit /branch minimal bleibt)
    let root: PathBuf = std::env::var_os("OTTER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap());

    // --- Früher Ausstieg für OTTER_OP=branch (keine Clap-Args nötig) ----------
    if let Ok(op) = std::env::var("OTTER_OP") {
        if op.eq_ignore_ascii_case("branch") {
            match branch_mode_run(&root) {
                Err(e) => {
                    eprintln!("[ERROR] {}", e);
                    std::process::exit(1);
                }
                Ok(pushed_ok) => {
                    let b = git_current_branch(&root).unwrap_or_else(|| "?".into());
                    summary::print_end_summary(summary::EndSummary {
                        success: true,
                        exit_code: 0,
                        started_ms: utils::epoch_ms(),
                        elapsed_ms: 0,
                        artifact_path: None,
                        commit_short: git_short_hash(&root),
                        commit_branch: Some(format!("origin/{}", b)),
                        autogit_pushed: pushed_ok,
                        notes: vec!["branch-mode".to_string()],
                    });
                    return;
                }
            }
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

    // Erfolgreicher Durchlauf – optionaler Autogit bei Full.
    let mut autogit_ok = true;
    let curr_branch = git_current_branch(&root)
        .or_else(|| std::env::var("OTTER_BRANCH").ok())
        .unwrap_or_else(|| "wupp".to_string());

    if auto_commit_after {
        autogit_ok = commands::autogit::run(&root, None, false, "origin", Some(&curr_branch), true).is_ok();
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
        commit_branch: Some(format!("origin/{}", curr_branch)),
        autogit_pushed: autogit_ok,
        notes: Vec::new(),
    });
}
