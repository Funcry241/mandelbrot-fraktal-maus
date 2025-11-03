///// Otter: Branch-Orchestrator – Build → ZIP (Rust) → Autogit → Summary → magenta HINT zum Starten.
///// Schneefuchs: Keine PowerShell; deterministisches Packaging; sauberes Logging; Windows-Run-Hinweis am Ende.
///// Maus: ASCII-Logs, OTTER_* Overrides, Default-Branch „wupp“, zwei pinke [HINT]-Zeilen nach der Summary.
///// Datei: rust/otter_proc/src/ops_branch.rs

use std::ffi::OsStr;
use std::path::Path;

use crate::artifact::find_artifact;
use crate::commands;
use crate::runner::runner_term;
use crate::summary;
use crate::utils::epoch_ms;

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_opt_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|s| s.parse::<u32>().ok())
}

fn current_branch_or_default(root: &Path) -> String {
    crate::vcs::git_current_branch(root)
        .or_else(|| std::env::var("OTTER_BRANCH").ok())
        .unwrap_or_else(|| "wupp".to_string())
}

/// Öffentlicher Einstieg für den Branch-Pfad.
/// Ablauf: Build (Full) → Pack (Rust) → Autogit push → Summary (+ magenta HINT).
/// Rückgabe: Prozess-Exitcode.
pub fn exec(root: &Path) -> i32 {
    runner_term::enable_ansi();

    let start_ms = epoch_ms();
    let cfg   = env_str("OTTER_CFG", "RelWithDebInfo");
    let cp    = std::env::var("OTTER_CONFIGURE_PRESET").ok();
    let bp    = std::env::var("OTTER_BUILD_PRESET").ok();
    let par   = env_opt_u32("OTTER_PARALLEL");
    let br    = current_branch_or_default(root);

    crate::runner::runner_term::out_info(
        "RUNNER",
        &format!(
            "branch-mode start ts_ms={} root={} cfg={} branch={}",
            start_ms,
            crate::prockit::display_path(root),
            cfg,
            br
        ),
    );

    // 1) Full Build fahren
    let build_rc = commands::full::run(root, &cfg, cp.as_deref(), bp.as_deref(), par);
    let (ok_build, code_build) = match build_rc {
        Ok(code) => (code == 0, code),
        Err(e) => {
            eprintln!("[ERROR] full build error: {}", e);
            return 1;
        }
    };
    if !ok_build {
        // Früh zusammenfassen (Build fehlgeschlagen)
        let end_ms = epoch_ms();
        let artifact_str = find_artifact(root).as_ref().map(|p| p.to_string_lossy().to_string());
        summary::print_end_summary(summary::EndSummary {
            success: false,
            exit_code: code_build,
            started_ms: start_ms,
            elapsed_ms: end_ms.saturating_sub(start_ms),
            artifact_path: artifact_str,
            commit_short: crate::vcs::git_short_hash(root),
            commit_branch: Some(format!("origin/{}", br)),
            autogit_pushed: false,
            notes: vec!["build failed before packing".into()],
        });
        return code_build;
    }

    // 2) Quellen packen (Rust, ohne PowerShell)
    let zip_path = match commands::pack::run(root, None, false) {
        Ok(p) => {
            crate::runner::runner_term::out_info("RUNNER", &format!("packed sources: {}", p.display()));
            Some(p)
        }
        Err(e) => {
            crate::runner::runner_term::out_warn("RUNNER", &format!("packing skipped/failed: {}", e));
            None
        }
    };

    // 3) Autogit push (auch wenn Pack scheitert — Build war OK)
    let msg = if let Some(z) = &zip_path {
        format!(
            "chore: branch build + pack ({})",
            z.file_name().and_then(OsStr::to_str).unwrap_or("zip")
        )
    } else {
        "chore: branch build".to_string()
    };
    let autogit_ok = commands::autogit::run(root, Some(msg), false, "origin", Some(&br), true)
        .unwrap_or(1)
        == 0;

    // 4) Abschluss-Summary
    let end_ms = epoch_ms();
    let artifact_path = find_artifact(root);
    let artifact_str = artifact_path.as_ref().map(|p| p.to_string_lossy().to_string());
    let mut notes = Vec::new();
    if let Some(z) = &zip_path {
        notes.push(format!("sources_zip={}", z.display()));
    }

    summary::print_end_summary(summary::EndSummary {
        success: true,
        exit_code: 0,
        started_ms: start_ms,
        elapsed_ms: end_ms.saturating_sub(start_ms),
        artifact_path: artifact_str.clone(),
        commit_short: crate::vcs::git_short_hash(root),
        commit_branch: Some(format!("origin/{}", br)),
        autogit_pushed: autogit_ok,
        notes,
    });

    // 5) Magenta HINT (wie starten)
    let magenta = "\x1b[95m";
    let reset = "\x1b[0m";
    let exe_hint = artifact_str.unwrap_or_else(|| ".\\build\\mandelbrot_otterdream.exe".to_string());

    println!("{}[HINT] Run now:{}", magenta, reset);
    println!(
        "{}[HINT] $env:Path=\"{}\\vcpkg_installed\\x64-windows\\bin;$env:Path\"; {}{}",
        magenta,
        crate::prockit::display_path(root),
        exe_hint,
        reset
    );

    0
}
