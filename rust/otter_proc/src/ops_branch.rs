///// Otter: Branch-Orchestrator – Build → pack ZIP (Rust) → Autogit → Summary → pinker Run-Hint.
///// Schneefuchs: Keine PowerShell; saubere Pfade ohne \\?\-Präfix; keine ungenutzten Imports; borrows statt Moves.
///// Maus: ASCII-Block am Ende; Branch default „wupp“; deterministische Logs.
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

fn win_path_str(p: &Path) -> String {
    // Windows-freundlich: Backslashes, ohne \\?\ Präfix
    let mut s = p.to_string_lossy().to_string();
    if s.starts_with(r"\\?\") { s = s[4..].to_string(); }
    s.replace('/', "\\")
}

fn print_magenta_run_hint(root: &Path, artifact: &Path) {
    let vcpkg_bin = root.join("vcpkg_installed").join("x64-windows").join("bin");
    let have_vcpkg_bin = vcpkg_bin.is_dir();

    let exe = win_path_str(artifact);
    let bin = win_path_str(&vcpkg_bin);

    // ANSI magenta
    println!("\x1b[95m+------------------------------------------------------------------+");
    println!("| RUN NOW                                                          |");
    println!("+------------------------------------------------------------------+");
    if have_vcpkg_bin {
        println!("| 1) Add DLL path for this session:                                |");
        println!("|    $env:Path=\"{};$env:Path\" |", bin);
        println!("|                                                                  |");
        println!("| 2) Launch the app:                                               |");
        println!("|    \"{}\" |", exe);
    } else {
        println!("| Launch the app:                                                  |");
        println!("|    \"{}\" |", exe);
        println!("| (vcpkg bin dir not found; if needed, add it to PATH manually)    |");
    }
    println!("+------------------------------------------------------------------+\x1b[0m");
}

/// Öffentlicher Einstieg für den Branch-Pfad.
/// Ablauf: Build (Full) → Pack (Rust) → Autogit push → Summary (+ Run-Hint).
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
        &format!("branch-mode start ts_ms={} root={} cfg={} branch={}",
                 start_ms, crate::prockit::display_path(root), cfg, br),
    );

    // 1) Full Build fahren
    let build_rc = commands::full::run(
        root,
        &cfg,
        cp.as_deref(),
        bp.as_deref(),
        par,
    );
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
        let artifact = find_artifact(root);
        summary::print_end_summary(summary::EndSummary {
            success: false,
            exit_code: code_build,
            started_ms: start_ms,
            elapsed_ms: end_ms.saturating_sub(start_ms),
            artifact_path: artifact.as_ref().map(|p| p.to_string_lossy().to_string()),
            commit_short: crate::vcs::git_short_hash(root),
            commit_branch: Some(format!("origin/{}", br)),
            autogit_pushed: false,
            notes: vec!["build failed before packing".into()],
        });
        return code_build;
    }

    // 2) Quellen packen (reiner Rust)
    let zip_path = match commands::pack::run(root, None, false) {
        Ok(p) => {
            crate::runner::runner_term::out_info("RUNNER",
                &format!("packed sources: {}", p.display()));
            Some(p)
        }
        Err(e) => {
            crate::runner::runner_term::out_warn("RUNNER",
                &format!("packing skipped/failed: {}", e));
            None
        }
    };

    // 3) Autogit push (auch wenn Pack scheitert — Build war OK)
    let mut autogit_ok = false;
    let msg = if let Some(z) = &zip_path {
        format!("chore: branch build + pack ({})", z.file_name().and_then(OsStr::to_str).unwrap_or("zip"))
    } else {
        "chore: branch build".to_string()
    };
    if commands::autogit::run(root, Some(msg), false, "origin", Some(&br), true).unwrap_or(1) == 0 {
        autogit_ok = true;
    }

    // 4) Abschluss-Summary
    let end_ms = epoch_ms();
    let artifact = find_artifact(root);
    let mut notes = Vec::new();
    if let Some(z) = &zip_path {
        notes.push(format!("sources_zip={}", z.display()));
    }

    summary::print_end_summary(summary::EndSummary {
        success: true,
        exit_code: 0,
        started_ms: start_ms,
        elapsed_ms: end_ms.saturating_sub(start_ms),
        artifact_path: artifact.as_ref().map(|p| p.to_string_lossy().to_string()),
        commit_short: crate::vcs::git_short_hash(root),
        commit_branch: Some(format!("origin/{}", br)),
        autogit_pushed: autogit_ok,
        notes,
    });

    if let Some(art) = artifact.as_deref() {
        print_magenta_run_hint(root, art);
    }

    0
}
