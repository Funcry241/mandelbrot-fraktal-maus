///// Otter: Branch-Orchestrator – Build → ZIP-Pack (Rust) → Autogit Push → Magenta-Run-Hint.
///// Schneefuchs: Kein PowerShell; stabile Pfade; ASCII-only; Backslashes für Windows-Ausgabe.
///// Maus: Saubere Endzusammenfassung + hübscher Pink-Hinweis; Default-Branch „wupp“.
///// Datei: rust/otter_proc/src/ops_branch.rs

use std::ffi::OsStr;
use std::io;
use std::path::{Path, PathBuf};

use crate::artifact::find_artifact;
use crate::commands;
use crate::runner::runner_term;
use crate::summary;
use crate::utils::epoch_ms;

/// Umgebungsvariable lesen (String) mit Default.
fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// Umgebungsvariable als u32 (optional) parsen.
fn env_opt_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|s| s.parse::<u32>().ok())
}

/// Aktueller Branch aus Git oder OTTER_BRANCH oder "wupp".
fn current_branch_or_default(root: &Path) -> String {
    crate::vcs::git_current_branch(root)
        .or_else(|| std::env::var("OTTER_BRANCH").ok())
        .unwrap_or_else(|| "wupp".to_string())
}

/// Windows-freundlicher Pfadstring (Backslashes). Bleibt ASCII.
fn win_path_str(p: &Path) -> String {
    p.to_string_lossy().replace('/', "\\")
}

/// Am Ende einen gut lesbaren, pinken (magenta) Ausführungs-Hinweis drucken.
/// Zeigt optional Schritt (1) zum Ergänzen des DLL-Pfads, falls vcpkg-bin existiert.
fn print_magenta_run_hint(root: &Path, artifact: &Path) {
    let root_abs = match root.canonicalize() {
        Ok(p) => p,
        Err(_) => root.to_path_buf(),
    };
    let art_abs = match artifact.canonicalize() {
        Ok(p) => p,
        Err(_) => artifact.to_path_buf(),
    };
    let root_s = win_path_str(&root_abs);
    let art_s  = win_path_str(&art_abs);

    let vcpkg_bin_s = format!(r"{}\vcpkg_installed\x64-windows\bin", root_s);
    let vcpkg_bin_exists = Path::new(&vcpkg_bin_s).exists();

    // ANSI: helle Magenta (Pink). ASCII-Rahmen.
    println!("\x1b[95m+------------------------------------------------------------------+");
    println!(  "\x1b[95m| RUN NOW                                                          |");
    println!(  "\x1b[95m+------------------------------------------------------------------+");
    if vcpkg_bin_exists {
        println!("\x1b[95m| 1) Add DLL path for this session:                                |");
        println!("\x1b[95m|    $env:Path=\"{};$env:Path\" |", vcpkg_bin_s);
        println!("\x1b[95m|                                                                  |");
        println!("\x1b[95m| 2) Launch the app:                                               |");
        println!("\x1b[95m|    \"{}\" |", art_s);
    } else {
        println!("\x1b[95m| Launch the app:                                                  |");
        println!("\x1b[95m|    \"{}\" |", art_s);
    }
    println!(  "\x1b[95m+------------------------------------------------------------------+\x1b[0m");
}

/// Öffentlicher Einstieg für den Branch-Pfad.
/// Ablauf: Build (Full) → Pack (Rust/ZIP) → Autogit push → Summary (+Run-Hint).
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

    // (1) Full Build
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
        let artifact = find_artifact(root);
        summary::print_end_summary(summary::EndSummary {
            success: false,
            exit_code: code_build,
            started_ms: start_ms,
            elapsed_ms: end_ms.saturating_sub(start_ms),
            artifact_path: artifact.map(|p| p.to_string_lossy().to_string()),
            commit_short: crate::vcs::git_short_hash(root),
            commit_branch: Some(format!("origin/{}", br)),
            autogit_pushed: false,
            notes: vec!["build failed before packing".into()],
        });
        return code_build;
    }

    // (2) Quellen packen (reines Rust; kein PowerShell)
    let zip_path = match crate::commands::pack::run(root, None, false) {
        Ok(p) => {
            crate::runner::runner_term::out_info("RUNNER", &format!("packed sources: {}", p.display()));
            Some(p)
        }
        Err(e) => {
            crate::runner::runner_term::out_warn("RUNNER", &format!("packing skipped/failed: {}", e));
            None
        }
    };

    // (3) Autogit push (auch wenn Pack scheitert — Build war OK)
    let mut autogit_ok = false;
    let msg = if let Some(z) = &zip_path {
        format!(
            "chore: branch build + pack ({})",
            z.file_name().and_then(OsStr::to_str).unwrap_or("zip")
        )
    } else {
        "chore: branch build".to_string()
    };
    if commands::autogit::run(root, Some(msg), false, "origin", Some(&br), true).unwrap_or(1) == 0 {
        autogit_ok = true;
    }

    // (4) Abschluss-Summary
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

    // (5) Hübscher Magenta-Hinweis mit exakten Backslashes
    if let Some(art) = artifact.as_ref() {
        print_magenta_run_hint(root, art);
    }

    0
}
