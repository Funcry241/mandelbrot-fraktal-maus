///// Otter: Simple git automation (add/commit/push) with optional HTTPS fallback.
///// Schneefuchs: ASCII-only logs; no secrets; robust exit codes (0=OK, 1=issues). CRLF-Warnungen pro Aufruf unterdrückt via -c core.safecrlf=false & -c core.autocrlf=input.
///// Maus: Accepts optional commit message; falls back to "chore: update"; Branch-Autodetect.
///// Datei: rust/otter_proc/src/commands/autogit.rs

use std::io;
use std::path::Path;
use std::process::{Command, Stdio};

fn run_cmd_in(root: &Path, program: &str, args: &[&str]) -> io::Result<i32> {
    // Für git-Befehle per-Aufruf Konfigs setzen, um CRLF→LF-Warnungen zu vermeiden.
    let is_git = program == "git";
    let mut full_args: Vec<&str> = Vec::new();
    if is_git {
        full_args.extend_from_slice(&[
            "-c", "core.safecrlf=false",
            "-c", "core.autocrlf=input",
        ]);
    }
    full_args.extend_from_slice(args);

    println!("[AUTOGIT][RUN] {} {}", program, full_args.join(" "));
    let status = Command::new(program)
        .args(&full_args)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    Ok(status.code().unwrap_or(1))
}

fn git_exists() -> bool {
    Command::new("git")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ensure_repo(root: &Path) -> io::Result<()> {
    let dotgit = root.join(".git");
    if !dotgit.exists() {
        println!("[AUTOGIT] no .git found — init new repo");
        let code = run_cmd_in(root, "git", &["init"])?;
        if code != 0 {
            return Err(io::Error::new(io::ErrorKind::Other, "git init failed"));
        }
    }
    Ok(())
}

fn current_branch(root: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if name.is_empty() || name == "HEAD" { None } else { Some(name) }
}

fn ssh_to_https(url: &str) -> Option<String> {
    // Convert URLs like git@github.com:owner/repo.git -> https://github.com/owner/repo.git
    if let Some(rest) = url.strip_prefix("git@github.com:") {
        return Some(format!("https://github.com/{}", rest));
    }
    None
}

/// Add/commit/push with optional HTTPS fallback if SSH push fails.
/// Returns 0 on success; 1 if there were errors.
pub fn run(
    root: &Path,
    message: Option<String>,
    allow_empty: bool,
    remote: &str,
    branch: Option<&str>,
    auto_https_fallback: bool,
) -> io::Result<i32> {
    let commit_msg = message.unwrap_or_else(|| "chore: update".to_string());

    println!(
        "[AUTOGIT] start root={} msg=\"{}\" allow_empty={} remote={} branch={:?} https_fallback={}",
        root.display(),
        commit_msg,
        allow_empty,
        remote,
        branch,
        auto_https_fallback
    );

    if !git_exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "git not found on PATH",
        ));
    }
    ensure_repo(root)?;

    // Always add -A, then commit.
    let mut errs = 0usize;

    if run_cmd_in(root, "git", &["add", "-A"])? != 0 {
        println!("[AUTOGIT][ERR] git add failed");
        errs += 1;
    }

    // Build commit args.
    let mut commit_args = vec!["commit", "-m", &commit_msg];
    if allow_empty {
        commit_args.push("--allow-empty");
    }
    let code_commit = run_cmd_in(root, "git", &commit_args)?;
    if code_commit != 0 {
        // Common benign case: nothing to commit -> exit code 1. Treat as non-fatal information.
        println!(
            "[AUTOGIT][WARN] git commit returned code {} (possibly nothing to commit)",
            code_commit
        );
    }

    // Determine branch (explicit or auto).
    let branch_owned = branch.map(|s| s.to_string()).or_else(|| current_branch(root));

    // Optional push.
    if !remote.is_empty() {
        if let Some(br_name) = branch_owned.as_deref() {
            let mut pushed = false;

            let code_push = run_cmd_in(root, "git", &["push", "-u", remote, br_name])?;
            if code_push == 0 {
                pushed = true;
            } else if auto_https_fallback {
                // Try to detect SSH remote and translate to HTTPS.
                println!("[AUTOGIT][WARN] initial push failed; trying HTTPS fallback…");
                let output = Command::new("git")
                    .args(["remote", "get-url", remote])
                    .current_dir(root)
                    .output();

                if let Ok(out) = output {
                    if out.status.success() {
                        let old = String::from_utf8_lossy(&out.stdout).trim().to_string();
                        if let Some(https_url) = ssh_to_https(&old) {
                            println!("[AUTOGIT] set-url {} -> {}", remote, https_url);
                            let su =
                                run_cmd_in(root, "git", &["remote", "set-url", remote, &https_url])?;
                            if su == 0 {
                                let code_push2 =
                                    run_cmd_in(root, "git", &["push", "-u", remote, br_name])?;
                                pushed = code_push2 == 0;
                            }
                        } else {
                            println!("[AUTOGIT][WARN] remote is not SSH github.com; skip fallback");
                        }
                    } else {
                        println!("[AUTOGIT][WARN] git remote get-url failed");
                    }
                } else {
                    println!("[AUTOGIT][WARN] failed to query remote URL for fallback");
                }
            }

            if !pushed {
                println!("[AUTOGIT][ERR] push did not succeed");
                errs += 1;
            }
        } else {
            println!("[AUTOGIT][WARN] could not determine current branch; skip push");
        }
    }

    println!(
        "[AUTOGIT] done status={}",
        if errs == 0 { "OK" } else { "WITH_ERRORS" }
    );
    Ok(if errs == 0 { 0 } else { 1 })
}
