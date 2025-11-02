///// Otter: Simple git automation (add/commit/push) with optional HTTPS fallback + upstream/branch ensure.
///// Schneefuchs: ASCII-only logs; no secrets; robust exit codes (0=OK, 1=issues). CRLF-Warnungen je Call unterdrückt.
///// Maus: Autodetect current branch; falls keiner → „wupp“; legt Branch bei Bedarf lokal an, setzt Upstream und pusht.
///// Datei: rust/otter_proc/src/commands/autogit.rs

use std::io;
use std::path::Path;
use std::process::{Command, Stdio};

fn run_cmd_in(root: &Path, program: &str, args: &[&str]) -> io::Result<i32> {
    // Für git-Befehle je Aufruf Konfigs setzen, um CRLF→LF-Warnungen zu vermeiden.
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

fn local_branch_exists(root: &Path, name: &str) -> bool {
    Command::new("git")
        .args(["show-ref", "--verify", &format!("refs/heads/{}", name)])
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn remote_branch_exists(root: &Path, remote: &str, name: &str) -> bool {
    Command::new("git")
        .args(["ls-remote", "--exit-code", "--heads", remote, name])
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn checkout_or_create_branch(root: &Path, name: &str, remote: &str) -> io::Result<()> {
    if local_branch_exists(root, name) {
        let rc = run_cmd_in(root, "git", &["checkout", name])?;
        if rc != 0 { return Err(io::Error::new(io::ErrorKind::Other, "git checkout failed")); }
        return Ok(());
    }

    // Falls remote-Branch existiert, daraus erstellen/tracken; sonst von HEAD neu erstellen.
    if remote_branch_exists(root, remote, name) {
        let rc = run_cmd_in(root, "git", &["checkout", "-b", name, &format!("{}/{}", remote, name)])?;
        if rc != 0 { return Err(io::Error::new(io::ErrorKind::Other, "git checkout -b from remote failed")); }
    } else {
        let rc = run_cmd_in(root, "git", &["checkout", "-b", name])?;
        if rc != 0 { return Err(io::Error::new(io::ErrorKind::Other, "git checkout -b failed")); }
    }
    Ok(())
}

fn has_upstream(root: &Path, branch: &str) -> bool {
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "--symbolic-full-name", &format!("{}@{{u}}", branch)])
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ssh_to_https(url: &str) -> Option<String> {
    // Convert URLs like git@github.com:owner/repo.git -> https://github.com/owner/repo.git
    if let Some(rest) = url.strip_prefix("git@github.com:") {
        return Some(format!("https://github.com/{}", rest));
    }
    None
}

fn remote_get_url(root: &Path, remote: &str) -> Option<String> {
    Command::new("git")
        .args(["remote", "get-url", remote])
        .current_dir(root)
        .output()
        .ok()
        .and_then(|o| if o.status.success() {
            Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
        } else { None })
}

/// Add/commit/push with optional HTTPS fallback, ensuring local branch, upstream and push.
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

    // Branch ableiten: explizit > aktuell > "wupp" (Default für Branch-Modus/Detached HEAD)
    let target_branch = branch.map(|s| s.to_string())
        .or_else(|| current_branch(root))
        .unwrap_or_else(|| "wupp".to_string());

    println!(
        "[AUTOGIT] start root={} msg=\"{}\" allow_empty={} remote={} branch={} https_fallback={}",
        root.display(), commit_msg, allow_empty, remote, target_branch, auto_https_fallback
    );

    if !git_exists() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "git not found on PATH"));
    }
    ensure_repo(root)?;

    // 1) Sicherstellen, dass der Ziel-Branch ausgecheckt ist (lokal neu anlegen falls nötig)
    if current_branch(root).as_deref() != Some(&*target_branch) {
        checkout_or_create_branch(root, &target_branch, remote)?;
    }

    // 2) Stage + Commit (commit-Fehler „nichts zu committen“ ist ok)
    let mut errs = 0usize;

    if run_cmd_in(root, "git", &["add", "-A"])? != 0 {
        println!("[AUTOGIT][ERR] git add failed");
        errs += 1;
    }

    let mut commit_args = vec!["commit", "-m", &commit_msg];
    if allow_empty {
        commit_args.push("--allow-empty");
    }
    let code_commit = run_cmd_in(root, "git", &commit_args)?;
    if code_commit != 0 {
        println!("[AUTOGIT][INFO] git commit returned code {} (possibly nothing to commit)", code_commit);
    }

    // 3) Push (Upstream setzen, falls noch keiner existiert)
    let mut pushed = false;
    let mut tried_https = false;

    let do_push = |r: &str, br: &str| -> io::Result<i32> {
        // Wenn Upstream fehlt → mit -u pushen, sonst normal pushen
        if has_upstream(root, br) {
            run_cmd_in(root, "git", &["push", r, br])
        } else {
            run_cmd_in(root, "git", &["push", "-u", r, br])
        }
    };

    let code_push = do_push(remote, &target_branch)?;
    if code_push == 0 {
        pushed = true;
    } else if auto_https_fallback {
        println!("[AUTOGIT][WARN] initial push failed; trying HTTPS fallback…");
        if let Some(old) = remote_get_url(root, remote) {
            if let Some(https_url) = ssh_to_https(&old) {
                let su = run_cmd_in(root, "git", &["remote", "set-url", remote, &https_url])?;
                if su == 0 {
                    tried_https = true;
                    let code_push2 = do_push(remote, &target_branch)?;
                    pushed = code_push2 == 0;
                } else {
                    println!("[AUTOGIT][WARN] failed to set remote URL to HTTPS");
                }
            } else {
                println!("[AUTOGIT][WARN] remote is not SSH github.com; skip fallback");
            }
        } else {
            println!("[AUTOGIT][WARN] failed to query remote URL for fallback");
        }
    }

    if !pushed {
        println!("[AUTOGIT][ERR] push did not succeed (https_fallback_tried={})", tried_https);
        errs += 1;
    }

    println!("[AUTOGIT] done status={}", if errs == 0 { "OK" } else { "WITH_ERRORS" });
    Ok(if errs == 0 { 0 } else { 1 })
}
