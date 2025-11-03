///// Otter: Git-Utilities – Status, Branch-Existenz, Checkout (quiet) für den Runner.
///// Schneefuchs: Nur std::process::Command; ASCII-stabil; keine Seiteneffekte außer Git.
///// Maus: pub(crate); minimal, deterministisch; -q für leise Checkouts.
///// Datei: rust/otter_proc/src/vcs.rs

use std::path::Path;
use std::process::Command;

pub(crate) fn git_short_hash(root: &Path) -> Option<String> {
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

pub(crate) fn git_current_branch(root: &Path) -> Option<String> {
    let root_s = root.to_str()?;
    let out = Command::new("git")
        .args(["-C", root_s, "rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s == "HEAD" || s.is_empty() { None } else { Some(s) }
}

pub(crate) fn git_is_repo(root: &Path) -> bool {
    let root_s = match root.to_str() { Some(s) => s, None => return false };
    Command::new("git")
        .args(["-C", root_s, "rev-parse", "--is-inside-work-tree"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

pub(crate) fn git_local_branch_exists(root: &Path, name: &str) -> bool {
    let root_s = match root.to_str() { Some(s) => s, None => return false };
    Command::new("git")
        .args(["-C", root_s, "show-ref", "--verify", "--quiet", &format!("refs/heads/{}", name)])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub(crate) fn git_remote_branch_exists(root: &Path, remote: &str, name: &str) -> bool {
    let root_s = match root.to_str() { Some(s) => s, None => return false };
    Command::new("git")
        .args(["-C", root_s, "ls-remote", "--exit-code", "--heads", remote, name])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

pub(crate) fn git_checkout_new_local(root: &Path, name: &str) -> anyhow::Result<()> {
    let root_s = root.to_str().ok_or_else(|| anyhow::anyhow!("bad path"))?;
    let st = Command::new("git")
        .args(["-C", root_s, "checkout", "-q", "-b", name])
        .status()?;
    if !st.success() {
        anyhow::bail!("git checkout -b {} failed", name);
    }
    Ok(())
}

pub(crate) fn git_checkout_from_remote(root: &Path, remote: &str, name: &str) -> anyhow::Result<()> {
    let root_s = root.to_str().ok_or_else(|| anyhow::anyhow!("bad path"))?;
    let st = Command::new("git")
        .args(["-C", root_s, "checkout", "-q", "-b", name, &format!("{}/{}", remote, name)])
        .status()?;
    if !st.success() {
        anyhow::bail!("git checkout -b {} {}/{} failed", name, remote, name);
    }
    Ok(())
}

pub(crate) fn git_checkout_existing(root: &Path, name: &str) -> anyhow::Result<()> {
    let root_s = root.to_str().ok_or_else(|| anyhow::anyhow!("bad path"))?;
    let st = Command::new("git")
        .args(["-C", root_s, "checkout", "-q", name])
        .status()?;
    if !st.success() {
        anyhow::bail!("git checkout {} failed", name);
    }
    Ok(())
}
