///// Otter: Build-Pipeline für probe/configure/build/stage – CMake-Aufrufe mit PATH/ENV-Overlay.
/// ///// Schneefuchs: Saubere Fehlerweitergabe (PipelineError), ASCII-Logs, vorsichtiger Hard-Clean mit Cache-Heimatprüfung.
/// ///// Maus: Minimal mut; keine toten Imports; deterministisch & kompakt – Warning „unused_mut“ beseitigt.
///// Datei: rust/otter_proc/src/pipeline.rs

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

use crate::toolchain::{configure_args_and_env, find_cl_exe, find_windows_sdk_bin_x64};
use crate::utils::{canonical, ensure_dir, hard_clean_build_dir_safe, same_path};

pub type PResult<T> = Result<T, PipelineError>;

#[derive(Debug)]
pub struct PipelineError {
    pub msg: String,
    pub code: Option<i32>,
}
impl PipelineError {
    fn new(msg: impl Into<String>) -> Self { Self { msg: msg.into(), code: None } }
    fn with_code(msg: impl Into<String>, code: i32) -> Self { Self { msg: msg.into(), code: Some(code) } }
}
impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.msg) }
}
impl std::error::Error for PipelineError {}
impl From<io::Error> for PipelineError {
    fn from(e: io::Error) -> Self { PipelineError::new(format!("io error: {e}")) }
}

// --- simple log helper (ASCII) -----------------------------------------------

fn logln(w: &mut File, line: &str) -> io::Result<()> {
    let ms = epoch_ms();
    writeln!(w, "[ts{ms}] {line}")?;
    w.flush()
}
fn epoch_ms() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
}

// --- phases ------------------------------------------------------------------

pub fn run_probe(log: &mut File) -> PResult<()> {
    logln(log, "[PASS] probe")?;

    let ok = Command::new("cmake").arg("--version").status()
        .map_err(|e| PipelineError::new(format!("cmake not runnable: {e}")))?;
    if !ok.success() { return Err(PipelineError::with_code("cmake --version failed", 11)); }

    if let Some(cl) = find_cl_exe() {
        logln(log, &format!("[PROBE] cl={}", cl.display()))?;
    } else {
        logln(log, "[WARN] could not pre-detect cl.exe; will try default PATH")?;
    }
    if let Some(sdk) = find_windows_sdk_bin_x64() {
        logln(log, &format!("[PROBE] win_sdk_bin={}", sdk.display()))?;
    } else {
        logln(log, "[WARN] Windows SDK bin (x64) not found; rc/mt may be missing")?;
    }

    Ok(())
}

pub fn run_configure(root: &Path, build_rel: &str, log: &mut File) -> PResult<()> {
    logln(log, "[PASS] configure")?;

    let build_dir = root.join(build_rel);
    ensure_cache_ownership(root, &build_dir, build_rel, log)?;

    // Args/Env vorbereiten (CMAKE_CUDA_HOST_COMPILER, CMAKE_RC_COMPILER, CMAKE_MT, CUDAHOSTCXX, PATH prepend)
    let (args, prepend_paths, extra_env) = configure_args_and_env();

    let status = run_in_env(root, "cmake", &args, &prepend_paths, &extra_env, log)?;
    if status.success() {
        return Ok(());
    }

    // Fallback: harter, sicherer Clean + erneuter Configure
    logln(log, "[INFO] configure failed — attempting hard clean/regenerate")?;
    match hard_clean_build_dir_safe(root, &build_dir, build_rel) {
        Ok(removed) => {
            for p in removed {
                logln(log, &format!("[CLEAN] removed {}", p.display()))?;
            }
        }
        Err(e) => {
            return Err(PipelineError::new(format!("hard_clean_build_dir_safe failed: {e}")));
        }
    }
    ensure_dir(&build_dir);

    let status2 = run_in_env(root, "cmake", &args, &prepend_paths, &extra_env, log)?;
    ensure_ok(status2, "cmake --preset windows-msvc")
}

pub fn run_build(root: &Path, exe_path: &Path, _build_rel: &str, log: &mut File) -> PResult<()> {
    logln(log, "[PASS] build")?;

    // Für den Build reicht CUDAHOSTCXX (PATH-Prepend ist i. d. R. nach Configure nicht mehr nötig).
    let mut args: Vec<String> = vec!["--build".into(), "--preset".into(), "windows-build".into(), "-j".into()];
    let prepend_paths: Vec<PathBuf> = Vec::new(); // <— nicht mut: Warning beseitigt
    let mut extra_env: Vec<(String, String)> = Vec::new();

    if let Some(cl) = find_cl_exe() {
        let cl_norm = cl.to_string_lossy().replace('\\', "/");
        extra_env.push(("CUDAHOSTCXX".to_string(), cl_norm));
    }

    let status = run_in_env(root, "cmake", &args.iter().map(|s| s.as_str()).collect::<Vec<_>>(), &prepend_paths, &extra_env, log)?;
    ensure_ok(status, "cmake --build --preset windows-build -j")?;

    if !exe_path.exists() {
        return Err(PipelineError::with_code(format!("expected exe not found: {}", exe_path.display()), 14));
    }
    Ok(())
}

pub fn run_stage(root: &Path, exe_path: &Path, dist_rel: &str, log: &mut File) -> PResult<()> {
    logln(log, "[PASS] stage")?;

    if !exe_path.exists() {
        return Err(PipelineError::with_code(format!("exe missing before stage: {}", exe_path.display()), 15));
    }

    let dist = root.join(dist_rel);
    fs::create_dir_all(&dist).map_err(|e| PipelineError::new(format!("cannot create dist dir {}: {e}", dist.display())))?;

    // EXE kopieren
    let exe_name = exe_path.file_name().ok_or_else(|| PipelineError::new(format!("no file_name for {}", exe_path.display())))?;
    let exe_dst = dist.join(exe_name);
    fs::copy(exe_path, &exe_dst).map_err(|e| PipelineError::new(format!("copy {} -> {} failed: {e}", exe_path.display(), exe_dst.display())))?;

    // DLLs aus dem Build-Ordner (nahe der EXE) kopieren
    if let Some(build_dir) = exe_path.parent() {
        if build_dir.is_dir() {
            if let Ok(read_it) = fs::read_dir(build_dir) {
                for ent in read_it.flatten() {
                    let p = ent.path();
                    if p.extension().and_then(|s| s.to_str()).map(|e| e.eq_ignore_ascii_case("dll")).unwrap_or(false) {
                        if let Some(name) = p.file_name() {
                            let dst = dist.join(name);
                            let _ = fs::copy(&p, &dst);
                        }
                    }
                }
            }
        }
    }

    logln(log, &format!("[STAGE] -> {}", dist.display()))?;
    Ok(())
}

// --- helpers -----------------------------------------------------------------

fn run_in_env(
    cwd: &Path,
    exe: &str,
    args: &[&str],
    prepend_paths: &[PathBuf],
    extra_env: &[(String, String)],
    log: &mut File,
) -> PResult<ExitStatus> {
    logln(log, &format!("[CMD] {} {}", exe, args.join(" ")))?;

    let mut cmd = Command::new(exe);
    cmd.args(args).current_dir(cwd);

    // PATH-Overlay (prepend)
    if !prepend_paths.is_empty() {
        let old_path = std::env::var_os("PATH").unwrap_or_default();
        let mut new_path = std::ffi::OsString::new();
        for p in prepend_paths {
            if !new_path.is_empty() { new_path.push(";"); }
            new_path.push(p.as_os_str());
        }
        if !old_path.is_empty() {
            if !new_path.is_empty() { new_path.push(";"); }
            new_path.push(old_path);
        }
        cmd.env("PATH", new_path);
    }

    // zusätzliche ENV-Variablen (z. B. CUDAHOSTCXX)
    for (k, v) in extra_env {
        cmd.env(k, v);
    }

    cmd.status().map_err(|e| PipelineError::new(format!("spawn {exe} failed: {e}")))
}

fn ensure_ok(status: ExitStatus, label: &str) -> PResult<()> {
    if status.success() { Ok(()) } else {
        Err(PipelineError::with_code(format!("{label} failed (code={:?})", status.code()), status.code().unwrap_or(1)))
    }
}

/// Löscht den Build-Ordner bei Projektwechsel (CMAKE_HOME_DIRECTORY ungleich aktuellem Root).
/// Safety: löscht nur, wenn `build_dir == root.join(build_rel)`.
fn ensure_cache_ownership(root: &Path, build_dir: &Path, build_rel: &str, log: &mut File) -> PResult<()> {
    let cache = build_dir.join("CMakeCache.txt");
    if !cache.exists() { return Ok(()); }

    let home = read_cache_home_dir(&cache)?;
    if home.is_none() { return Ok(()); }

    let root_c = canonical(root);
    let home_c = canonical(&home.unwrap());
    if same_path(&root_c, &home_c) {
        return Ok(());
    }

    // Safety-Gurt
    let expected = root.join(build_rel);
    let expected_c = canonical(&expected);
    let build_c = canonical(build_dir);
    if !same_path(&build_c, &expected_c) {
        return Err(PipelineError::with_code(
            format!("refusing to delete unexpected build dir: {}", build_c.display()),
            10,
        ));
    }

    logln(log, &format!("[CLEAN] removing stale build dir: {}", build_dir.display()))?;
    fs::remove_dir_all(build_dir)
        .map_err(|e| PipelineError::new(format!("remove_dir_all {} failed: {e}", build_dir.display())))?;
    Ok(())
}

fn read_cache_home_dir(cache: &Path) -> PResult<Option<PathBuf>> {
    let f = File::open(cache).map_err(|e| PipelineError::new(format!("open {} failed: {e}", cache.display())))?;
    let r = BufReader::new(f);
    for line in r.lines() {
        let l = line.map_err(|e| PipelineError::new(format!("read {} failed: {e}", cache.display())))?;
        if l.starts_with("CMAKE_HOME_DIRECTORY:") {
            if let Some(idx) = l.find('=') {
                let val = l[(idx + 1)..].trim();
                if !val.is_empty() { return Ok(Some(PathBuf::from(val))); }
            }
        }
    }
    Ok(None)
}
