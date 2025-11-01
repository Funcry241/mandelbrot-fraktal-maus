///// Otter: Windows-Buildfahrt (VsDev/vcvars-Kette + Fallback), CMake with --preset.
///// Schneefuchs: Übergibt --parallel N an "cmake --build"; prüft Artifakt-Pfad(e) und loggt Status.
///// Maus: ASCII-Logs; klare Pfad-Kandidaten für mandelbrot_otterdream.exe; kein unnötiger Noise.
///// Datei: rust/otter_proc/src/commands/winenv.rs

use std::io;
use std::path::{Path, PathBuf};

use crate::prockit;

/// Führt "cmake --preset <configure_preset> -D CMAKE_BUILD_TYPE=<cfg>" im gegebenen Arbeitsverzeichnis aus.
fn run_configure(project_root: &Path, configure_preset: &str, build_cfg: &str) -> io::Result<()> {
    let args: Vec<String> = vec![
        "--preset".into(),
        configure_preset.into(),
        "-D".into(),
        format!("CMAKE_BUILD_TYPE={}", build_cfg),
    ];
    let st = prockit::run_stream_status("cmake", &args, Some(project_root))?;
    if !st.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "cmake configure failed"));
    }
    Ok(())
}

/// Führt "cmake --build --preset <build_preset> --config <cfg> [--parallel N]" aus.
fn run_build(project_root: &Path, build_preset: &str, build_cfg: &str, parallel: Option<u32>) -> io::Result<()> {
    let mut args: Vec<String> = vec![
        "--build".into(),
        "--preset".into(),
        build_preset.into(),
        "--config".into(),
        build_cfg.into(),
    ];
    if let Some(n) = parallel {
        args.push("--parallel".into());
        args.push(n.to_string());
    }
    let st = prockit::run_stream_status("cmake", &args, Some(project_root))?;
    if !st.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "cmake build failed"));
    }
    Ok(())
}

/// Kandidaten für das erzeugte Exe (Multi-/Single-Config, übliche Layouts).
fn artifact_candidates(project_root: &Path, build_cfg: &str) -> Vec<PathBuf> {
    let exe = "mandelbrot_otterdream.exe";
    let b = project_root.join("build");
    vec![
        b.join(build_cfg).join(exe),          // Ninja Multi-Config
        b.join("bin").join(build_cfg).join(exe),
        b.join("bin").join(exe),
        b.join(exe),                           // Single-Config Ninja/Makefiles
    ]
}

/// Loggt Kandidaten & meldet finalen Fund (falls vorhanden).
fn report_artifact_status(project_root: &Path, build_cfg: &str) {
    let mut found: Option<PathBuf> = None;
    for p in artifact_candidates(project_root, build_cfg) {
        let exists = p.exists();
        println!("[RUNNER] artifact-candidate: {} exists={}", p.display(), if exists { "yes" } else { "no" });
        if exists && found.is_none() {
            found = Some(p);
        }
    }
    if let Some(ok) = found {
        println!("[RUNNER] artifact: {}", ok.display());
    } else {
        println!("[RUNNER][WARN] build finished but no artifact found (check presets/targets).");
    }
}

/// Ruft ein VS-Umgebungsskript (VsDevCmd/vcvars*) auf und kettet dann "where cl" + cmake configure/build an.
fn try_vsdev_like(
    project_root: &Path,
    script_path: &Path,
    extra_args: &[&str],
    configure_preset: &str,
    build_preset: &str,
    build_cfg: &str,
    parallel: Option<u32>,
) -> io::Result<()> {
    println!(
        "[RUNNER] env-script({})={}",
        script_path.file_stem().and_then(|s| s.to_str()).unwrap_or("vsdev"),
        script_path.display()
    );

    // cmd /C call "<script>" <extra> && where cl && cmake --preset ... && cmake --build ...
    let mut chain: Vec<String> = vec![
        "/C".into(),
        "call".into(),
        script_path.to_string_lossy().into_owned(),
    ];
    chain.extend(extra_args.iter().map(|s| s.to_string()));
    chain.push("&&".into());
    chain.push("where".into());
    chain.push("cl".into());
    chain.push("&&".into());
    chain.push("cmake".into());
    chain.push("--preset".into());
    chain.push(configure_preset.into());
    chain.push("-D".into());
    chain.push(format!("CMAKE_BUILD_TYPE={}", build_cfg));
    chain.push("&&".into());
    chain.push("cmake".into());
    chain.push("--build".into());
    chain.push("--preset".into());
    chain.push(build_preset.into());
    chain.push("--config".into());
    chain.push(build_cfg.into());
    if let Some(n) = parallel {
        chain.push("--parallel".into());
        chain.push(n.to_string());
    }

    let st = prockit::run_stream_status("cmd", &chain, Some(project_root))?;
    if !st.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "VS dev chain failed"));
    }
    Ok(())
}

/// Öffentliche Orchestrierung für den Windows-Build:
/// 1) VsDevCmd -> 2) vcvars64 -> 3) vcvarsall x64 -> 4) Direkt (Fallback)
pub fn run_cmake_windows(
    project_root: &Path,
    configure_preset: &str,
    build_preset: &str,
    build_cfg: &str,
    parallel: Option<u32>,
) -> io::Result<i32> {
    // 1) VsDevCmd
    let vsdev = Path::new(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat");
    if vsdev.exists() {
        match try_vsdev_like(project_root, vsdev, &["-arch=x64"], configure_preset, build_preset, build_cfg, parallel) {
            Ok(_) => {
                report_artifact_status(project_root, build_cfg);
                return Ok(0);
            }
            Err(e) => {
                println!("[RUNNER][WARN] vsdev chain failed (exit!=0) -> trying next…");
                println!("{}", e);
            }
        }
    }

    // 2) vcvars64
    let vcvars64 = Path::new(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat");
    if vcvars64.exists() {
        match try_vsdev_like(project_root, vcvars64, &[], configure_preset, build_preset, build_cfg, parallel) {
            Ok(_) => {
                report_artifact_status(project_root, build_cfg);
                return Ok(0);
            }
            Err(e) => {
                println!("[RUNNER][WARN] vcvars64 chain failed (exit!=0) -> trying next…");
                println!("{}", e);
            }
        }
    }

    // 3) vcvarsall x64
    let vcvarsall = Path::new(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat");
    if vcvarsall.exists() {
        match try_vsdev_like(project_root, vcvarsall, &["x64"], configure_preset, build_preset, build_cfg, parallel) {
            Ok(_) => {
                report_artifact_status(project_root, build_cfg);
                return Ok(0);
            }
            Err(e) => {
                println!("[RUNNER][WARN] vcvarsall chain failed (exit!=0) -> trying fallback…");
                println!("{}", e);
            }
        }
    }

    // 4) Fallback: direkter Aufruf ohne Dev-Bat (kann scheitern, wenn cl/nvcc nicht im PATH sind)
    println!("[RUNNER][WARN] VsDev/vcvars chain exhausted. Switching to direct-env fallback…");
    run_configure(project_root, configure_preset, build_cfg)?;
    run_build(project_root, build_preset, build_cfg, parallel)?;
    report_artifact_status(project_root, build_cfg);
    Ok(0)
}
