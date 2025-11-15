///// Otter: Windows-Buildfahrt (VsDev/vcvars-Kette + Fallback) + DIST-Pack nach erfolgreichem Build.
///// Schneefuchs: Zentrales Logging via runner_term; farbiges yes/no; kein Doppel-Pack in PS (Rust-only).
///// Maus: ASCII-Logs; klare Artefakt-Kandidaten; OTTER_PACK=0 zum Deaktivieren; robustes Overwrite in dist\ (inkl. LICENSE).
///// Datei: rust/otter_proc/src/commands/winenv.rs
#![deny(warnings)]

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::artifact::fmt_exists;
use crate::build_metrics::BuildMetrics;
use crate::runner;

// ------------------------------ Metrics-Helfer --------------------------------

fn record_phase_ms(metrics: &mut BuildMetrics, root: &Path, sig: &str, phase: &str, ms: u128) {
    metrics.upsert_phase_ms(sig, phase, ms);
    let _ = metrics.save(root);
}

// --------------------------- Direkter CMake-Aufruf ----------------------------

fn run_configure_direct(project_root: &Path, configure_preset: &str, build_cfg: &str) -> io::Result<()> {
    let args: Vec<String> = vec![
        "--preset".into(),
        configure_preset.into(),
        "-D".into(),
        format!("CMAKE_BUILD_TYPE={}", build_cfg),
    ];
    let st = runner::run_streamed_with_env("cmake", &args, None, Some(project_root));
    if st.code != 0 {
        return Err(io::Error::new(io::ErrorKind::Other, "cmake configure failed"));
    }
    Ok(())
}

fn run_build_direct(project_root: &Path, build_preset: &str, build_cfg: &str, parallel: Option<u32>) -> io::Result<()> {
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
    let st = runner::run_streamed_with_env("cmake", &args, None, Some(project_root));
    if st.code != 0 {
        return Err(io::Error::new(io::ErrorKind::Other, "cmake build failed"));
    }
    Ok(())
}

// ---------------------------- VS-Skripte: Ketten ------------------------------

fn run_with_script_configure(
    project_root: &Path,
    script_path: &Path,
    extra_args: &[&str],
    configure_preset: &str,
    build_cfg: &str,
) -> io::Result<()> {
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

    let st = runner::run_streamed_with_env("cmd", &chain, None, Some(project_root));
    if st.code != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "VS dev configure chain failed",
        ));
    }
    Ok(())
}

fn run_with_script_build(
    project_root: &Path,
    script_path: &Path,
    extra_args: &[&str],
    build_preset: &str,
    build_cfg: &str,
    parallel: Option<u32>,
) -> io::Result<()> {
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
    chain.push("--build".into());
    chain.push("--preset".into());
    chain.push(build_preset.into());
    chain.push("--config".into());
    chain.push(build_cfg.into());
    if let Some(n) = parallel {
        chain.push("--parallel".into());
        chain.push(n.to_string());
    }

    let st = runner::run_streamed_with_env("cmd", &chain, None, Some(project_root));
    if st.code != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "VS dev build chain failed",
        ));
    }
    Ok(())
}

// --------------------------- Artefakt-Erkennung -------------------------------

fn artifact_candidates(project_root: &Path, build_cfg: &str) -> Vec<PathBuf> {
    let exe = "mandelbrot_otterdream.exe";
    let b = project_root.join("build");
    vec![
        b.join(build_cfg).join(exe),             // Ninja Multi-Config
        b.join("bin").join(build_cfg).join(exe), // gängige Layouts
        b.join("bin").join(exe),
        b.join(exe),                             // Single-Config
    ]
}

fn report_artifact_status(project_root: &Path, build_cfg: &str) -> Option<PathBuf> {
    let mut found: Option<PathBuf> = None;
    for p in artifact_candidates(project_root, build_cfg) {
        let exists = p.is_file();
        runner::runner_term::out_info(
            "RUNNER",
            &format!("artifact-candidate: {} exists={}", p.display(), fmt_exists(exists)),
        );
        if exists && found.is_none() {
            found = Some(p);
        }
    }
    if let Some(ok) = &found {
        runner::runner_term::out_info("RUNNER", &format!("artifact: {}", ok.display()));
    } else {
        runner::runner_term::out_info(
            "RUNNER",
            "[WARN] build finished but no artifact found (check presets/targets).",
        );
    }
    found
}

// ------------------------------ DIST-Pack (Rust) ------------------------------

fn pack_enabled() -> bool {
    match env::var("OTTER_PACK") {
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !(t == "0" || t == "off" || t == "no")
        }
        Err(_) => true, // Default: an
    }
}

fn maybe_pack_dist(project_root: &Path, build_cfg: &str) {
    if !pack_enabled() {
        runner::runner_term::out_info("PACK", "disabled by OTTER_PACK=0");
        return;
    }

    let Some(artifact) = report_artifact_status(project_root, build_cfg) else {
        runner::runner_term::out_info("PACK", "skip: no artifact to copy");
        return;
    };

    let dist = project_root.join("dist");
    if let Err(e) = fs::create_dir_all(&dist) {
        runner::runner_term::out_err("PACK", &format!("create dist failed: {}", e));
        return;
    }

    let dst = dist.join("mandelbrot_otterdream.exe");
    match fs::copy(&artifact, &dst) {
        Ok(_) => runner::runner_term::out_info("PACK", &format!("updated: {}", dst.display())),
        Err(e) => runner::runner_term::out_err(
            "PACK",
            &format!("copy failed {} -> {}: {}", artifact.display(), dst.display(), e),
        ),
    }

    // LICENSE in dist\ beilegen, damit das Binary-Bundle rechtlich eigenständig ist.
    let license_src = project_root.join("LICENSE");
    let license_dst = dist.join("LICENSE");
    if license_src.is_file() {
        match fs::copy(&license_src, &license_dst) {
            Ok(_) => runner::runner_term::out_info(
                "PACK",
                &format!("LICENSE updated: {}", license_dst.display()),
            ),
            Err(e) => runner::runner_term::out_err(
                "PACK",
                &format!(
                    "copy LICENSE failed {} -> {}: {}",
                    license_src.display(),
                    license_dst.display(),
                    e
                ),
            ),
        }
    } else {
        runner::runner_term::out_info(
            "PACK",
            "LICENSE not found in project root; skipping copy",
        );
    }
}

// ------------------------------ Öffentlicher Lauf -----------------------------

/// 1) VsDevCmd → 2) vcvars64 → 3) vcvarsall x64 → 4) Direkter Fallback — jeweils inkl. DIST-Pack
pub fn run_cmake_windows(
    project_root: &Path,
    configure_preset: &str,
    build_preset: &str,
    build_cfg: &str,
    parallel: Option<u32>,
) -> io::Result<i32> {
    let (mut metrics, _path, _seed) = BuildMetrics::load_or_seed(project_root);
    let t_total = Instant::now();

    // 1) VsDevCmd
    let vsdev = Path::new(
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    );
    if vsdev.exists() {
        runner::runner_term::out_info("ENV", &format!("script(vsdev)={}", vsdev.display()));

        let t0 = Instant::now();
        let conf_res =
            run_with_script_configure(project_root, vsdev, &["-arch=x64"], configure_preset, build_cfg);
        let dt_conf = t0.elapsed().as_millis();
        if conf_res.is_ok() {
            record_phase_ms(
                &mut metrics,
                project_root,
                "cmake:configure",
                "configure",
                dt_conf,
            );
        }

        let t1 = Instant::now();
        let build_res = match conf_res {
            Ok(_) => run_with_script_build(
                project_root,
                vsdev,
                &["-arch=x64"],
                build_preset,
                build_cfg,
                parallel,
            ),
            Err(e) => Err(e),
        };
        let dt_build = t1.elapsed().as_millis();
        if build_res.is_ok() {
            record_phase_ms(&mut metrics, project_root, "cmd:proc", "build", dt_build);
            record_phase_ms(
                &mut metrics,
                project_root,
                "cmd:proc",
                "proc",
                t_total.elapsed().as_millis(),
            );
            maybe_pack_dist(project_root, build_cfg);
            return Ok(0);
        } else {
            runner::runner_term::out_info(
                "RUNNER",
                "[WARN] vsdev chain failed (exit!=0) -> trying next…",
            );
        }
    }

    // 2) vcvars64
    let vcvars64 = Path::new(
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    );
    if vcvars64.exists() {
        runner::runner_term::out_info("ENV", &format!("script(vcvars64)={}", vcvars64.display()));

        let t0 = Instant::now();
        let conf_res =
            run_with_script_configure(project_root, vcvars64, &[], configure_preset, build_cfg);
        let dt_conf = t0.elapsed().as_millis();
        if conf_res.is_ok() {
            record_phase_ms(
                &mut metrics,
                project_root,
                "cmake:configure",
                "configure",
                dt_conf,
            );
        }

        let t1 = Instant::now();
        let build_res = match conf_res {
            Ok(_) => run_with_script_build(
                project_root,
                vcvars64,
                &[],
                build_preset,
                build_cfg,
                parallel,
            ),
            Err(e) => Err(e),
        };
        let dt_build = t1.elapsed().as_millis();
        if build_res.is_ok() {
            record_phase_ms(&mut metrics, project_root, "cmd:proc", "build", dt_build);
            record_phase_ms(
                &mut metrics,
                project_root,
                "cmd:proc",
                "proc",
                t_total.elapsed().as_millis(),
            );
            maybe_pack_dist(project_root, build_cfg);
            return Ok(0);
        } else {
            runner::runner_term::out_info(
                "RUNNER",
                "[WARN] vcvars64 chain failed (exit!=0) -> trying next…",
            );
        }
    }

    // 3) vcvarsall x64
    let vcvarsall = Path::new(
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
    );
    if vcvarsall.exists() {
        runner::runner_term::out_info(
            "ENV",
            &format!("script(vcvarsall x64)={}", vcvarsall.display()),
        );

        let t0 = Instant::now();
        let conf_res =
            run_with_script_configure(project_root, vcvarsall, &["x64"], configure_preset, build_cfg);
        let dt_conf = t0.elapsed().as_millis();
        if conf_res.is_ok() {
            record_phase_ms(
                &mut metrics,
                project_root,
                "cmake:configure",
                "configure",
                dt_conf,
            );
        }

        let t1 = Instant::now();
        let build_res = match conf_res {
            Ok(_) => run_with_script_build(
                project_root,
                vcvarsall,
                &["x64"],
                build_preset,
                build_cfg,
                parallel,
            ),
            Err(e) => Err(e),
        };
        let dt_build = t1.elapsed().as_millis();
        if build_res.is_ok() {
            record_phase_ms(&mut metrics, project_root, "cmd:proc", "build", dt_build);
            record_phase_ms(
                &mut metrics,
                project_root,
                "cmd:proc",
                "proc",
                t_total.elapsed().as_millis(),
            );
            maybe_pack_dist(project_root, build_cfg);
            return Ok(0);
        } else {
            runner::runner_term::out_info(
                "RUNNER",
                "[WARN] vcvarsall chain failed (exit!=0) -> trying fallback…",
            );
        }
    }

    // 4) Direkter Fallback (ohne Dev-Bat)
    runner::runner_term::out_info(
        "RUNNER",
        "[WARN] VsDev/vcvars chain exhausted. Switching to direct-env fallback…",
    );

    let t0 = Instant::now();
    run_configure_direct(project_root, configure_preset, build_cfg)?;
    let dt_conf = t0.elapsed().as_millis();
    record_phase_ms(
        &mut metrics,
        project_root,
        "cmake:configure",
        "configure",
        dt_conf,
    );

    let t1 = Instant::now();
    run_build_direct(project_root, build_preset, build_cfg, parallel)?;
    let dt_build = t1.elapsed().as_millis();
    record_phase_ms(&mut metrics, project_root, "cmd:proc", "build", dt_build);
    record_phase_ms(
        &mut metrics,
        project_root,
        "cmd:proc",
        "proc",
        t_total.elapsed().as_millis(),
    );

    maybe_pack_dist(project_root, build_cfg);
    Ok(0)
}
