///// Otter: Toolchain-Finder & CMake/NVCC-Argumente – cl.exe/Windows SDK (rc/mt) + Host-Compiler-Setup.
/// ///// Schneefuchs: Keine externen Crates; Pfade CMake-sicher (forward slashes); Spaces sauber gequotet.
/// ///// Maus: Rückgabewerte für Pipeline nutzbar: (args, prepend_PATH, extra_env) für cmake.
/// ///// Datei: rust/otter_proc/src/toolchain.rs
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::utils::{quote_if_space, strip_quotes, to_cmake_path};

/// Ermittelt Argumente & Umgebungsvariablen für `cmake --preset windows-msvc`,
/// sodass nvcc den Host-Compiler (cl.exe) und Windows SDK-Tools (rc.exe/mt.exe) findet.
pub fn configure_args_and_env() -> (Vec<String>, Vec<PathBuf>, Vec<(String, String)>) {
    let mut args: Vec<String> = vec![
        "--preset".into(),
        "windows-msvc".into(),
        "-D".into(),
        "CMAKE_EXPORT_COMPILE_COMMANDS=ON".into(),
    ];
    let mut prepend_paths: Vec<PathBuf> = Vec::new();
    let mut extra_env: Vec<(String, String)> = Vec::new();

    if let Some(cl) = find_cl_exe() {
        let cl_dir = cl.parent().unwrap().to_path_buf();
        let cl_val = quote_if_space(&to_cmake_path(&cl));
        prepend_paths.push(cl_dir);
        // Für CMake (nvcc host compiler)
        args.push("-D".into());
        args.push(format!("CMAKE_CUDA_HOST_COMPILER={}", cl_val));
        // Für nvcc direkt
        extra_env.push(("CUDAHOSTCXX".into(), strip_quotes(&cl_val).to_string()));
    }

    if let Some(sdk) = find_windows_sdk_bin_x64() {
        prepend_paths.push(sdk.clone());
        let rc = quote_if_space(&to_cmake_path(&sdk.join("rc.exe")));
        let mt = quote_if_space(&to_cmake_path(&sdk.join("mt.exe")));
        args.push("-D".into());
        args.push(format!("CMAKE_RC_COMPILER={}", rc));
        args.push("-D".into());
        args.push(format!("CMAKE_MT={}", mt));
    }

    (args, prepend_paths, extra_env)
}

/// Sucht cl.exe (VS2022) über mehrere Strategien (VCINSTALLDIR, Standardpfade, vswhere).
pub fn find_cl_exe() -> Option<PathBuf> {
    if let Some(v) = env::var_os("VCINSTALLDIR") {
        let vc = PathBuf::from(v);
        if let Some(p) = probe_msvc_bin(&vc) {
            return Some(p);
        }
    }
    for base in &[
        r"C:\Program Files\Microsoft Visual Studio\2022",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022",
    ] {
        for edition in &["BuildTools", "Community", "Professional", "Enterprise"] {
            let root = Path::new(base).join(edition).join(r"VC\Tools\MSVC");
            if let Some(p) = newest_msvc_cl(&root) {
                return Some(p);
            }
        }
    }
    if let Some(vswhere) = vswhere_path() {
        let out = Command::new(vswhere)
            .args(&[
                "-latest",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ])
            .output()
            .ok()?;
        if out.status.success() {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !path.is_empty() {
                let root = Path::new(&path).join(r"VC\Tools\MSVC");
                if let Some(p) = newest_msvc_cl(&root) {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Findet Windows SDK `bin\*\x64` (rc.exe/mt.exe).
pub fn find_windows_sdk_bin_x64() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("WindowsSdkDir") {
        if let Some(p) = newest_sdk_bin_x64(&PathBuf::from(dir).join("bin")) {
            return Some(p);
        }
    }
    if let Some(pf86) = env::var_os("ProgramFiles(x86)") {
        if let Some(p) = newest_sdk_bin_x64(&PathBuf::from(pf86).join(r"Windows Kits\10\bin")) {
            return Some(p);
        }
    }
    if let Some(pf) = env::var_os("ProgramFiles") {
        if let Some(p) = newest_sdk_bin_x64(&PathBuf::from(pf).join(r"Windows Kits\10\bin")) {
            return Some(p);
        }
    }
    None
}

// ---- interne Helfer ---------------------------------------------------------

fn vswhere_path() -> Option<PathBuf> {
    for base in &["ProgramFiles(x86)", "ProgramFiles"] {
        if let Some(pf) = env::var_os(base) {
            let p = PathBuf::from(pf).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

fn probe_msvc_bin(vc_install_dir: &Path) -> Option<PathBuf> {
    newest_msvc_cl(&vc_install_dir.join("Tools").join("MSVC"))
}

fn newest_msvc_cl(msvc_tools_dir: &Path) -> Option<PathBuf> {
    if !msvc_tools_dir.is_dir() {
        return None;
    }
    let mut best: Option<PathBuf> = None;
    for e in fs::read_dir(msvc_tools_dir).ok()? {
        let e = e.ok()?;
        if !e.file_type().ok()?.is_dir() {
            continue;
        }
        let cand = e.path().join(r"bin\Hostx64\x64\cl.exe");
        if cand.exists() {
            best = match best {
                Some(cur) if cand <= cur => Some(cur),
                _ => Some(cand),
            };
        }
    }
    best
}

fn newest_sdk_bin_x64(base: &Path) -> Option<PathBuf> {
    if !base.is_dir() {
        return None;
    }
    let mut best: Option<PathBuf> = None;
    for e in fs::read_dir(base).ok()? {
        let e = e.ok()?;
        if !e.file_type().ok()?.is_dir() {
            continue;
        }
        let cand = e.path().join("x64");
        if cand.join("rc.exe").exists() && cand.join("mt.exe").exists() {
            best = match best {
                Some(cur) if cand <= cur => Some(cur),
                _ => Some(cand),
            };
        }
    }
    best
}
