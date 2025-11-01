// Otter: VS-DevCmd-Erkennung & Env-Übernahme (cmd /c "call VsDevCmd && set").
// Schneefuchs: vswhere nutzen, dann Default-Pfad Fallback; Pfade der Tools separat prüfen.
// Maus: Klares Ergebnisobjekt; Fehlermeldungen als ASCII; keine externen Crates.
// Datei: rust/otter_proc/src/vsenv.rs

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::runner::{out_err, out_info};

#[derive(Clone)]
pub struct VsEnv {
    pub env: HashMap<String,String>,
    pub tool_cmake: PathBuf,
    pub tool_ninja: PathBuf,
    pub tool_cl: PathBuf,
    pub tool_nvcc: PathBuf,
}

fn where_exe(name: &str) -> Option<PathBuf> {
    let out = Command::new("where").arg(name).output().ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        let p = PathBuf::from(line.trim());
        if p.exists() { return Some(p); }
    }
    None
}

fn guess_vsdevcmd() -> Option<PathBuf> {
    // 1) via vswhere
    if let Some(vswhere) = where_exe("vswhere.exe") {
        let out = Command::new(vswhere)
            .args(&[
                "-latest", "-prerelease", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath"
            ])
            .output().ok()?;
        if out.status.success() {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !path.is_empty() {
                let p = PathBuf::from(path).join("Common7\\Tools\\VsDevCmd.bat");
                if p.exists() { return Some(p); }
            }
        }
    }
    // 2) Default-Pfad (Community 2022)
    let def = PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat");
    if def.exists() { return Some(def); }
    None
}

fn import_env_from_vsdevcmd(vsdev: &Path) -> Option<HashMap<String,String>> {
    // cmd.exe /c "<VsDevCmd.bat> -arch=x64 -host_arch=x64 && set"
    let cmdline = format!("\"{}\" -arch=x64 -host_arch=x64 && set", vsdev.display());
    let out = Command::new("cmd.exe").args(&["/c", &cmdline]).output().ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut map = HashMap::new();
    for line in s.lines() {
        if let Some(pos) = line.find('=') {
            let k = &line[..pos];
            let v = &line[pos+1..];
            if !k.is_empty() { map.insert(k.to_string(), v.to_string()); }
        }
    }
    Some(map)
}

pub fn probe_vsdevcmd_and_env() -> Result<VsEnv, String> {
    let vsdev = guess_vsdevcmd().ok_or_else(|| "VsDevCmd.bat not found".to_string())?;
    out_info("VS", &format!("Importing env from: {}", vsdev.display()));

    let envmap = import_env_from_vsdevcmd(&vsdev).ok_or_else(|| "failed to import VS environment".to_string())?;

    // Tools lokalisieren (unter importierter Env arbeiten)
    let cmake = where_exe("cmake.exe").ok_or_else(|| "cmake.exe not found".to_string())?;
    let ninja = where_exe("ninja.exe").ok_or_else(|| "ninja.exe not found".to_string())?;
    let cl    = where_exe("cl.exe").ok_or_else(|| "cl.exe not found".to_string())?;
    let nvcc  = where_exe("nvcc.exe").ok_or_else(|| "nvcc.exe not found".to_string())?;

    Ok(VsEnv { env: envmap, tool_cmake: cmake, tool_ninja: ninja, tool_cl: cl, tool_nvcc: nvcc })
}
