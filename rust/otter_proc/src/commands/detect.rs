///// Otter: Toolchain-Detektion (cl.exe, Windows SDK) – optionale Helpers, ASCII-only.
///// Schneefuchs: Getrennt, keine Seiteneffekte; darf ungenutzt bleiben (dead_code erlaubt).
///// Maus: Minimal, robust gegen fehlende Pfade; kein panic.
///// Datei: rust/otter_proc/src/commands/detect.rs

use std::fs;
use std::path::{Path, PathBuf};

/// Versucht `cl.exe` unter einem typischen VS2022-Pfad zu finden.
/// Rückgabe: Pfad zu `cl.exe` oder `None`.
#[allow(dead_code)]
pub fn detect_cl_exe() -> Option<PathBuf> {
    let base = Path::new(
        r#"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"#,
    );
    if !base.exists() {
        return None;
    }
    // Nimm die lexikographisch höchste Versions-Untermappe.
    let mut best: Option<PathBuf> = None;
    if let Ok(rd) = fs::read_dir(base) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if best.as_ref().map_or(true, |b| p.file_name() > b.file_name()) {
                    best = Some(p);
                }
            }
        }
    }
    let Some(msvc_ver) = best else { return None; };
    let cl = msvc_ver
        .join("bin")
        .join("Hostx64")
        .join("x64")
        .join("cl.exe");
    cl.exists().then_some(cl)
}

/// Sucht das Windows 10/11 SDK (Include/Lib Version) und liefert `(bin_x64, version, base)`.
#[allow(dead_code)]
pub fn find_windows_sdk() -> Option<(PathBuf, String, PathBuf)> {
    let base = Path::new(r#"C:\Program Files (x86)\Windows Kits\10"#);
    if !base.exists() {
        return None;
    }
    let lib = base.join("Lib");
    let inc = base.join("Include");

    // Höchste Version unter Lib/ und Include/ wählen, die in beiden existiert.
    let vers = fs::read_dir(&lib)
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|s| s.starts_with("10.0."))
        .collect::<Vec<_>>();

    let mut best: Option<String> = None;
    for v in vers {
        let has_inc = inc.join(&v).exists();
        let has_lib = lib.join(&v).exists();
        if has_inc && has_lib {
            if best.as_ref().map_or(true, |b| &v > b) {
                best = Some(v);
            }
        }
    }
    let ver = best?;
    let bin_x64 = base.join("bin").join(&ver).join("x64");
    Some((bin_x64, ver, base.to_path_buf()))
}
