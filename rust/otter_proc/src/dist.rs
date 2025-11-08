///// Otter: Dist-Sync — kopiert EXE + whitelisted DLLs nach /dist (copy-if-different), klare [DIST]-Logs.
/// /// Schneefuchs: Ohne Extra-Dependencies; robustes FS-Handling; env-Override OTTER_DIST_DLLS (csv).
/// /// Maus: Windows-first (".dll"); ignoriert System-DLLs; /WX-sicher dank #[allow(dead_code)] bis Runner-Hook.
///// Datei: rust/otter_proc/src/dist.rs

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::runner_term::{out_err, out_info, out_warn};

#[cfg(windows)]
const DLL_EXT: &str = "dll";

#[cfg(not(windows))]
const DLL_EXT: &str = "so"; // Platzhalter; Projektziel ist Windows, bleibt hier neutral.

/// Schneller Gleichheits-Check: vergleicht Existenz, Größe, mtime (aufgerundet).
fn same_file_quick(src: &Path, dst: &Path) -> io::Result<bool> {
    let src_md = fs::metadata(src)?;
    let dst_md = match fs::metadata(dst) {
        Ok(m) => m,
        Err(_) => return Ok(false),
    };
    if src_md.len() != dst_md.len() {
        return Ok(false);
    }
    let sm = src_md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let dm = dst_md.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    Ok(sm.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        == dm
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs())
}

/// Copy mit „copy-if-different“-Semantik. Gibt true zurück, wenn tatsächlich kopiert wurde.
fn copy_if_different(src: &Path, dst: &Path) -> io::Result<bool> {
    if same_file_quick(src, dst).unwrap_or(false) {
        return Ok(false);
    }
    if let Some(p) = dst.parent() {
        fs::create_dir_all(p)?;
    }
    fs::copy(src, dst)?;
    Ok(true)
}

/// Ermittelt die /dist-Directory unterhalb des Projekt-Roots.
fn dist_dir(root: &Path) -> PathBuf {
    root.join("dist")
}

/// Default-Whitelist der zu spiegelnden DLLs. Kann via OTTER_DIST_DLLS überschrieben/ergänzt werden.
/// Beispiel: OTTER_DIST_DLLS="glew32.dll,glfw3.dll,openvr_api.dll"
fn dll_whitelist() -> Vec<String> {
    let mut list = vec!["glew32.dll".to_string(), "glfw3.dll".to_string()];
    if let Ok(extra) = std::env::var("OTTER_DIST_DLLS") {
        for part in extra.split(',') {
            let n = part.trim();
            if !n.is_empty() {
                list.push(n.to_string());
            }
        }
    }
    // deduplizieren (einfach)
    list.sort_unstable();
    list.dedup();
    list
}

/// Sucht im EXE-Verzeichnis nach whitelisted DLLs.
fn find_whitelisted_dlls(exe_dir: &Path, names_lower: &[String]) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let rd = match fs::read_dir(exe_dir) {
        Ok(r) => r,
        Err(e) => {
            out_warn("DIST", &format!("read_dir failed dir={} err={}", exe_dir.display(), e));
            return Ok(out);
        }
    };
    'entries: for ent in rd {
        if let Ok(e) = ent {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case(DLL_EXT)) != Some(true) {
                continue;
            }
            let fname = match p.file_name().and_then(|o| o.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let fname_l = fname.to_ascii_lowercase();
            for want in names_lower {
                if &fname_l == want {
                    out.push(p);
                    continue 'entries;
                }
            }
        }
    }
    Ok(out)
}

/// Öffentliche Hauptfunktion: spiegelt EXE + DLLs nach <root>/dist.
/// Rückgabewert: Anzahl aktualisierter Dateien.
#[allow(dead_code)]
pub fn sync_dist(project_root: &Path, exe_path: &Path) -> io::Result<usize> {
    let mut updated: usize = 0;

    if !exe_path.exists() {
        out_err("DIST", &format!("exe not found: {}", exe_path.display()));
        return Err(io::Error::new(io::ErrorKind::NotFound, "exe not found"));
    }

    let exe_name = exe_path
        .file_name()
        .and_then(|o| o.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "invalid exe file name"))?
        .to_string();

    let exe_dir = exe_path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "exe has no parent"))?;

    let dist = dist_dir(project_root);
    fs::create_dir_all(&dist)?;

    out_info("DIST", &format!("root={}", project_root.display()));
    out_info("DIST", &format!("dist={}", dist.display()));
    out_info("DIST", &format!("exe={}", exe_path.display()));

    // 1) EXE kopieren
    let dst_exe = dist.join(&exe_name);
    match copy_if_different(exe_path, &dst_exe) {
        Ok(true) => {
            updated += 1;
            out_info("DIST", &format!("exe -> {}", dst_exe.display()));
        }
        Ok(false) => {
            out_info("DIST", &format!("exe up-to-date ({})", dst_exe.display()));
        }
        Err(e) => {
            out_err("DIST", &format!("copy exe failed: {}", e));
            return Err(e);
        }
    }

    // 2) Whitelisted DLLs kopieren (aus EXE-Verzeichnis)
    let wanted = dll_whitelist()
        .into_iter()
        .map(|s| s.to_ascii_lowercase())
        .collect::<Vec<_>>();

    let dlls = find_whitelisted_dlls(exe_dir, &wanted)?;
    if dlls.is_empty() {
        out_warn("DIST", "no whitelisted DLLs found beside the exe");
    }

    for dll in dlls {
        let fname = dll.file_name().and_then(|o| o.to_str()).unwrap_or("?.dll");
        let dst = dist.join(fname);
        match copy_if_different(&dll, &dst) {
            Ok(true) => {
                updated += 1;
                out_info("DIST", &format!("dll {} -> {}", fname, dst.display()));
            }
            Ok(false) => {
                out_info("DIST", &format!("dll {} up-to-date", fname));
            }
            Err(e) => {
                out_warn("DIST", &format!("dll copy failed {}: {}", fname, e));
            }
        }
    }

    // 3) Optional: Hinweis, wenn das dist-EXE älter wirkt als das Build (Heuristik)
    if let (Ok(src_md), Ok(dst_md)) = (fs::metadata(exe_path), fs::metadata(&dst_exe)) {
        if let (Ok(sm), Ok(dm)) = (src_md.modified(), dst_md.modified()) {
            if sm > dm {
                out_warn(
                    "DIST",
                    &format!(
                        "dist exe seems older than src ({} < {})",
                        humantime::format_rfc3339(dm),
                        humantime::format_rfc3339(sm)
                    ),
                );
            }
        }
    }

    out_info("DIST", &format!("updated={} (exe+dlls)", updated));
    Ok(updated)
}
