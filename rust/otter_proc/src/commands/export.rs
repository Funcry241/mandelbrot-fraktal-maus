///// Otter: Export – erzeugt immer ein ZIP aus out/.
///// Schneefuchs: Dateiname enthält OP und STATUS; Self-exclude (robust via canonicalize); ASCII-Logs; E0716-Fix.
///// Maus: Pruning pro Sorte (OP+STATUS), max_keep je Sorte; kompatibel zu anyhow::Result.
/// Datei: rust/otter_proc/src/commands/export.rs

use std::{
    fs,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use walkdir::WalkDir;
use zip::{write::FileOptions, CompressionMethod, ZipWriter};

fn epoch_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn zip_dir(src_dir: &Path, zip_path: &Path) -> Result<()> {
    let src_dir = src_dir.canonicalize()?;
    let zip_file = File::create(zip_path)?;
    let mut zip = ZipWriter::new(zip_file);
    let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

    // für robusten Self-Exclude: ZIP-Pfad ebenfalls kanonisch
    let zip_cmp = zip_path
        .canonicalize()
        .unwrap_or_else(|_| zip_path.to_path_buf());

    for entry in WalkDir::new(&src_dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // ZIP selbst ausschließen
        if path == zip_cmp {
            continue;
        }

        let rel = path.strip_prefix(&src_dir).unwrap();
        let name = rel.to_string_lossy().replace('\\', "/"); // portable

        if entry.file_type().is_dir() {
            if !name.is_empty() {
                zip.add_directory(name, options)?;
            }
            continue;
        }

        let mut f = File::open(path)?;
        zip.start_file(name, options)?;
        let mut buf = Vec::with_capacity(64 * 1024);
        f.read_to_end(&mut buf)?;
        zip.write_all(&buf)?;
    }

    zip.finish()?;
    Ok(())
}

fn prune_old_zips(out_dir: &Path, op: &str, status: &str, keep: usize) -> Result<()> {
    // Sorte = OP + STATUS  ->  z.B.  branch_ok_*.zip, branch_fail_*.zip
    let prefix = format!("{}_{}_", op, status);
    let mut zips: Vec<(PathBuf, std::time::SystemTime)> = fs::read_dir(out_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            // E0716-Fix: OsString binden, dann lossy-String ableiten
            let name_os = e.file_name();
            let n = name_os.to_string_lossy();
            e.path().is_file() && n.starts_with(&prefix) && n.ends_with(".zip")
        })
        .filter_map(|e| {
            let mtime = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((e.path(), mtime))
        })
        .collect();

    zips.sort_by_key(|(_, t)| std::cmp::Reverse(*t));
    if zips.len() > keep {
        for (p, _) in zips.into_iter().skip(keep) {
            let _ = fs::remove_file(p);
        }
    }
    Ok(())
}

/// Erzeugt ein ZIP **unter `out/exports/`**, das den **Inhalt von `out/`** packt.
/// Rückgabe: Pfad zur erzeugten ZIP-Datei.
pub fn run(root: &Path, out_dir: Option<&Path>, max_keep: usize, dry_run: bool) -> Result<PathBuf> {
    // Zielverzeichnis für ZIPs (Default: out/exports)
    let exports_dir = out_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| root.join("out").join("exports"));
    fs::create_dir_all(&exports_dir)?;

    // Quelle: out/
    let src_out = root.join("out");
    fs::create_dir_all(&src_out)?;

    let op = std::env::var("OTTER_OP").unwrap_or_else(|_| "export".to_string());
    let status = std::env::var("OTTER_STATUS").unwrap_or_else(|_| "any".to_string());
    let ts = epoch_ms();

    // Unterschiedliche Namen pro Sorte: <op>_<status>_<epoch>.zip
    let zip_name = format!("{}_{}_{}.zip", op, status, ts);
    let zip_path = exports_dir.join(&zip_name);

    if dry_run {
        println!("[Otter/export] DRY-RUN would write: {}", zip_path.display());
        return Ok(zip_path);
    }

    // Falls out/ leer ist, Marker erzeugen, damit es überhaupt etwas zu packen gibt
    if fs::read_dir(&src_out)?.next().is_none() {
        let _ = fs::write(src_out.join(".otter.empty"), b"");
    }

    zip_dir(&src_out, &zip_path)?;
    println!("[Otter/export] wrote {}", zip_path.display());

    if max_keep > 0 {
        // WICHTIG: prune pro Sorte (op+status) **im exports-dir**
        let _ = prune_old_zips(&exports_dir, &op, &status, max_keep);
    }

    Ok(zip_path)
}
