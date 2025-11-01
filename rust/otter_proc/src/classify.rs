///// Otter: Heuristics for MSVC/CMake/Ninja/CUDA output - compact, robust.
// //// Schneefuchs: ASCII-insensitive (lowercase); localized “Fehler/Warnung” included; fast path trims.
///// Maus: Single enum Severity; pure function classify_line(&str) -> Severity.
///// Datei: rust/otter_proc/src/classify.rs

#[derive(Copy, Clone, Debug)]
pub enum Severity {
    Info,
    Warn,
    Err,
}

pub fn classify_line(line: &str) -> Severity {
    let l = line.to_lowercase();
    let t = l.trim_start();

    // ----- errors -----
    if t.starts_with("ninja: build stopped:")
        || l.contains("fatal error")
        || l.contains("cmake error")
        || l.contains("nvcc fatal")
        || l.contains("ptxas fatal")
        || l.contains(" error c")       // cl.exe: "error Cxxxx"
        || (l.contains("lnk") && l.contains("error"))
        || (l.contains("msb") && l.contains("error"))
        || l.contains(" fehler")        // localized
        || l.contains(" error:")
    {
        return Severity::Err;
    }

    // ----- warnings -----
    if l.contains("cmake warning")
        || l.contains(" warning c")     // cl.exe: "warning Cxxxx"
        || (l.contains("lnk") && l.contains("warning"))
        || (l.contains("msb") && l.contains("warning"))
        || l.contains(" warnung")       // localized
    {
        return Severity::Warn;
    }

    Severity::Info
}
