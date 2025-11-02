///// Otter: ASCII-Endblock – kompakte, deterministische Abschlusszusammenfassung.
///// Schneefuchs: Feste Struktur; keine Abhängigkeiten; nur benötigte Formatierungen.
///// Maus: Minimalistisch, klar; keine Farben/Unicode.
///// Datei: rust/otter_proc/src/summary.rs

pub struct EndSummary {
    pub success: bool,
    pub exit_code: i32,
    pub started_ms: u128,
    pub elapsed_ms: u128,
    pub artifact_path: Option<String>,
    pub commit_short: Option<String>,
    pub commit_branch: Option<String>,
    pub autogit_pushed: bool,
    pub notes: Vec<String>,
}

fn fmt_secs(ms: u128) -> String {
    // eine Nachkommastelle, ASCII
    let secs = ms as f64 / 1000.0;
    format!("{:.1} s", secs)
}

pub fn print_end_summary(s: EndSummary) {
    let status = if s.success { "SUCCESS" } else { "FAIL" };
    let artifact = s
        .artifact_path
        .as_deref()
        .unwrap_or("(not found)");

    let commit = s.commit_short.as_deref().unwrap_or("n/a");
    let branch = s.commit_branch.as_deref().unwrap_or("n/a");
    let push = if s.autogit_pushed { "OK" } else { "ERROR" };

    println!("[END] +--------------------------------------------------------------+");
    println!("[END] | BUILD SUMMARY: {} (code={})                              |", status, s.exit_code);
    println!("[END] +--------------------------------------------------------------+");
    println!("[END] | Artifact : {}", artifact);
    println!("[END] | Started  : ts_ms={}", s.started_ms);
    println!("[END] | Elapsed  : ~{}", fmt_secs(s.elapsed_ms));
    println!("[END] | Commit   : {} -> {} (push={})", commit, branch, push);
    println!("[END] +--------------------------------------------------------------+");

    // „GUT-Sachen“ geordnet, kurz und lesbar
    let artifact_exists = if s.artifact_path.is_some() { "yes" } else { "no" };
    println!("[GOOD] Artifact exists: {}", artifact_exists);
    println!("[GOOD] Autogit: {}", if s.autogit_pushed { "push OK" } else { "push skipped/failed" });

    for note in s.notes {
        println!("[INFO]  {}", note);
    }
}
