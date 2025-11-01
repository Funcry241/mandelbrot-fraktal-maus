///// Otter: Progress state & pretty renderer (spinner, %, ETA, seeded by metrics).
///// Schneefuchs: Keine externen Crates; 200 ms Rate-Limit; robuste Parser für % und [n/m].
///// Maus: Standardmäßig aktiv (OTTER_PROGRESS=0 zum Abschalten); ASCII-Bar, term_cols() aus runner_term.
///// Datei: rust/otter_proc/src/runner_progress.rs

use std::env;
use std::time::{Duration, Instant};

use super::runner_term;
use super::runner_term::term_cols;

/// Interner Zustand für die Live-Anzeige.
pub struct ProgressState {
    pub start: Instant,
    pub last_tick: Instant,
    pub spinner_ix: usize,
    pub runtime_phase: String,       // "proc" | "configure" | "build"
    pub best_builder_pct: Option<f32>,
    pub last_snippet: String,
}

impl ProgressState {
    pub fn new(phase: &str) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_tick: now
                .checked_sub(Duration::from_millis(201))
                .unwrap_or(now),
            spinner_ix: 0,
            runtime_phase: phase.to_string(),
            best_builder_pct: None,
            last_snippet: String::new(),
        }
    }
}

/// Default: Progress an. Mit Umgebungsvariable OTTER_PROGRESS=0 kann man es abschalten.
pub fn progress_enabled() -> bool {
    match env::var("OTTER_PROGRESS") {
        Ok(v) if v.trim() == "0" => false,
        _ => true,
    }
}

/// Soll die nächste Animationszeile raus? (200 ms Takt)
pub fn due(p: &ProgressState) -> bool {
    p.last_tick.elapsed() >= Duration::from_millis(200)
}

/// Kürzt auf das rechte Ende (meist ist dort der interessante Teil).
pub fn last_nonempty_snippet(s: &str, limit: usize) -> String {
    let t = s.trim();
    if t.is_empty() {
        return String::new();
    }
    if t.len() <= limit {
        return t.to_string();
    }
    // rechte Seite behalten
    let take = limit.min(t.len());
    t[t.len() - take..].to_string()
}

/// Findet eine Prozentzahl wie "68%" oder " 68 % ".
pub fn parse_percent(s: &str) -> Option<f32> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            // rückwärts nach Ziffern + optionalem Punkt/Leerzeichen
            let mut j = i;
            // überspringe Leerzeichen links vom '%'
            while j > 0 && (bytes[j - 1] as char).is_whitespace() {
                j -= 1;
            }
            // sammle Ziffern und .
            let mut k = j;
            while k > 0 {
                let c = bytes[k - 1] as char;
                if c.is_ascii_digit() || c == '.' || c == ',' || c == ' ' {
                    k -= 1;
                } else {
                    break;
                }
            }
            let num = s[k..j].trim().replace(',', ".");
            if let Ok(v) = num.parse::<f32>() {
                if (0.0..=100.0).contains(&v) {
                    return Some(v / 100.0);
                }
            }
        }
        i += 1;
    }
    None
}

/// Findet Muster wie "[17/45]" (CMake, Ninja, MSBuild-ähnliche Ratio-Zeilen).
pub fn parse_ratio_percent(s: &str) -> Option<f32> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            // parse n
            let mut j = i + 1;
            let mut n: u32 = 0;
            let mut has_n = false;
            while j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                has_n = true;
                n = n.saturating_mul(10).saturating_add((bytes[j] - b'0') as u32);
                j += 1;
            }
            // slash
            if has_n && j < bytes.len() && bytes[j] == b'/' {
                j += 1;
            } else {
                i += 1;
                continue;
            }
            // parse m
            let mut m: u32 = 0;
            let mut has_m = false;
            while j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                has_m = true;
                m = m.saturating_mul(10).saturating_add((bytes[j] - b'0') as u32);
                j += 1;
            }
            // closing ]
            if has_m && j < bytes.len() && bytes[j] == b']' && m > 0 {
                let pct = (n as f32 / m as f32).clamp(0.0, 1.0);
                return Some(pct);
            }
        }
        i += 1;
    }
    None
}

/// Rendert eine einzelne, ephemere Statuszeile (inkl. ETA) und schreibt sie über `print_ephemeral`.
/// `predicted_ms` stammt aus den Metriken – wir fusionieren es konservativ mit Builder-Fortschritt.
pub fn render_and_print(p: &mut ProgressState, predicted_ms: u128) {
    // 1) Prozent bestimmen (Builder-Signal vs. Zeit-Schätzung aus Metriken)
    let elapsed_ms = p.start.elapsed().as_millis() as u128;
    let time_pct = if predicted_ms > 0 {
        // Bis 99% steigen, 100% erst beim Abschluss.
        ((elapsed_ms as f32) / (predicted_ms as f32)).clamp(0.0, 0.99)
    } else {
        0.0
    };
    let pct = match p.best_builder_pct {
        Some(b) => if predicted_ms > 0 { b.max(time_pct) } else { b },
        None => time_pct,
    };

    // 2) ETA berechnen (simpel: (1-p)/p * elapsed), geschützt gegen 0
    let eta = if pct > 0.0001 {
        let rem = ((1.0 - pct) / pct) * (elapsed_ms as f32 / 1000.0);
        rem.max(0.0)
    } else {
        // keine sinnvolle Schätzung – nur "…" anzeigen
        -1.0
    };

    // 3) Spinner-Frames (Unicode/ASCII)
    let ascii = matches!(env::var("OTTER_ASCII"), Ok(ref v) if v == "1" || v.eq_ignore_ascii_case("true"));
    const SPIN_UNI: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    const SPIN_ASCII: &[char] = &['|', '/', '-', '\\'];
    p.spinner_ix = if ascii {
        (p.spinner_ix + 1) % SPIN_ASCII.len()
    } else {
        (p.spinner_ix + 1) % SPIN_UNI.len()
    };
    let spin_char = if ascii { SPIN_ASCII[p.spinner_ix] } else { SPIN_UNI[p.spinner_ix] };

    // ----- Layout zuerst UNFARBIG berechnen -----
    let cols = term_cols();
    let pct100 = (pct * 100.0).round() as i32;

    let phase_plain = format!("[{}]", p.runtime_phase);
    let left_plain = format!("{phase_plain} {} {:>3}%", spin_char, pct100);

    let mid = " | ";
    let eta_plain = if eta >= 0.0 {
        format!("ETA {:>4.0}s", eta)
    } else {
        "ETA  …s".to_string()
    };

    // Barbreite dynamisch: Platz nach links/mid/eta + 2 Klammern
    let bar_room = cols
        .saturating_sub(left_plain.len() + mid.len() + eta_plain.len() + 2)
        .clamp(10, 120);
    let filled = ((bar_room as f32) * pct).round() as usize;
    let rest = bar_room.saturating_sub(filled);

    let fill_char = if ascii { '#' } else { '█' };
    let bar_plain = format!(
        "[{}{}]",
        std::iter::repeat(fill_char).take(filled).collect::<String>(),
        std::iter::repeat(' ').take(rest).collect::<String>()
    );

    // Plain-Line (für Breitenbudget & Snippet-Kürzung)
    let line_plain = format!("{left_plain}{mid}{eta_plain} {bar_plain}");
    let remain = cols.saturating_sub(line_plain.len());

    // Snippet vorbereiten (plain, rechtsbündig gekürzt)
    let mut snip = String::new();
    if remain > 4 && !p.last_snippet.is_empty() {
        let mut t = p.last_snippet.replace('\r', "").replace('\n', " ");
        if t.len() > remain {
            t = format!("…{}", &t[t.len() - (remain - 1)..]);
        }
        snip = t;
    }

    // ----- Ab hier Farben anwenden (L1 + L2 + L3) -----
    let colors_on = runner_term::color_enabled();

    // Phase-Tint (L2): configure=blau, build=grün, proc=neutral
    let phase_col = match p.runtime_phase.as_str() {
        "configure" => "\x1b[34m", // BLUE
        "build"     => "\x1b[32m", // GREEN
        _           => "",
    };
    let phase_colored = if colors_on && !phase_col.is_empty() {
        format!("{phase_col}{phase_plain}\x1b[0m")
    } else {
        phase_plain
    };

    // Bar-Farbe (L2): 0–33% rot, 34–66% gelb, 67–99% grün
    let bar_col = if pct < 0.34 {
        "\x1b[31m" // RED
    } else if pct < 0.67 {
        "\x1b[33m" // YELLOW
    } else {
        "\x1b[32m" // GREEN
    };
    let bar_colored = if colors_on {
        format!("{bar_col}{bar_plain}\x1b[0m")
    } else {
        bar_plain
    };

    // Spinner-Farbe (L3): wie Bar-Farbe
    let spin_str_colored = if colors_on {
        format!("{bar_col}{spin_char}\x1b[0m")
    } else {
        spin_char.to_string()
    };

    // Linke Seite neu zusammensetzen (Phase getintet, Spinner farbig)
    let left_colored = format!("{phase_colored} {spin_str_colored} {:>3}%", pct100);

    // Snippet dezent grau (L3)
    let snip_colored = if colors_on && !snip.is_empty() {
        format!("\x1b[90m{snip}\x1b[0m")
    } else {
        snip
    };

    // Finale farbige Zeile (Layout basiert auf den plain-Längen)
    let mut line = format!("{left_colored}{mid}{eta_plain} {bar_colored}");
    if !snip_colored.is_empty() {
        line.push(' ');
        line.push_str(&snip_colored);
    }

    // Takt zurücksetzen und ausgeben
    p.last_tick = Instant::now();
    runner_term::print_ephemeral(&line);
}
