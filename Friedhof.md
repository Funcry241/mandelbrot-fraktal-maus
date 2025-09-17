///// Otter: Archiv der Tiercodenamen, die im aktuellen Fallback-Zip nicht aktiv im Code vorkommen (nur Doku, keine Funktionalität).
///// Schneefuchs: Quelle ist der Scan von src.zip am 2025-09-17 (Europe/Berlin); Liste dient als Kanon für Reserviert/Retired.
///// Maus: Regeln: eindeutig, ASCII, keine Doppelvergabe; Wiederbelebung = bewusste Reaktivierung mit Commit-Notiz.
///// Datei: Friedhof.md

# Tierfriedhof – Archiv ungenutzter/retirierter Codenamen

**Zweck.** Diese Datei hält Tier-Codenamen fest, die im aktuellen Fallback-Stand der Codebasis **nicht** aktiv im Code vorkommen.
Sie verhindert Doppelvergabe, dokumentiert Historie und skizziert eine mögliche „Wiederbelebung“.

---

## Aufnahme-Kriterien

- **Nicht im Code gefunden** (aktueller /src-Scan).
- **Konzeptuell genutzt**, aber im Fallback-Zip **nicht** verdrahtet.
- **Reserviert** für spätere Features, um Namenskonflikte zu vermeiden.

> Quelle des Scans: `src.zip` Stand 2025-09-17.

---

## Roster (Retired / Reserviert)

| Tiercodename | Status       | Letztbezug (Historie)          | Begründung (kurz)                                  | Idee für Wiederbelebung |
|---|---|---|---|---|
| **Eule** | reserviert (in Arbeit) | 2025-08-07 „Projekt Eule“, README Features | Koordinaten-/Heatmap-Thema, in README aktiv, im Codepfad noch nicht | „Eule v2“: zentrale Transform-Lib + GPU-Heatmap |
| **Mücke** | archiviert | 2025-08-11 „Mücke“-Baseline | Perf-/Nerv-Codename; kein aktiver Codepfad im Fallback | Micro-Scheduler für Frame-Budget-Pacing |
| **Ringelrobbe** | archiviert | früheres Debug-Zweiglein | Ehem. Visual-/Debug-Variante, im Fallback entfernt | Shader-Showcase-Branch (Palette/Param-Morph) |
| **Biber** | reserviert | – | Noch nie aktiv vergeben; passt zu Persist/IO/Cache | Robuste Session-/Preset-Speicher („Biber“) |
| **Dachs** | reserviert | – | Freigehalten für Stabilitäts-/Recovery-Module | Crash-Recovery + Auto-Restore Pipelines |
| **Krähe** | reserviert | – | Monitoring/Watcher passend, noch ungenutzt | Event-Watchdog + Heuristik-Alarme im Loop |

> **Nicht auf dem Friedhof:** Bereits **aktiv vergebene** oder **in Doku-Prinzipien aktive** Codenamen (z. B. *Otter, Schneefuchs, Maus, Warzenschwein, Hermelin, Nacktmull, Kolibri, Pfau, Bär, Robbe, Waschbär, Luchs*) gehören **nicht** hierher.

---

## Regeln für Aufnahme/Entnahme

1. **Eindeutigkeit.** Ein Codename existiert zu einem Zeitpunkt nur in **einem** Status (aktiv *oder* archiviert/reserviert).
2. **Änderungen mit Commit-Notiz.** Jede Verschiebung (aktiv ⇄ Friedhof) bekommt eine kurze **Einzeiler-Notiz** im Commit.
3. **Kein Auto-Recycling.** Reaktivierung bedeutet **bewusste** Wiederverwendung samt kurzer README-Begründung.
4. **ASCII & Datumsangaben.** Einträge sind ASCII-rein, Datum im Format `YYYY-MM-DD`.

---

## Changelog

- **2025-09-17**: Initiale Anlage basierend auf Fallback-Scan. Einträge: Eule (reserviert/in Arbeit), Mücke (archiviert), Ringelrobbe (archiviert), Biber (reserviert), Dachs (reserviert), Krähe (reserviert).

