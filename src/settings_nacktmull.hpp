///// MAUS
///// OWNER
///// RESERVED
///// Datei: src/settings_nacktmull.hpp

#pragma once

// ============================================================================
// Nacktmull-Settings (additive, funktionsneutral bis zur Nutzung)
// Policy:
//   - Logs sind ASCII-only (gilt projektweit); in diesem Header KEINE Logs.
//   - Kommentare dürfen Deutsch sein; Runtime-Strings bleiben Englisch.
//   - Defaults: Planner3D jetzt AKTIV (enabled=true) für Anti-Crawl & Snap.
//               Progressive bewusst noch AUS (enabled=false) — sicherer Schritt.
//   - Header-only; keine versteckten Abhängigkeiten; /WX-fest.
//   - Host-Only Datenträger (keine __host__/__device__ Anmerkungen hier).
// ============================================================================

#include <cstdint>

namespace NacktmullSettings {

// ---------------------------
// 3D Planner (x, y, logZoom)
// ---------------------------
// Ziel: Kein "End-Creep". PD-Regler arbeitet in (x,y,logZoom) mit Mindest-
// geschwindigkeit (v_min) und Snap-Zone. In diesem Header NUR Parameter;
// keine Wirkung, bis konsumierende Stellen sie verwenden.
struct Planner3D {
    // Aktiviert den 3D-Planner-Pfad (x, y, logZoom).
    // Bereich: {false, true} | Default: true (aktiv)
    bool enabled;

    // Skaliert den Fehler in der Zoom-Achse: ez' = ez / kZ.
    // Wirkung: Größer => Zoom-Fehler wiegt weniger (ruhiger in z),
    //          Kleiner => aggressiveres Nachziehen in z.
    // Empfehlung: 1.2 .. 2.0 | Default: 1.6
    double kZ;

    // Dämpfungsverhältnis ζ (zeta) für PD (kritisch bei ~1.0).
    // Wirkung: Größer => stärker gedämpft, weniger Überschwingen,
    //          Kleiner => schneller, aber wackeliger.
    // Empfehlung: 0.6 .. 1.1 | Default: 0.8
    double zeta;

    // Eigenfrequenz ωn [1/s] (Tempo der Annäherung).
    // Wirkung: Größer => schnelleres Einregeln,
    //          Kleiner => weicher/behäbiger.
    // Empfehlung: 2.0 .. 5.0 | Default: 3.0
    double omegaN;

    // Mindestgeschwindigkeits-Faktor für Anti-Crawl auf die normierte
    // Fehlerlänge: v_min = factor * ||e'|| / dt.
    // Wirkung: Größer => härteres "Durchziehen" nahe Ziel,
    //          Kleiner => sanfter, aber potentiell Kriechen.
    // Empfehlung: 0.03 .. 0.10 | Default: 0.06
    double vminFactor;

    // Snap-Schwelle auf die normierte Fehlerlänge ||e'||.
    // Wirkung: Größer => früheres hartes Einrasten (schnelles "fertig"),
    //          Kleiner => längeres, weicheres Auslaufen.
    // Empfehlung: 0.001 .. 0.010 | Default: 0.003
    double snapEps;

    // Optional: Planner-spezifische Logzeile aktivieren.
    // Bereich: {false, true} | Default: true (nur wirksam, wenn konsumiert)
    bool logEnabled;
};

// Abgeleitete PD-Gewichte (normierter Raum):
//   Kp = ωn^2, Kd = 2ζωn
struct PlannerGains {
    double Kp;
    double Kd;
};

[[nodiscard]] inline constexpr PlannerGains gainsFrom(const Planner3D& p) {
    return PlannerGains{ p.omegaN * p.omegaN, 2.0 * p.zeta * p.omegaN };
}

// ---------------------------
// Progressive Iteration (GPU)
// ---------------------------
// Ziel: Arbeit in die Zeitdimension strecken (Resume). Hier nur Parameter;
// erst Konsumenten (Kernels/Wrapper) setzen das um.
struct Progressive {
    // Aktiviert den Progressive-Pfad (per-pixel Resume).
    // Bereich: {false, true} | Default: false (vorerst aus)
    bool enabled;

    // Minimale Iterations-Slice-Länge pro Touch.
    // Wirkung: Größer => weniger Scheduling-Overhead, aber gröbere Granularität.
    //          Kleiner => feinere Verteilung, potenziell mehr Overhead.
    // Empfehlung: 32 .. 256 | Default: 64
    std::uint32_t sliceMin;

    // Maximale Slice-Länge relativ zu maxIter (z. B. 0.5 = maxIter/2).
    // Wirkung: Größer => schnellere Fertigstellung einzelner "Zement-Pixel",
    //          Kleiner => geringere Tail-Latency-Schwankungen.
    // Empfehlung: 0.25 .. 0.75 | Default: 0.5
    double sliceMaxPct;

    // Aufteilung Survivor-Queues (Hi/Lo) in Prozent für "Hi".
    // Wirkung: Größer => Hi bekommt mehr Budget und wird schneller leer,
    //          Kleiner => ausgewogener, aber potenziell längerer Tail.
    // Empfehlung: 60 .. 90 | Default: 80
    double hiLoSplitPct;

    // Abbruchschwelle (in Prozent der Gesamtpixel) für "wir machen im
    // nächsten Frame weiter", um den letzten Rest nicht zu erzwingen.
    // Wirkung: Größer => frühes Weiterreichen an nächsten Frame (kürzere Peaks),
    //          Kleiner => mehr "zu Ende quälen" innerhalb eines Frames.
    // Empfehlung: 0.05 .. 1.0 | Default: 0.2
    double stopThresholdSurvivorsPct;

    // Optional: Device-Debug-Logging innerhalb der Kernels.
    // Bereich: {false, true} | Default: false (Logs aus)
    bool deviceDebugLog;
};

// ---------------------------
// Settings-Druck bei Start
// ---------------------------
struct Printing {
    // Postet alle Settings beim Start ins Host-Log (ASCII).
    // Bereich: {false, true} | Default: false (funktionsneutral)
    bool printAllAtStartup;
};

// ---------------------------
// Defaults (inline constexpr)
// ---------------------------
inline constexpr Planner3D Planner3D_Default {
    /*enabled*/          true,
    /*kZ*/               1.6,
    /*zeta*/             0.8,
    /*omegaN*/           3.0,
    /*vminFactor*/       0.06,
    /*snapEps*/          0.003,
    /*logEnabled*/       true
};

inline constexpr Progressive Progressive_Default {
    /*enabled*/                      false,
    /*sliceMin*/                     64u,
    /*sliceMaxPct*/                  0.5,
    /*hiLoSplitPct*/                 80.0,
    /*stopThresholdSurvivorsPct*/    0.2,
    /*deviceDebugLog*/               false
};

inline constexpr Printing Printing_Default {
    /*printAllAtStartup*/ false
};

} // namespace NacktmullSettings
