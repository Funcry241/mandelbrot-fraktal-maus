// Datei: src/zoom_logic.hpp
// 🦦 Otter: Architekturklar, SIMD-kompatibel. Pragma jetzt lokal. Kein globales alignas nötig.
// 🐅 Schneefuchs: Layout stabil, Verhalten eindeutig. Kein Fehlalarm mehr unter /WX. 
// 🐼 Panda: Entscheidungsstruktur für Auto-Zoom-Logik klar abgegrenzt und kommentiert.

#pragma once
#include "common.hpp"
#include <vector>
#include <vector_types.h> // float2

namespace ZoomLogic {

// 🛡️ Fallback für make_float2() – nur wenn CUDA-seitig nicht vorhanden
#ifndef __CUDACC__
[[nodiscard]] static inline float2 make_float2(float x, float y) {
    float2 f;
    f.x = x;
    f.y = y;
    return f;
}
#endif

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4324) // Strukturende mit float2 kann Padding erzeugen – kein Problem
#endif

/// 🎯 Ergebnisstruktur für das Auto-Zoom-Ziel.
/// Wird pro Frame neu berechnet. Speichert Zielkoordinaten, Score und Analysewerte.
class ZoomResult {
public:
    int bestIndex = -1;                // Index des besten Tiles (Rasterindex)
    float bestEntropy = 0.0f;          // Entropie dieses Tiles
    float bestContrast = 0.0f;         // Kontrastwert
    float bestScore = 0.0f;            // Gesamtscore (kombiniert Entropie/Distanz/Bias)

    float distance = 0.0f;             // Entfernung zum aktuellen Offset
    float minDistance = 0.0f;          // Mindestabstand zur Zieländerung

    float relEntropyGain  = 0.0f;      // Entropiezuwachs gegenüber vorherigem Ziel
    float relContrastGain = 0.0f;      // Kontrastzuwachs gegenüber vorherigem Ziel

    bool isNewTarget = false;          // Wechselt das Ziel im Vergleich zum Vorframe?
    bool shouldZoom   = false;         // Sollte gezoomt werden?

    float2 newOffset = make_float2(0.0f, 0.0f); // Zielkoordinaten im Fraktalraum
    std::vector<float> perTileContrast;        // Optionale Visualisierung der Kontraste
};

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

/// 🐼 Kontrastanalyse über direkte Nachbarn (oben/unten/links/rechts) pro Tile.
/// Liefert Kontrastwert zwischen Zentrum und Umgebung.
/// Liefert 0.0 wenn Breite oder Höhe < tileSize
[[nodiscard]]
float computeEntropyContrast(
    const std::vector<float>& entropy,
    int width, int height, int tileSize);

/// 🐘 Hauptentscheidung für Auto-Zoom.
/// Bewertet alle Tiles nach Score, Zielentfernung, Bias und Analysewerten.
/// Gibt vollständige ZoomResult-Struktur zurück.
[[nodiscard]]
ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,
    const std::vector<float>& contrast,
    float2 currentOffset, float zoom,
    int width, int height, int tileSize,
    float2 previousOffset, int previousIndex,
    float previousEntropy, float previousContrast
);

} // namespace ZoomLogic
