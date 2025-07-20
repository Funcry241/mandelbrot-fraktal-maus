// Datei: src/zoom_logic.hpp
// Zeilen: 47
// 🐭 Maus-Kommentar: Alpha 49.1 – ZoomResult nun auch selbst [[nodiscard]], schützt gegen unbeachtete Konstrukte. Vollständig Clang/CUDA-kompatibel, exakt dokumentiert.
// 🦦 Otter: Eindeutige Semantik – Ergebnis muss verwendet werden, sonst droht Zoomverlust.
// 🐅 Maus: Kompakt, robust, klar priorisiert – ideal als Public API des Zoommoduls.

#pragma once
#include "common.hpp"
#include <vector>
#include <vector_types.h> // float2

namespace ZoomLogic {

/// 🎯 Datenstruktur für das beste Zoom-Ziel
/// Wird jedes Frame neu berechnet – enthält Bewertung & Koordinaten
struct ZoomResult {
    int bestIndex = -1;                // Index im Tile-Raster
    float bestEntropy = 0.0f;          // Entropiewert dieses Tiles
    float bestContrast = 0.0f;         // Kontrast zum Nachbarumfeld
    float bestScore = 0.0f;            // Gesamtscore (gewichtete Mischung)

    float distance = 0.0f;             // Abstand zum aktuellen Ziel
    float minDistance = 0.0f;          // minimale akzeptierte Distanz (Zoomschwelle)

    float relEntropyGain = 0.0f;       // Entropiezuwachs gegenüber vorherigem Ziel
    float relContrastGain = 0.0f;      // Kontrastzuwachs gegenüber vorherigem Ziel

    bool isNewTarget = false;          // Wechselt das Ziel? (Relevanzsprung)
    bool shouldZoom = false;           // Soll in das Ziel hineingezoomt werden?

    float2 newOffset = make_float2(0.0f, 0.0f); // Koordinaten im Fraktalraum
    std::vector<float> perTileContrast;         // Optional: Rückkanal für Heatmap
};

/// 🐼 Panda: Entropie-Kontrastberechnung – mittelt über 4 direkte Nachbarn (oben, unten, links, rechts)
/// Liefert Maß für lokale visuelle Struktur (Gradienten/Übergänge).
[[nodiscard]]
float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize);

/// 🐘 + 🦦 + 🕊️ evaluateZoomTarget – zentrales Entscheidungssystem für Auto-Zoom.
/// Analysiert die Entropie- und Kontrastkarten, trifft Entscheidung über das nächste Ziel.
/// Gibt vollständige Bewertungsstruktur (ZoomResult) zurück.
[[nodiscard]]
ZoomResult evaluateZoomTarget(
    const std::vector<float>& entropy,          // Entropiekarte vom GPU-Kernel
    const std::vector<float>& contrast,         // Kontrastwerte pro Tile
    float2 currentOffset, float zoom,           // Aktuelle Ansicht (Kamera)
    int width, int height, int tileSize,        // Bildgröße & Tile-Auflösung
    float2 previousOffset, int previousIndex,   // Letztes Ziel
    float previousEntropy, float previousContrast // Letzte Zielbewertung
);

} // namespace ZoomLogic
