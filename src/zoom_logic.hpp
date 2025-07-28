// Datei: src/zoom_logic.hpp
// 🦦 Otter: Architekturklar, SIMD-kompatibel. Pragma jetzt lokal. Kein globales alignas nötig.
// 🐅 Schneefuchs: Layout stabil, Verhalten eindeutig. Kein Fehlalarm mehr unter /WX.

#pragma once
#include "common.hpp"
#include <vector>
#include <vector_types.h> // float2

namespace ZoomLogic {

// 🛡️ Fallback für make_float2() - nur wenn nicht CUDA-seitig vorhanden
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
    #pragma warning(disable: 4324) // Struktur gepadded wegen float2 am Ende
#endif

/// 🎯 Datenstruktur für das beste Zoom-Ziel
/// Wird jedes Frame neu berechnet - enthält Bewertung & Koordinaten
class ZoomResult {
public:
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

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

/// 🐼 Panda: Entropie-Kontrastberechnung - mittelt über 4 direkte Nachbarn (oben, unten, links, rechts)
[[nodiscard]]
float computeEntropyContrast(const std::vector<float>& entropy, int width, int height, int tileSize);

/// 🐘 + 🦦 + 🕊️ evaluateZoomTarget - zentrales Entscheidungssystem für Auto-Zoom.
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
