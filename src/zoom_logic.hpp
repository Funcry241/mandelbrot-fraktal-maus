// zoom_logic.hpp - Zeilen: 52

/*
🐭 Maus-Kommentar: Deklariert die Zoom-Zielbewertungslogik getrennt von CUDA. Erlaubt saubere Trennung von Zustandsdaten und Zielauswahl. Schneefuchs: „Ein klarer Kopf entscheidet besser.“
*/

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "renderer_state.hpp"

namespace ZoomLogic {

// Ergebnisstruktur einer Zielbewertung
struct ZoomResult {
    float2 newOffset;
    bool shouldZoom;
    float bestScore;
    int bestIndex;
    float bestEntropy;
    float bestContrast;
    float distance;
    float minDistance;
    float relEntropyGain;
    float relContrastGain;
    bool isNewTarget;
};

// Bewertet alle Tiles und bestimmt, ob ein neues Zoom-Ziel gewählt werden soll
ZoomResult evaluateZoomTarget(
    const std::vector<float>& h_entropy,
    float2 offset,
    float zoom,
    int width,
    int height,
    int tileSize,
    RendererState& state
);

// Berechnet lokalen Entropiekontrast eines Tiles (Mittelwert zu Nachbarn)
float computeEntropyContrast(
    const std::vector<float>& h,
    int index,
    int tilesX,
    int tilesY
);

}  // namespace ZoomLogic
