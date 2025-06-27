// Datei: src/zoom_logic.hpp
// Zeilen: 37
// 🐭 Maus-Kommentar: ZoomResult liefert jetzt double2 für präzise Zielkoordinaten. Kein float2 mehr, um Genauigkeit bei starkem Zoom zu bewahren. Schneefuchs nickt zustimmend.

#pragma once
#include "common.hpp"
#include "renderer_state.hpp"
#include <vector>

namespace ZoomLogic {

struct ZoomResult {
    int bestIndex = -1;
    float bestEntropy = 0.0f;
    float bestContrast = 0.0f;
    float bestScore = 0.0f;
    float distance = 0.0f;
    float minDistance = 0.0f;
    float relEntropyGain = 0.0f;
    float relContrastGain = 0.0f;
    bool isNewTarget = false;
    bool shouldZoom = false;
    double2 newOffset = make_double2(0.0, 0.0);
};

ZoomResult evaluateZoomTarget(
    const std::vector<float>& h_entropy,
    double2 currentOffset,
    double currentZoom,
    int width,
    int height,
    int tileSize,
    const RendererState& state
);

} // namespace ZoomLogic
