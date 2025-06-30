// Datei: src/frame_context.hpp
// Zeilen: 75
/* 🐭 interner Maus-Kommentar:
   Diese Datei definiert `FrameContext`, den zentralen Container pro Frame.
   Alle Module greifen ausschließlich über dieses Objekt auf aktuelle Zustände zu.
   Damit wird Auto-Zoom, HUD, Heatmap und Rendering deterministisch und modular.
   → Grundlage für CommandBus und Replay.
   → Alle Koordinaten und Entropiedaten liegen hier zentral.
   → FIX: zoom, offset, newOffset jetzt double/double2 – volle Präzision für tiefe Zoomstufen (Schneefuchs-Fund)
*/

#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "settings.hpp"        // für Settings::tileSize, debugLogging etc.

struct FrameContext {
    // Bilddimensionen
    int width = 0;
    int height = 0;

    // CUDA-Parameter
    int maxIterations = 500;
    int tileSize = 32;
    int supersampling = 1;

    // Kamera / Fraktalkoordinaten
    double zoom = 1.0;
    double2 offset = {0.0, 0.0};

    // Auto-Zoom-Steuerung
    bool pauseZoom = false;
    bool shouldZoom = false;
    double2 newOffset = {0.0, 0.0}; // neues Ziel (wenn shouldZoom = true)

    // Entropie-Daten
    std::vector<float> h_entropy;   // hostseitig – pro Tile
    std::vector<float> h_contrast;  // Kontrast pro Tile – für Heatmap
    float* d_entropy = nullptr;     // device-seitig
    int* d_iterations = nullptr;    // Iterationstiefe je Pixel

    // Statuswerte zur Analyse / Logging
    float lastEntropy = 0.0f;
    float lastContrast = 0.0f;
    int lastTileIndex = -1;

    // Heatmap Overlay
    bool overlayActive = false;

    // Zeit / Performance
    double frameTime = 0.0;
    double totalTime = 0.0;
    double timeSinceLastZoom = 0.0;

    // Debug-Ausgaben
    void printDebug() const {
        if (!Settings::debugLogging) return;
        printf("[Frame] w=%d h=%d zoom=%.1e offset=(%.5f, %.5f) tiles=%d\n",
            width, height, zoom, offset.x, offset.y, tileSize);
    }

    // Speicher zurücksetzen – z. B. bei Resize
    void clear() {
        h_entropy.clear();
        h_contrast.clear();
        d_entropy = nullptr;
        d_iterations = nullptr;
        shouldZoom = false;
        lastTileIndex = -1;
    }
};
