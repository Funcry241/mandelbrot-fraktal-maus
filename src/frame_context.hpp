// Datei: src/frame_context.hpp
// Zeilen: 76
/* 🐭 interner Maus-Kommentar:
   Diese Datei definiert `FrameContext`, den zentralen Container pro Frame.
   Flugente-konform: zoom, offset und newOffset wurden zurück auf float/float2 gestellt.
   Hintergrund: double-Präzision war nicht nötig und kostete unnötig FPS.
   Alle Entropie- und Kontrastdaten bleiben erhalten.
   Schneefuchs sagte: „Wer zu genau sieht, sieht weniger schnell.“
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
    float zoom = 1.0f;
    float2 offset = {0.0f, 0.0f};

    // Auto-Zoom-Steuerung
    bool pauseZoom = false;
    bool shouldZoom = false;
    float2 newOffset = {0.0f, 0.0f}; // neues Ziel (wenn shouldZoom = true)

    // Entropie-Daten
    std::vector<float> h_entropy;   // hostseitig – pro Tile
    std::vector<float> h_contrast;  // Kontrast pro Tile – für Heatmap
    float* d_entropy = nullptr;     // device-seitig
    float* d_contrast = nullptr;    // ✅ NEU: Kontrastwerte auf GPU
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
        d_contrast = nullptr;      // ✅ mit aufräumen
        d_iterations = nullptr;
        shouldZoom = false;
        lastTileIndex = -1;
    }
};
