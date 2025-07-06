// Datei: src/frame_context.hpp
// Zeilen: 82
/* 🐭 interner Maus-Kommentar:
   FrameContext initialisiert ab sofort alle Kernwerte aus Settings.
   Keine Magic Numbers mehr! Flugente: immer synchron, immer robust.
   Schneefuchs: „Wartbarkeit ist, wenn Settings überall gelten.“
*/

#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "settings.hpp"        // Settings::BASE_TILE_SIZE etc.

struct FrameContext {
    // Bilddimensionen
    int width = 0;
    int height = 0;

    // CUDA-Parameter
    int maxIterations;
    int tileSize;
    int supersampling;

    // Kamera / Fraktalkoordinaten
    float zoom;
    float2 offset;

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

    // Konstruktor initialisiert alles aus Settings
    FrameContext()
        : maxIterations(Settings::INITIAL_ITERATIONS),
          tileSize(Settings::BASE_TILE_SIZE),
          supersampling(Settings::defaultSupersampling),
          zoom(Settings::initialZoom),
          offset{Settings::initialOffsetX, Settings::initialOffsetY}
    {}

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
        d_contrast = nullptr;
        d_iterations = nullptr;
        shouldZoom = false;
        lastTileIndex = -1;
    }
};
