// Datei: src/frame_context.hpp
// 🦦 Otter: Klar sichtbar als Kapsel, keine faulen pragmas. Konstruktor & Logging unverändert.
// 🦊 Schneefuchs: Speicherstruktur explizit – deterministisch, loggingkompatibel.

#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "settings.hpp"  // Settings::BASE_TILE_SIZE etc.
#include "luchs_log_host.hpp"

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4324) // 🛡️ MSVC: Padding wegen float2 erlaubt – Struktur korrekt genutzt
#endif

class FrameContext {
public:
    // Bilddimensionen
    int width = 0;
    int height = 0;

    // CUDA-Parameter
    int maxIterations;
    int tileSize;

    // Kamera / Fraktalkoordinaten
    float zoom;
    float2 offset;

    // Auto-Zoom-Steuerung
    bool pauseZoom = false;
    bool shouldZoom = false;
    float2 newOffset = {0.0f, 0.0f}; // neues Ziel (wenn shouldZoom = true)

    // Entropie-Daten
    std::vector<float> h_entropy;   // hostseitig - pro Tile
    std::vector<float> h_contrast;  // Kontrast pro Tile - für Heatmap
    float* d_entropy = nullptr;     // device-seitig
    float* d_contrast = nullptr;    // Kontrastwerte auf GPU
    int* d_iterations = nullptr;    // Iterationstiefe je Pixel

    // Statuswerte zur Analyse / Logging
    float lastEntropy = 0.0f;
    float lastContrast = 0.0f;

    // Heatmap Overlay
    bool overlayActive = false;

    // Zeit / Performance
    double frameTime = 0.0;
    double totalTime = 0.0;
    double timeSinceLastZoom = 0.0;

    // Konstruktor initialisiert aus Settings
    FrameContext()
        : maxIterations(Settings::INITIAL_ITERATIONS),
          tileSize(Settings::BASE_TILE_SIZE),
          zoom(Settings::initialZoom),
          offset{Settings::initialOffsetX, Settings::initialOffsetY}
    {}

    // Debug-Ausgaben
    void printDebug() const {
        if (!Settings::debugLogging) return;
        LUCHS_LOG_HOST("[Frame] w=%d h=%d zoom=%.1e offset=(%.5f, %.5f) tiles=%d",
            width, height, zoom, offset.x, offset.y, tileSize);
    }

    // Speicher zurücksetzen - z. B. bei Resize
    void clear() {
        h_entropy.clear();
        h_contrast.clear();
        d_entropy = nullptr;
        d_contrast = nullptr;
        d_iterations = nullptr;
        shouldZoom = false;
    }
};

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
