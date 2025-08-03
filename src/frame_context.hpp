// Datei: src/frame_context.hpp
// 🦦 Otter: Klar sichtbar als Kapsel, keine faulen pragmas. Konstruktor & Logging deklariert.
// 🦊 Schneefuchs: Speicherstruktur explizit - deterministisch, loggingkompatibel.

#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "settings.hpp"
#include "luchs_log_host.hpp"

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4324) // 🛡️ MSVC: Padding wegen float2 erlaubt - Struktur korrekt genutzt
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

    int frameIndex = 0; // 🦦 Otter: für Kontext-Zeitachsenanalyse – wird pro Frame gesetzt

    // Konstruktor initialisiert aus Settings
    FrameContext();

    // Debug-Ausgaben
    void printDebug() const;

    // Speicher zurücksetzen - z. B. bei Resize
    void clear();
};

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
