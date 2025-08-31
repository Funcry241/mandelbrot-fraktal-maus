///// Otter: Klar sichtbare Kapsel; keine schweren Includes im Header.
///// Schneefuchs: float2 via <vector_types.h>; gezielte C4324-Unterdrückung; noexcept wo sinnvoll.
///// Maus: ASCII-only; Header/Source synchron; keine impliziten GL-Includes.
///// Datei: src/frame_context.hpp

#pragma once
#include <vector>
#include <vector_types.h> // float2 (__align__ erzwingt 8-Byte-Alignment → C4324 unter /WX)

// 🐭 Maus: Intentional alignment due to CUDA float2 members; silence MSVC C4324 lokal.
// 🦦 Otter: Local pragma keeps /WX global strikt.
// 🐑 Schneefuchs: Gezielt und dokumentiert – nur um FrameContext.

#if defined(_MSC_VER)
  #pragma warning(push)
  #pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif

class FrameContext {
public:
    // Bilddimensionen
    int   width  = 0;
    int   height = 0;

    // CUDA-Parameter
    int   maxIterations = 0;
    int   tileSize      = 0;

    // Kamera / Fraktalkoordinaten
    float  zoom   = 1.0f;
    float2 offset = {0.0f, 0.0f};

    // Auto-Zoom-Steuerung
    bool   pauseZoom  = false;
    bool   shouldZoom = false;
    float2 newOffset  = {0.0f, 0.0f}; // neues Ziel (wenn shouldZoom = true)

    // Entropie-Daten
    std::vector<float> h_entropy;   // hostseitig - pro Tile
    std::vector<float> h_contrast;  // Kontrast pro Tile - für Heatmap
    float* d_entropy    = nullptr;  // device-seitig
    float* d_contrast   = nullptr;  // Kontrastwerte auf GPU
    int*   d_iterations = nullptr;  // Iterationstiefe je Pixel

    // Statuswerte zur Analyse / Logging
    float lastEntropy  = 0.0f;
    float lastContrast = 0.0f;

    // Heatmap Overlay
    bool overlayActive = false;

    // Zeit / Performance
    double frameTime         = 0.0;
    double totalTime         = 0.0;
    double timeSinceLastZoom = 0.0;

    int frameIndex = 0; // 🦦 Otter: für Kontext-Zeitachsenanalyse – wird pro Frame gesetzt

    // Konstruktor initialisiert aus Settings (Definition in .cpp)
    FrameContext();

    // Debug-Ausgaben
    void printDebug() const noexcept;

    // Speicher zurücksetzen - z. B. bei Resize
    void clear() noexcept;
};

#if defined(_MSC_VER)
  #pragma warning(pop)
#endif
