///// Otter: Klar sichtbare Kapsel; keine schweren Includes im Header (nur gezielt vector_types.h).
///// Schneefuchs: float2 via <vector_types.h>; gezielte C4324-Unterdrückung; noexcept wo sinnvoll.
///// Maus: ASCII-only; Header/Source synchron; keine impliziten GL-Includes.
///// Datei: src/frame_context.hpp

#pragma once
#include <vector_types.h> // float2 (__align__ → kann C4324 auslösen, daher pragma im .hpp)

#if defined(_MSC_VER)
  #pragma warning(push)
  #pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif

// Ein Frame-Schnappschuss der *Parameter* (GPU/Host-Puffer liegen im RendererState)
struct FrameContext {
    // Zielauflösung (Pixel)
    int   width        = 0;
    int   height       = 0;

    // Iterations- und Tiling-Parameter
    int   maxIterations = 0;
    int   tileSize      = 0;

    // Fraktal-Ansicht
    float   zoom   = 1.0f;
    float2  offset {0.0f, 0.0f};

    // Ergebnis der Analyse/Zoomlogik (vom Pipeline-Schritt beschrieben)
    bool    shouldZoom = false;
    float2  newOffset  {0.0f, 0.0f};

    // Konstruktor initialisiert aus Settings (Definition in .cpp)
    FrameContext();

    // Debug-Ausgaben
    void printDebug() const noexcept;

    // Zurücksetzen von Zoom-Entscheidungen (z. B. nach Resize)
    void clear() noexcept;
};

#if defined(_MSC_VER)
  #pragma warning(pop)
#endif
