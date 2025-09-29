///// Otter: FrameContext V3 — feste Typen Vec2f/Vec2d (keine anonymen Structs) -> Zuweisungen kompiliert.
///// Schneefuchs: /WX-kompatibel, ASCII-only; keine GL/CUDA-Includes im Header.
///// Maus: Backcompat-Felder (newOffset) wieder da; Double ist Quelle der Wahrheit. ***/
///// Datei: src/frame_context.hpp

#pragma once

#include <vector>

// Kleine Vektor-Typen (ASCII, ohne Operator-Overloads)
struct Vec2f { float  x = 0.0f; float  y = 0.0f; };
struct Vec2d { double x = 0.0;  double y = 0.0;  };

struct FrameContext {
    // Geometrie / Render-Parameter
    int   width         = 0;
    int   height        = 0;
    int   maxIterations = 0;
    int   tileSize      = 0;

    // Float-Spiegel (wird aus Double synchronisiert)
    float zoom          = 0.0f;
    Vec2f offset        = {};
    Vec2f newOffset     = {};   // Backcompat: wird aus newOffsetD gespiegelt

    // Autoritative Double-Werte (Quelle der Wahrheit)
    double zoomD        = 0.0;
    Vec2d  offsetD      = {};
    Vec2d  newOffsetD   = {};

    bool   shouldZoom   = false;

    // Zeitdelta fuer Normierung (Sekunden)
    float deltaSeconds  = 0.0f;

    // Heatmap-Statistiken (Groesse == tilesX*tilesY)
    std::vector<float> entropy;
    std::vector<float> contrast;

    // API
    FrameContext();
    void clear() noexcept;
    void printDebug() const noexcept;

    // Spiegelt zoomD/offsetD/newOffsetD -> zoom/offset/newOffset
    void syncFloatFromDouble() noexcept;
};
