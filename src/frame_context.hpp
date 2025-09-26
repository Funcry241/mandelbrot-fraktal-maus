///// Otter: Klar sichtbare Kapsel; keine schweren Includes im Header (nur gezielt vector_types.h).
///// Schneefuchs: float2 via <vector_types.h>; gezielte C4324-Unterdrückung; noexcept wo sinnvoll.
///// Maus: ASCII-only; Header/Source synchron; keine impliziten GL-Includes.
///// Datei: src/frame_context.hpp

#pragma once
#include <vector_types.h> // float2/double2 (__align__ → kann C4324 auslösen, daher pragma im .hpp)

#if defined(_MSC_VER)
  #pragma warning(push)
  #pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif

// Ein Frame-Schnappschuss der *Parameter* (GPU/Host-Puffer liegen im RendererState).
// Maus: Ab tiefen Zoomstufen (≈1e8) brauchen wir höhere Präzision im Mapping.
//       Daher führen wir *autoritative* Double-Felder ein und spiegeln nach float.
struct FrameContext {
    // Zielauflösung (Pixel)
    int   width         = 0;
    int   height        = 0;

    // Iterations- und Tiling-Parameter
    int   maxIterations = 0;
    int   tileSize      = 0;

    // ---- Autoritative Präzision (Double) – von Zoom-Logik & Mapping verwendet ----
    double  zoomD       = 1.0;            // primärer Zoom (double, Quelle der Wahrheit)
    double2 offsetD     = {0.0, 0.0};     // Bildzentrum (double)
    bool    shouldZoom  = false;          // Ergebnis der Zielsuche
    double2 newOffsetD  = {0.0, 0.0};     // neues Zielzentrum (double)

    // ---- Legacy/Convenience (Float) – für bestehende Pfade, die float erwarten ----
    // Hinweis: Diese Felder werden aus den Double-Werten gespiegelt.
    float   zoom        = 1.0f;
    float2  offset      = {0.0f, 0.0f};
    float2  newOffset   = {0.0f, 0.0f};

    // Konstruktor initialisiert aus Settings (Definition in .cpp)
    FrameContext();

    // Debug-Ausgaben
    void printDebug() const noexcept;

    // Zurücksetzen von Zoom-Entscheidungen (z. B. nach Resize)
    void clear() noexcept;

    // ----------------- Sync-Helfer (Header-only, noexcept) -----------------
    // Spiegelt die autoritativen Double-Felder in die Float-Spiegel (ohne Runden-Logik).
    inline void syncFloatFromDouble() noexcept {
        zoom        = static_cast<float>(zoomD);
        offset.x    = static_cast<float>(offsetD.x);
        offset.y    = static_cast<float>(offsetD.y);
        newOffset.x = static_cast<float>(newOffsetD.x);
        newOffset.y = static_cast<float>(newOffsetD.y);
    }

    // Spiegelt die Float-Spiegel (falls befüllt) zurück in die autoritativen Double-Felder.
    inline void syncDoubleFromFloat() noexcept {
        zoom        = (zoom <= 0.0f) ? 1.0f : zoom; // kleine Robustheit
        zoomD       = static_cast<double>(zoom);
        offsetD.x   = static_cast<double>(offset.x);
        offsetD.y   = static_cast<double>(offset.y);
        newOffsetD.x= static_cast<double>(newOffset.x);
        newOffsetD.y= static_cast<double>(newOffset.y);
    }
};

#if defined(_MSC_VER)
  #pragma warning(pop)
#endif
