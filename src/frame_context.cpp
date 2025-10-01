///// Otter: C4702 fix - logging compiled-in only when enabled; no unreachable code.
///// Schneefuchs: /WX strikt, ASCII-only Logs/Kommentare; robuste, klare Initialisierung.
///// Maus: Einheitlicher Vec2*-Einsatz; Backcompat fuer newOffset; keine anonymen Typen mehr. ***/
///// Datei: src/frame_context.cpp

#include "frame_context.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

FrameContext::FrameContext()
{
    // Basis
    width         = 0;
    height        = 0;
    maxIterations = Settings::INITIAL_ITERATIONS;
    tileSize      = Settings::BASE_TILE_SIZE;

    // Zeit/Stats: neutrale Defaults
    deltaSeconds  = 0.0f;
    entropy.clear();
    contrast.clear();

    // Autoritative Double-Werte initialisieren ...
    zoomD         = static_cast<double>(Settings::initialZoom);
    offsetD       = { 0.0, 0.0 };
    newOffsetD    = offsetD;      // gleicher Typ -> Zuweisung OK
    shouldZoom    = false;

    // ... und Float-Spiegel daraus ableiten.
    syncFloatFromDouble();
}

void FrameContext::clear() noexcept {
    // Nur die Zoom-Entscheidung zuruecksetzen; Pufferverwaltung ist im Renderer/State.
    shouldZoom = false;
    newOffsetD = offsetD;     // Double ist Quelle der Wahrheit
    syncFloatFromDouble();    // Float-Spiegel aktualisieren
}

void FrameContext::syncFloatFromDouble() noexcept {
    zoom      = static_cast<float>(zoomD);
    offset    = { static_cast<float>(offsetD.x),    static_cast<float>(offsetD.y) };
    newOffset = { static_cast<float>(newOffsetD.x), static_cast<float>(newOffsetD.y) };
}

void FrameContext::printDebug() const noexcept {
    if constexpr (Settings::debugLogging) { // C4702: wird nur einkompiliert, wenn aktiv
        LUCHS_LOG_HOST(
            "[Frame] w=%d h=%d it=%d tile=%d stats=%d "
            "zoomD=%.12e offD=(%.12e,%.12e) newOffD=(%.12e,%.12e) "
            "zoom=%.6f off=(%.6f,%.6f) newOff=(%.6f,%.6f) dt=%.6f e=%zu c=%zu",
            width, height, maxIterations, tileSize, statsTileSize,
            zoomD, offsetD.x, offsetD.y, newOffsetD.x, newOffsetD.y,
            zoom,  offset.x,  offset.y,  newOffset.x,  newOffset.y,
            deltaSeconds, entropy.size(), contrast.size()
        );
    }
}
