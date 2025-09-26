///// Otter: C4702 fix – logging compiled-in only when enabled; no unreachable code.
///// Schneefuchs: /WX strikt, ASCII-only Logs/Kommentare; Verhalten unverändert.
///// Maus: Saubere Trennung von FrameContext-Daten und Logik (Ameise).
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

    // Autoritative Double-Werte initialisieren …
    zoomD         = static_cast<double>(Settings::initialZoom);
    offsetD       = { 0.0, 0.0 };
    newOffsetD    = offsetD;
    shouldZoom    = false;

    // … und Float-Spiegel daraus ableiten.
    syncFloatFromDouble();
}

void FrameContext::clear() noexcept {
    // Nur die Zoom-Entscheidung zurücksetzen; Pufferverwaltung ist im RendererState.
    shouldZoom = false;
    newOffsetD = offsetD;     // Double ist Quelle der Wahrheit
    syncFloatFromDouble();    // Float-Spiegel aktualisieren
}

void FrameContext::printDebug() const noexcept {
    if constexpr (Settings::debugLogging) { // C4702: kompiliert nur rein, wenn aktiv
        LUCHS_LOG_HOST(
            "[Frame] w=%d h=%d it=%d tile=%d "
            "zoomD=%.12e offD=(%.12e,%.12e) "
            "zoom=%.6f off=(%.6f,%.6f)",
            width, height, maxIterations, tileSize,
            zoomD, offsetD.x, offsetD.y,
            zoom,  offset.x,  offset.y
        );
    }
}
