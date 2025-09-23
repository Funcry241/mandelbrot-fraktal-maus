///// Otter: C4702 fix – logging compiled-in only when enabled; no unreachable code.
///// Schneefuchs: /WX strikt, ASCII-only Logs/Kommentare; Verhalten unverändert.
///// Maus: Saubere Trennung von FrameContext-Daten und Logik (Ameise).
///// Datei: src/frame_context.cpp

#include "frame_context.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

FrameContext::FrameContext()
{
    width         = 0;
    height        = 0;
    maxIterations = Settings::INITIAL_ITERATIONS;
    tileSize      = Settings::BASE_TILE_SIZE;
    zoom          = Settings::initialZoom;
    offset        = { 0.0f, 0.0f };
    newOffset     = offset;
    shouldZoom    = false;
}

void FrameContext::clear() noexcept {
    // Nur die Zoom-Entscheidung zurücksetzen; Pufferverwaltung ist im RendererState.
    shouldZoom = false;
    newOffset  = offset;
}

void FrameContext::printDebug() const noexcept {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[Frame] w=%d h=%d it=%d tile=%d zoom=%.6f off=(%.6f,%.6f)",
            width, height, maxIterations, tileSize, zoom, offset.x, offset.y);
    }
}
