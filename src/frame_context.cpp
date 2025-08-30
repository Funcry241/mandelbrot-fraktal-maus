// Datei: src/frame_context.cpp
// 🐜 Schwarze Ameise: Saubere Trennung von FrameContext-Daten und Logik.
// 🦦 Otter: Methoden ausgelagert, um Header schlank zu halten.
// 🦊 Schneefuchs: Debug-Logging und Reset zentral implementiert.

#include "frame_context.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"

FrameContext::FrameContext()
: width(0)
, height(0)
, maxIterations(Settings::INITIAL_ITERATIONS)
, tileSize(Settings::BASE_TILE_SIZE)
, zoom(Settings::initialZoom)
, offset{0.0f, 0.0f}
, pauseZoom(false)
, shouldZoom(false)
, d_entropy(nullptr)
, d_contrast(nullptr)
, d_iterations(nullptr)
, lastEntropy(0.0f)
, lastContrast(0.0f)
, overlayActive(false)
, frameTime(0.0)
, totalTime(0.0)
, timeSinceLastZoom(0.0)
{
    // Hostseitige Vektoren h_entropy und h_contrast sind per Default leer.
    // Initialisierung erfolgt später, wenn tileSize und Bildgröße bekannt sind.
}

void FrameContext::clear() noexcept {
    // 🧹 Buffer zurücksetzen bei Resize oder Reset
    h_entropy.clear();
    h_contrast.clear();

    // Device-Zeiger auf nullptr setzen, ohne Freigabe (muss extern erfolgen)
    d_entropy = nullptr;
    d_contrast = nullptr;
    d_iterations = nullptr;

    // Zoom-Flags zurücksetzen
    shouldZoom = false;
}

void FrameContext::printDebug() const noexcept {
    if constexpr (!Settings::debugLogging) return;

    // 📣 Wichtige Statusinformationen zum Frame ausgeben (ASCII-only)
    LUCHS_LOG_HOST("[Frame] width=%d height=%d zoom=%.5f offset=(%.5f, %.5f) tileSize=%d",
                   width, height, zoom, offset.x, offset.y, tileSize);
}
