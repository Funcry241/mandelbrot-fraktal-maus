///// Otter: C4702 fix – logging compiled-in only when enabled; no unreachable code.
///// Schneefuchs: /WX strikt, ASCII-only Logs/Kommentare; Verhalten unverändert.
///// Maus: Saubere Trennung von FrameContext-Daten und Logik (Ameise).

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
    // Host-side vectors h_entropy and h_contrast are empty by default.
    // Initialization happens later once tileSize and image size are known.
}

void FrameContext::clear() noexcept {
    // Reset host buffers on resize/reset
    h_entropy.clear();
    h_contrast.clear();

    // Reset device pointers; deallocation must be handled externally
    d_entropy    = nullptr;
    d_contrast   = nullptr;
    d_iterations = nullptr;

    // Reset zoom flags
    shouldZoom = false;
}

void FrameContext::printDebug() const noexcept {
    // C4702 fix: avoid 'return' before subsequent statements under if constexpr.
    // Compile the log only when debugLogging is enabled at compile time.
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST(
            "[Frame] width=%d height=%d zoom=%.5f offset=(%.5f, %.5f) tileSize=%d",
            width, height, zoom, offset.x, offset.y, tileSize
        );
    }
    // If debugLogging is false at compile time, this function compiles to an empty body.
}
