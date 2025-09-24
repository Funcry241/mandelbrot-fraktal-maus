///// Otter: Central frame pipeline; timing + PBO mapping logged in one fixed ASCII line.
///// Schneefuchs: Headers/Sources in sync; no duplicate includes.
///// Maus: Performance logging default ON; epoch-millis, stable field order.
///// Datei: src/frame_pipeline.cpp

#include "pch.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>     // snprintf for dynamic ring logging
#include <algorithm>

#include "renderer_resources.hpp"
#include "renderer_pipeline.hpp"
#include "cuda_interop.hpp"   // CudaInterop::renderCudaFrame(RendererState&, FrameContext&, float&, float&)
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "zoom_logic.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "hud_text.hpp"
#include "fps_meter.hpp"
#include "luchs_log_host.hpp"
#include "common.hpp"
#include "settings.hpp"

namespace FramePipeline
{

// Sicherstellen, dass Ringgröße konsistent konfiguriert ist (keine OOBs)
static_assert(RendererState::kPboRingSize == Settings::pboRingSize,
              "RendererState::kPboRingSize must match Settings::pboRingSize");

// ------------------------------ TU-lokaler Zustand ----------------------------
static FrameContext         g_ctx;
static ZoomLogic::ZoomState g_zoomState;
static int                  g_frame = 0;

// Kleine Zoom-Stufe pro akzeptiertem Schritt
static constexpr float kZOOM_GAIN = 1.006f;

// Schneefuchs: lokale Perf-Zwischenspeicher (nur Host-Seite dieser TU)
namespace {
    using Clock = std::chrono::high_resolution_clock;
    using msd   = std::chrono::duration<double, std::milli>;

    constexpr int PERF_WARMUP_FRAMES = 30;
    constexpr int PERF_LOG_EVERY     = 30;
    constexpr int RING_LOG_EVERY     = 120;

    static double g_mapMs  = 0.0;
    static double g_mandMs = 0.0;
    static double g_entMs  = 0.0;
    static double g_conMs  = 0.0;
    static double g_texMs  = 0.0;
    static double g_ovlMs  = 0.0;
    static double g_totMs  = 0.0;

    inline bool perfShouldLog(int frameIdx) {
        if constexpr (Settings::performanceLogging) {
            if (frameIdx <= PERF_WARMUP_FRAMES) return false;
            return (frameIdx % PERF_LOG_EVERY) == 0;
        } else {
            (void)frameIdx;
            return false;
        }
    }

    // Epoch-Millis für stabile Zeitachse in Logs
    inline long long epochMillisNow() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    // [GLS] Einmalige Scissor-State-Ermittlung statt pro Frame abzufragen
    static bool       s_scissorInit = false;
    static GLboolean  s_scissorPrev = GL_FALSE;

    // Zeitdifferenz (lokal; FrameContext trägt diese Felder nicht mehr)
    static double s_prevNow = 0.0;
    inline double nextDeltaSeconds() {
        const double now = glfwGetTime();
        double delta = (s_prevNow > 0.0) ? (now - s_prevNow) : (1.0 / 60.0);
        s_prevNow = now;
        if (delta < 0.001) delta = 0.001;
        return delta;
    }

    // Dynamisches Ring-Logging (ASCII, determiniert), plus Reset der Zähler
    inline void logAndResetRingStats(RendererState& state) {
        // Ring-Use kompakt als "{a,b,c,...}" bauen
        char buf[1024];
        int  pos = 0;
        pos += std::snprintf(buf + pos, sizeof(buf) - pos, "{");
        for (int i = 0; i < RendererState::kPboRingSize; ++i) {
            pos += std::snprintf(buf + pos, sizeof(buf) - pos, (i == 0 ? "%u" : ",%u"), state.ringUse[i]);
        }
        std::snprintf(buf + pos, sizeof(buf) - pos, "}");

        LUCHS_LOG_HOST("[RING] use=%s skip=%u size=%d",
                       buf, state.ringSkip, RendererState::kPboRingSize);

        // Reset Counters
        for (int i = 0; i < RendererState::kPboRingSize; ++i) state.ringUse[i] = 0;
        state.ringSkip = 0;
    }
} // anon ns

// --------------------------------- frame begin --------------------------------
static void beginFrame(FrameContext& fctx, RendererState& state) {
    (void)state;
    state.lastTimings.resetHostFrame();

    if constexpr (Settings::debugLogging) {
        const double now = glfwGetTime();
        LUCHS_LOG_HOST("[PIPE] beginFrame: time=%.4f, totalFrames=%d", now, g_frame);
    }

    // Reset Zoom-Flags pro Frame (Zeitpflege lokal)
    (void)nextDeltaSeconds();
    fctx.shouldZoom = false;
    fctx.newOffset  = fctx.offset;
    ++g_frame;
}

// ------------------------------- CUDA (Compute) -------------------------------
static void computeCudaFrame(FrameContext& fctx, RendererState& state) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] compute begin: tile=%d it=%d zoom=%.6f",
                       fctx.tileSize, fctx.maxIterations, (double)fctx.zoom);
    }

    // Übergibt REFERENZEN auf newOffset.{x,y}, damit CUDA den neuen Mittelpunkt zurückschreibt.
    CudaInterop::renderCudaFrame(state, fctx, fctx.newOffset.x, fctx.newOffset.y);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] compute end: newOffset=(%.9f,%.9f)",
                       (double)fctx.newOffset.x, (double)fctx.newOffset.y);
    }
}

// ------------------------------ draw (GL upload + FSQ) -----------------------
static void drawFrame(FrameContext& fctx, RendererState& state) {
    const auto t0 = Clock::now();

    if (fctx.width <= 0 || fctx.height <= 0) [[unlikely]] return;

    glViewport(0, 0, fctx.width, fctx.height);

    if (!s_scissorInit) {
        s_scissorPrev = glIsEnabled(GL_SCISSOR_TEST);
        s_scissorInit = true;
    }
    if (s_scissorPrev) [[unlikely]] glDisable(GL_SCISSOR_TEST);

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] drawFrame begin: tex=%u pbo=%u %dx%d",
                       state.tex.id(), state.currentPBO().id(), fctx.width, fctx.height);
    }

    OpenGLUtils::setGLResourceContext("draw");

    // --- Texture upload (measure) ---
    if (!state.skipUploadThisFrame) {
        OpenGLUtils::updateTextureFromPBO(state.currentPBO().id(), state.tex.id(), fctx.width, fctx.height);
        if (state.pboFence[state.pboIndex]) { glDeleteSync(state.pboFence[state.pboIndex]); state.pboFence[state.pboIndex]=0; }
        state.pboFence[state.pboIndex] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZK][UP] fence set pbo=%u ring=%d", state.currentPBO().id(), state.pboIndex);
        }
    } else {
        state.skipUploadThisFrame = false;
        ++state.ringSkip;
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZK][UP] skip upload pbo=%u ring=%d", state.currentPBO().id(), state.pboIndex);
        }
    }
    const auto tUploadEnd = Clock::now();
    g_texMs = std::chrono::duration_cast<msd>(tUploadEnd - t0).count();

    // Heatmap Overlay (falls aktiv) – Host-Daten liegen im RendererState
    const auto tOv0 = Clock::now();
    if (state.heatmapOverlayEnabled) {
        // Overlay-Raster darf von Compute-Tiles abweichen, ohne die Datengröße zu verletzen.
        const int overlayTilePx = (Settings::Kolibri::gridScreenConstant)
                                  ? Settings::Kolibri::desiredTilePx
                                  : fctx.tileSize;

        if constexpr (Settings::performanceLogging) {
            const int compPx   = fctx.tileSize;
            const int ovTx     = (fctx.width  + overlayTilePx - 1) / overlayTilePx;
            const int ovTy     = (fctx.height + overlayTilePx - 1) / overlayTilePx;
            const int compTx   = (fctx.width  + compPx - 1) / compPx;
            const int compTy   = (fctx.height + compPx - 1) / compPx;
            if (overlayTilePx != compPx) {
                LUCHS_LOG_HOST("[GRID] overlayPx=%d overlay=%dx%d computePx=%d compute=%dx%d res=%dx%d",
                               overlayTilePx, ovTx, ovTy, compPx, compTx, compTy, fctx.width, fctx.height);
            }
        }

        HeatmapOverlay::drawOverlay(state.h_entropy, state.h_contrast,
                                    fctx.width, fctx.height, overlayTilePx,
                                    state.tex.id(), state);
    }

    // HUD-Text via Warzenschwein-Overlay (Text kommt aus state.warzenschweinText)
    if constexpr (Settings::warzenschweinOverlayEnabled) {
        if (!state.warzenschweinText.empty()) {
            WarzenschweinOverlay::drawOverlay(static_cast<float>(state.zoom));
        }
    }

    const auto tOv1 = Clock::now();
    g_ovlMs = std::chrono::duration_cast<msd>(tOv1 - tOv0).count();
    state.lastTimings.overlaysMs = g_ovlMs;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] drawFrame end: texMs=%.3f ovMs=%.3f", g_texMs, g_ovlMs);
    }
}

// ---------------------------------- execute ----------------------------------
void execute(RendererState& state) {
    const auto tFrame0 = Clock::now();

    beginFrame(g_ctx, state);

    // Kontext aus RendererState übernehmen
    g_ctx.width         = state.width;
    g_ctx.height        = state.height;
    g_ctx.maxIterations = state.maxIterations;
    g_ctx.zoom          = static_cast<float>(state.zoom);
    g_ctx.offset        = make_float2(static_cast<float>(state.center.x),
                                      static_cast<float>(state.center.y));

    // Compute-TileSize wird zentral bestimmt (Ameise): IMMER computeTileSizeFromZoom()
    // Overlay kann separat ein screen-konstantes Raster loggen/nutzen, ohne Compute zu beeinflussen.
    g_ctx.tileSize = computeTileSizeFromZoom(g_ctx.zoom);

    if constexpr (Settings::Kolibri::gridScreenConstant) {
        // Nur Logging der Overlay-Rasterung (kein Einfluss auf Compute-Tiles!)
        static int s_prevOverlayPx = -1;
        const int overlayPx = Settings::Kolibri::desiredTilePx;
        if constexpr (Settings::performanceLogging) {
            if (s_prevOverlayPx != overlayPx) {
                const int ovTx   = (g_ctx.width  + overlayPx - 1) / overlayPx;
                const int ovTy   = (g_ctx.height + overlayPx - 1) / overlayPx;
                const int compPx = g_ctx.tileSize;
                const int cTx    = (g_ctx.width  + compPx - 1) / compPx;
                const int cTy    = (g_ctx.height + compPx - 1) / compPx;
                LUCHS_LOG_HOST("[GRID] screen-const tilePx=%d tiles=%dx%d | computePx=%d tiles=%dx%d res=%dx%d",
                               overlayPx, ovTx, ovTy, compPx, cTx, cTy, g_ctx.width, g_ctx.height);
                s_prevOverlayPx = overlayPx;
            }
        }
    }

    // Compute (ausgelagert; schreibt newOffset.{x,y})
    computeCudaFrame(g_ctx, state);

    // Optional Zoom evaluieren (Pause global via CudaInterop)
    if (!CudaInterop::getPauseZoom()) {
        ZoomLogic::evaluateAndApply(g_ctx, state, g_zoomState, kZOOM_GAIN);
    }

    // HUD-Text für Warzenschwein-Overlay vorbereiten
    if constexpr (Settings::warzenschweinOverlayEnabled) {
        state.warzenschweinText = HudText::build(g_ctx, state);
    }

    // Draw
    drawFrame(g_ctx, state);

    // Host-Perf (gesamt)
    const auto tFrame1 = Clock::now();
    g_totMs = std::chrono::duration_cast<msd>(tFrame1 - tFrame0).count();
    state.lastTimings.frameTotalMs = g_totMs;

    if (perfShouldLog(g_frame)) {
        const long long tEpoch = epochMillisNow();
        const int resX = g_ctx.width, resY = g_ctx.height;
        const int it   = g_ctx.maxIterations;
        const double fps    = (g_totMs > 1e-3) ? (1000.0 / g_totMs) : 0.0;
        const double maxfps = (g_texMs > 1e-3) ? (1000.0 / g_texMs) : 0.0;

        // Build stable log line
        const float e0  = state.h_entropy.empty()  ? 0.f : state.h_entropy[0];
        const float c0  = state.h_contrast.empty() ? 0.f : state.h_contrast[0];
        const int   ringIx = state.pboIndex;
        const unsigned pbo = state.currentPBO().id();
        const unsigned tex = state.tex.id();
        const int dflush = 0;

        LUCHS_LOG_HOST(
            "[PERF] t=%lld frame=%d res=%dx%d zoom=%.6f it=%d fps=%.1f maxfps=%.1f mallocs=%d frees=%d dflush=%d map=%.2f mand=%.2f ent=%.2f con=%.2f tex=%.2f ovl=%.2f tot=%.2f e0=%.4f c0=%.4f ring=%d skip=%d pbo=%u tex=%u",
            tEpoch, g_frame, resX, resY, (double)g_ctx.zoom, it, fps, maxfps, 0, 0, dflush,
            g_mapMs, g_mandMs, g_entMs, g_conMs, g_texMs, g_ovlMs, g_totMs, e0, c0, ringIx, (int)state.skipUploadThisFrame, pbo, tex
        );
    }

    // [LOG-6] Periodische Ring-Nutzungsstatistik (dynamisch gemäß Ringgröße)
    if constexpr (Settings::performanceLogging) {
        if ((g_frame % RING_LOG_EVERY) == 0) {
            logAndResetRingStats(state);
        }
    }
}

} // namespace FramePipeline
