///// Otter: Central frame pipeline; timing + PBO mapping logged in one fixed ASCII line.
///// Schneefuchs: Headers/Sources in sync; no duplicate includes.
///// Maus: Performance logging default ON; epoch-millis, stable field order.
///// Datei: src/frame_pipeline.cpp

#include "pch.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

#include "renderer_resources.hpp"
#include "renderer_pipeline.hpp"
#include "cuda_interop.hpp"
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "zoom_command.hpp"
#include "zoom_logic.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "hud_text.hpp"
#include "fps_meter.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "common.hpp"
#include "settings.hpp"
#include "nacktmull_api.hpp"   // Host-Wrapper-API

namespace FramePipeline
{

// ------------------------------ TU-lokaler Zustand ----------------------------
static FrameContext g_ctx;
static CommandBus   g_zoomBus;
static int          g_frame = 0;

// Kleine Zoom-Stufe pro akzeptiertem Schritt
static constexpr float kZOOM_GAIN = 1.006f;

// Schneefuchs: lokale Perf-Zwischenspeicher (nur Host-Seite dieser TU)
namespace {
    using Clock = std::chrono::high_resolution_clock;
    using msd   = std::chrono::duration<double, std::milli>;

    constexpr int PERF_WARMUP_FRAMES = 30;
    constexpr int PERF_LOG_EVERY     = 30;

    double g_texMs      = 0.0;
    double g_ovlMs      = 0.0;
    double g_frameTotal = 0.0;

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

    // CUDA-Events früher hier genutzt – verbleiben ohne Schaden, falls später wieder erforderlich.
    static cudaEvent_t g_evStart = nullptr;
    static cudaEvent_t g_evStop  = nullptr;
    static bool        g_evInit  = false;
    static inline void ensureCudaEvents() {
        if (!g_evInit) {
            CUDA_CHECK(cudaEventCreate(&g_evStart));
            CUDA_CHECK(cudaEventCreate(&g_evStop));
            g_evInit = true;
        }
    }
} // anon ns

// --------------------------------- frame begin --------------------------------
static void beginFrame(FrameContext& fctx, RendererState& state) {
    (void)state;
    state.lastTimings.resetHostFrame();

    const double now = glfwGetTime();
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] beginFrame: time=%.4f, totalFrames=%d", now, g_frame);
    }

    const double delta = now - fctx.totalTime;
    fctx.frameTime          = (delta < 0.001) ? 0.001f : static_cast<float>(delta);
    fctx.totalTime          = now;
    fctx.timeSinceLastZoom += static_cast<float>(fctx.frameTime);
    fctx.shouldZoom         = false;
    fctx.newOffset          = fctx.offset;
    ++g_frame;
}

// ------------------------------- CUDA + analysis ------------------------------
// Dünner Wrapper: delegiert an NacktmullAPI (Host-Seite).
static void computeCudaFrame(FrameContext& fctx, RendererState& state) {
    // NacktmullAPI nimmt state.renderStream und leitet ihn bis in CudaInterop weiter.
    NacktmullAPI::computeCudaFrame(fctx, state);
}

// ------------------------------- apply zoom step ------------------------------
// Thin wrapper: delegiert an zoom_command.cpp mit FrameIndex/Gain.
static void applyZoomStep(FrameContext& fctx, CommandBus& bus) {
    extern void buildAndPushZoomCommand(FrameContext& fctx, CommandBus& bus, int frameIndex, float zoomGain);
    buildAndPushZoomCommand(fctx, bus, g_frame, kZOOM_GAIN);
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
        OpenGLUtils::peekPBO(state.currentPBO().id());
    }

    OpenGLUtils::setGLResourceContext("draw");

    if (!state.skipUploadThisFrame) {
        OpenGLUtils::updateTextureFromPBO(state.currentPBO().id(), state.tex.id(), fctx.width, fctx.height);
        if (state.pboFence[state.pboIndex]) { glDeleteSync(state.pboFence[state.pboIndex]); state.pboFence[state.pboIndex]=0; }
        state.pboFence[state.pboIndex] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZK][UP] fence set pbo=%u ring=%d", state.currentPBO().id(), state.pboIndex);
        }
    } else {
        if constexpr (Settings::performanceLogging) {
            LUCHS_LOG_HOST("[ZK][UP] no-upload this frame");
        }
    }

    RendererPipeline::drawFullscreenQuad(state.tex.id());

    if (s_scissorPrev) glEnable(GL_SCISSOR_TEST);

    const auto t1 = Clock::now();
    g_texMs = std::chrono::duration_cast<msd>(t1 - t0).count();
    state.lastTimings.uploadMs = g_texMs;

    const auto tOv0 = Clock::now();

    if (fctx.overlayActive) {
        HeatmapOverlay::drawOverlay(fctx.h_entropy, fctx.h_contrast,
                                    fctx.width, fctx.height, fctx.tileSize,
                                    state.tex.id(), state);
    }

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

    g_ctx.width         = state.width;
    g_ctx.height        = state.height;
    g_ctx.zoom          = static_cast<float>(state.zoom);
    g_ctx.offset        = make_float2((float)state.center.x, (float)state.center.y);
    g_ctx.maxIterations = state.maxIterations;

    if constexpr (Settings::Kolibri::gridScreenConstant) {
        const int desired = std::max(1, Settings::Kolibri::desiredTilePx);
        const int tilesX  = (g_ctx.width  + desired - 1) / desired;
        const int tilesY  = (g_ctx.height + desired - 1) / desired;
        const int safeX   = std::max(1, tilesX);
        const int safeY   = std::max(1, tilesY);
        const int tileW   = g_ctx.width  / safeX;
        const int tileH   = g_ctx.height / safeY;
        int tilePx        = std::min(tileW, tileH);
        tilePx            = std::max(8, std::min(tilePx, 256));
        g_ctx.tileSize    = tilePx;

        static int s_prevTilePx = -1;
        if (Settings::performanceLogging && tilePx != s_prevTilePx) {
            const int logTilesX = (g_ctx.width  + tilePx - 1) / tilePx;
            const int logTilesY = (g_ctx.height + tilePx - 1) / tilePx;
            LUCHS_LOG_HOST("[GRID] screen-const tilePx=%d tiles=%dx%d res=%dx%d",
                           tilePx, logTilesX, logTilesY, g_ctx.width, g_ctx.height);
            s_prevTilePx = tilePx;
        }
    } else {
        g_ctx.tileSize = computeTileSizeFromZoom(g_ctx.zoom);
    }

    g_ctx.overlayActive = state.heatmapOverlayEnabled;

    // Compute (ausgelagert)
    computeCudaFrame(g_ctx, state);

    // Zoom-Command
    applyZoomStep(g_ctx, g_zoomBus);

    // Ergebnisse zurück nach State
    state.zoom   = g_ctx.zoom;
    state.center = { (double)g_ctx.offset.x, (double)g_ctx.offset.y };

    // HUD/Text
    state.warzenschweinText = HudText::build(g_ctx, state);
    WarzenschweinOverlay::setText(state.warzenschweinText);

    // Draw
    drawFrame(g_ctx, state);

    const auto tFrame1 = Clock::now();
    g_frameTotal = std::chrono::duration_cast<msd>(tFrame1 - tFrame0).count();
    state.lastTimings.frameTotalMs = g_frameTotal;

    FpsMeter::updateCoreMs(g_frameTotal);

    if (perfShouldLog(g_frame)) {
        // Device-Log flushen (ASCII-only, deterministic)
        LuchsLogger::flushDeviceLogToHost(0);

        // Felder sammeln (stabile Reihenfolge)
        const long long tEpoch = epochMillisNow();
        const int    resX = g_ctx.width;
        const int    resY = g_ctx.height;
        const int    it   = g_ctx.maxIterations;
        const double fps  = (g_ctx.frameTime > 0.0f) ? (1.0 / g_ctx.frameTime) : 0.0;

        const bool   vt     = state.lastTimings.valid;
        const double mapMs  = vt ? state.lastTimings.pboMap          : 0.0; // PBO map/unmap
        const double mandMs = vt ? state.lastTimings.mandelbrotTotal : 0.0;
        const double entMs  = vt ? state.lastTimings.entropy         : 0.0;
        const double conMs  = vt ? state.lastTimings.contrast        : 0.0;

        const double txMs   = g_texMs;
        const double ovMs   = g_ovlMs;
        const double ttMs   = g_frameTotal;

        const double e0     = (double)g_ctx.lastEntropy;
        const double c0     = (double)g_ctx.lastContrast;

        const int    ringIx = state.pboIndex;
        const unsigned pbo  = state.currentPBO().id();
        const unsigned tex  = state.tex.id();

        const int maxfps    = FpsMeter::currentMaxFpsInt();
        const int mallocs   = 0;
        const int frees     = 0;
        const int dflush    = 1;

        // Eine feste ASCII-Zeile, epoch-millis zuerst, dann stabil sortierte Felder
        LUCHS_LOG_HOST(
            "[PERF] t=%lld f=%d r=%dx%d zm=%.6g it=%d fps=%.1f mx=%d ma=%d fr=%d df=%d map=%.2f md=%.2f en=%.2f ct=%.2f tx=%.2f ov=%.2f tt=%.2f e0=%.4f c0=%.4f ring=%d pbo=%u tex=%u",
            tEpoch, g_frame, resX, resY, (double)g_ctx.zoom, it, fps, maxfps, mallocs, frees, dflush,
            mapMs, mandMs, entMs, conMs, txMs, ovMs, ttMs, e0, c0, ringIx, pbo, tex
        );
    }
}

} // namespace FramePipeline
