#include "pch.hpp"
#include <chrono>
#include <algorithm>
#include <cstdio>     // snprintf for dynamic ring logging
#include <cmath>      // sqrt

#include "renderer_resources.hpp"
#include "renderer_pipeline.hpp"
#include "cuda_interop.hpp"
#include "frame_context.hpp"
#include "frame_pipeline.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "hud_text.hpp"
#include "zoom_logic.hpp"
#include "common.hpp"
#include "fps_meter.hpp"      // Maus: feeds FpsMeter once per frame

#include <vector_types.h>
#include <vector_functions.h>

static_assert(Settings::pboRingSize == RendererState::kPboRingSize, "pboRingSize must match Settings::pboRingSize");

// ------------------------------ TU-lokaler Zustand ----------------------------
static FrameContext         g_ctx;
static ZoomLogic::ZoomState g_zoomState;
static int                  g_frame = 0;

static constexpr float kZOOM_GAIN = 1.006f;

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

    inline long long epochMillisNow() {
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    static void tickFrameTime() {
        static double last = 0.0;
        const double now = glfwGetTime();
        last = now;
    }

    static void beginFrameLocal() {
        if constexpr (Settings::debugLogging) {
            const double now = glfwGetTime();
            LUCHS_LOG_HOST("[PIPE] beginFrame: time=%.4f, totalFrames=%d", now, g_frame);
        }
        tickFrameTime();
        g_ctx.shouldZoom = false;
        g_ctx.newOffset  = g_ctx.offset;
        g_ctx.newOffsetD = g_ctx.offsetD;  // NEU: Double-Spiegel mitführen
        ++g_frame;
    }

    // Fallback erzeugen, wenn (noch) keine Daten vorliegen.
    static void ensureHeatmapHostData(RendererState& state, int width, int height, int tilePx) {
        const int px = std::max(1, tilePx);
        const int tx = (width  + px - 1) / px;
        const int ty = (height + px - 1) / px;
        const size_t N = static_cast<size_t>(tx) * static_cast<size_t>(ty);

        const bool needEnt = state.h_entropy.size()  != N || state.h_entropy.empty();
        const bool needCon = state.h_contrast.size() != N || state.h_contrast.empty();
        if (!needEnt && !needCon) return;

        if (needEnt) state.h_entropy.assign(N, 0.0f);
        if (needCon) state.h_contrast.assign(N, 0.0f);

        for (int y = 0; y < ty; ++y) {
            for (int x = 0; x < tx; ++x) {
                const size_t i = static_cast<size_t>(y) * tx + x;
                const float fx = (tx > 1) ? (float)x / (float)(tx - 1) : 0.0f;
                const float fy = (ty > 1) ? (float)y / (float)(ty - 1) : 0.0f;
                const float r  = std::min(1.0f, std::sqrt(fx*fx + fy*fy));
                state.h_entropy[i]  = 0.15f + 0.8f * r;

                const int  checker = ((x ^ y) & 1);
                const float mix    = 0.3f + 0.7f * ((fx + (1.0f - fy)) * 0.5f);
                state.h_contrast[i] = checker ? mix : (1.0f - mix);
            }
        }

        LUCHS_LOG_HOST("[HM][FALLBACK] generated N=%zu tiles=%dx%d tilePx=%d", N, tx, ty, px);
    }

    // ------------------------------- CUDA (Compute) -------------------------------
    static void computeCudaFrame(FrameContext& fctx, RendererState& state) {
        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] compute begin: tile=%d it=%d zoom=%.6f",
                           fctx.tileSize, fctx.maxIterations, (double)fctx.zoom);
        }

        // NEU: Vereinheitlichter Render-Call (Convenience-Overload, Float-Spiegel)
        CudaInterop::renderCudaFrame(state, fctx, fctx.newOffset.x, fctx.newOffset.y);

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] compute end");
        }

        const auto t0 = Clock::now();
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

        RendererPipeline::drawFullscreenQuad(state.tex.id());

        // ---------- GL-STATE für Overlays ----------
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fctx.width, fctx.height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_STENCIL_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const auto tOv0 = Clock::now();

        // Heatmap Overlay (falls aktiv)
        if (state.heatmapOverlayEnabled) {
            const int overlayTilePx = std::max(1, (Settings::Kolibri::gridScreenConstant)
                                                  ? Settings::Kolibri::desiredTilePx
                                                  : fctx.tileSize);

            // NEU: GPU-Metriken; bei Fehler auf Fallback
            if (!CudaInterop::buildHeatmapMetrics(state, fctx.width, fctx.height, overlayTilePx, state.renderStream)) {
                ensureHeatmapHostData(state, fctx.width, fctx.height, overlayTilePx);
            }

            if constexpr (Settings::performanceLogging) {
                const int compPx = std::max(1, fctx.tileSize);
                const int ovTx   = (fctx.width  + overlayTilePx - 1) / overlayTilePx;
                const int ovTy   = (fctx.height + overlayTilePx - 1) / overlayTilePx;
                const int compTx = (fctx.width  + compPx - 1) / compPx;
                const int compTy = (fctx.height + compPx - 1) / compPx;
                LUCHS_LOG_HOST("[GRID] overlayPx=%d overlay=%dx%d computePx=%d compute=%dx%d res=%dx%d",
                               overlayTilePx, ovTx, ovTy, compPx, compTx, compTy, fctx.width, fctx.height);
            }

            HeatmapOverlay::drawOverlay(state.h_entropy, state.h_contrast,
                                        fctx.width, fctx.height, overlayTilePx,
                                        state.tex.id(), state);
        }

        // HUD danach → on top
        if constexpr (Settings::warzenschweinOverlayEnabled) {
            WarzenschweinOverlay::setText(state.warzenschweinText);
            WarzenschweinOverlay::drawOverlay(fctx.zoom);
        }

        const auto tOv1 = Clock::now();
        g_ovlMs = std::chrono::duration_cast<msd>(tOv1 - tOv0).count();
        state.lastTimings.overlaysMs = g_ovlMs;

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] drawFrame end: texMs=%.3f ovMs=%.3f", g_texMs, g_ovlMs);
        }
    }

    static void logAndResetRingStats(RendererState& state) {
        char buf[256]; int pos = 0;
        pos += std::snprintf(buf + pos, sizeof(buf) - pos, "{");
        for (int i = 0; i < RendererState::kPboRingSize; ++i) {
            pos += std::snprintf(buf + pos, sizeof(buf) - pos, (i == 0 ? "%u" : ",%u"), state.ringUse[i]);
        }
        std::snprintf(buf + pos, sizeof(buf) - pos, "}");
        LUCHS_LOG_HOST("[RING] use=%s skip=%u size=%d", buf, state.ringSkip, RendererState::kPboRingSize);

        for (int i = 0; i < RendererState::kPboRingSize; ++i) state.ringUse[i] = 0;
        state.ringSkip = 0;
    }

    inline int chooseComputeTileSize(float zoom) {
        int t = computeTileSizeFromZoom(zoom);
        t = std::clamp(t, Settings::MIN_TILE_SIZE, Settings::MAX_TILE_SIZE);
        if (t % 32 != 0) {
            const int up   = ((t + 31) / 32) * 32;
            const int down = (t / 32) * 32;
            int cand = (std::abs(up - t) < std::abs(t - down)) ? up : down;
            if (cand < 32) cand = 32;
            t = cand;
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[GRID] align compute tile from %d to %d", (int)computeTileSizeFromZoom(zoom), t);
            }
        }
        return t;
    }
} // anon ns

namespace FramePipeline {
void execute(RendererState& state) {
    const auto tFrame0 = Clock::now();

    beginFrameLocal();

    // ---- Autoritative Double-Werte aus dem RendererState spiegeln ----
    g_ctx.width         = state.width;
    g_ctx.height        = state.height;
    g_ctx.maxIterations = state.maxIterations;
    g_ctx.zoomD         = state.zoom;                       // double Quelle der Wahrheit
    g_ctx.offsetD       = { state.center.x, state.center.y };
    g_ctx.syncFloatFromDouble();                            // float-Spiegel aktualisieren

    g_ctx.tileSize = chooseComputeTileSize(g_ctx.zoom);

    if constexpr (Settings::Kolibri::gridScreenConstant) {
        static int s_prevOverlayPx = -1;
        const int overlayPx = Settings::Kolibri::desiredTilePx;
        if constexpr (Settings::performanceLogging) {
            if (s_prevOverlayPx != overlayPx) {
                const int px = std::max(1, overlayPx);
                const int ts = std::max(1, g_ctx.tileSize);

                const int ovTx = (g_ctx.width  + px - 1) / px;
                const int ovTy = (g_ctx.height + px - 1) / px;
                const int cTx  = (g_ctx.width  + ts - 1) / ts;
                const int cTy  = (g_ctx.height + ts - 1) / ts;

                LUCHS_LOG_HOST("[GRID] overlayPx=%d overlay=%dx%d computePx=%d tiles=%dx%d res=%dx%d",
                               px, ovTx, ovTy, ts, cTx, cTy, g_ctx.width, g_ctx.height);
                s_prevOverlayPx = overlayPx;
            }
        }
    }

    if constexpr (Settings::warzenschweinOverlayEnabled) {
        state.warzenschweinText = HudText::build(g_ctx, state);
    }

    computeCudaFrame(g_ctx, state);

    if (!CudaInterop::getPauseZoom()) {
        ZoomLogic::evaluateAndApply(g_ctx, state, g_zoomState, kZOOM_GAIN);
    }

    if (!CudaInterop::getPauseZoom() &&
        (state.h_entropy.empty() || state.h_contrast.empty()))
    {
        g_ctx.shouldZoom = true;
        g_ctx.newOffset  = g_ctx.offset;
        g_ctx.newOffsetD = g_ctx.offsetD; // Double-Spiegel ebenfalls setzen
    }

    if (g_ctx.shouldZoom) {
        // ---- Anwenden in double, dann float-Spiegel aktualisieren ----
        state.center.x = g_ctx.newOffsetD.x;
        state.center.y = g_ctx.newOffsetD.y;
        state.zoom     *= (double)kZOOM_GAIN;

        g_ctx.offsetD  = g_ctx.newOffsetD;
        g_ctx.zoomD    = state.zoom;
        g_ctx.syncFloatFromDouble();

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[ZOOM][APPLY] center=(%.9f,%.9f) zoom=%.6f gain=%.6f",
                           state.center.x, state.center.y, (double)state.zoom, (double)kZOOM_GAIN);
        }
    }

    const auto tFrame1 = Clock::now();
    g_totMs = std::chrono::duration_cast<msd>(tFrame1 - tFrame0).count();
    state.lastTimings.frameTotalMs = g_totMs;

    // Maus: Füttere den FpsMeter einmal pro Frame,
    //       damit der HUD-Klammerwert (max FPS, capped auf 60) != 0 ist.
    FpsMeter::updateCoreMs(g_totMs);

    if (perfShouldLog(g_frame)) {
        const long long tEpoch = epochMillisNow();
        const int resX = g_ctx.width, resY = g_ctx.height;
        const int it   = g_ctx.maxIterations;
        const double fps    = (g_totMs > 1e-3) ? (1000.0 / g_totMs) : 0.0;
        const double maxfps = (g_texMs > 1e-3) ? (1000.0 / g_texMs) : 0.0;

        const float e0 = state.h_entropy.empty()  ? 0.f : state.h_entropy[0];
        const float c0 = state.h_contrast.empty() ? 0.f : state.h_contrast[0];
        const int   ringIx = state.pboIndex;
        const unsigned pbo = state.currentPBO().id();
        const unsigned tex = state.tex.id();

        LUCHS_LOG_HOST(
            "[PERF] t=%lld frame=%d res=%dx%d zoom=%.6f it=%d fps=%.2f maxfps=%.2f map=%.2f mand=%.2f ent=%.2f con=%.2f up=%.2f ovl=%.2f tot=%.2f e0=%.4f c0=%.4f ring=%d skip=%d pbo=%u tex=%u",
            tEpoch, g_frame, resX, resY, (double)g_ctx.zoom, it, fps, maxfps,
            g_mapMs, g_mandMs, g_entMs, g_conMs, g_texMs, g_ovlMs, g_totMs, e0, c0, ringIx, (int)state.skipUploadThisFrame, pbo, tex
        );
    }

    if constexpr (Settings::performanceLogging) {
        if ((g_frame % RING_LOG_EVERY) == 0) {
            logAndResetRingStats(state);
        }
    }
}
} // namespace FramePipeline
