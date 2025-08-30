// Datei: src/frame_pipeline.cpp
///// Otter: Feste Aufrufreihenfolge – updateTextureFromPBO(PBO, TEX, W, H); ASCII-Logs; keine Compat-Wrapper.
///// Schneefuchs: /WX-fest; zusätzliche Diagnose (PBO-PEEK) vor Upload; RAII & deterministische Logs.
///// Maus: Eine Quelle für Tiles/Upload; Zoom-V2 außerhalb der CUDA-Interop bleibt unverändert.

#include "pch.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector_types.h>
#include <vector_functions.h> // make_float2
#include <GL/glew.h>          // <-- Viewport & PBO peek

#include "renderer_resources.hpp"    // OpenGLUtils::setGLResourceContext(const char*), OpenGLUtils::updateTextureFromPBO(GLuint pbo, GLuint tex, int w, int h)
#include "renderer_pipeline.hpp"     // RendererPipeline::drawFullscreenQuad(...)
#include "cuda_interop.hpp"          // CudaInterop::renderCudaFrame(...)
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

namespace FramePipeline
{

// ------------------------------ TU-lokaler Zustand ----------------------------
static FrameContext g_ctx;
static CommandBus   g_zoomBus;
static int          g_frame = 0;

// Kleine Zoom-Stufe pro akzeptiertem Schritt
static constexpr float kZOOM_GAIN = 1.006f;

// 🐑 Schneefuchs: lokale Perf-Zwischenspeicher (nur Host-Seite dieser TU)
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

    // --- Diagnose: GL-PBO kurz mappen und die ersten Bytes prüfen ---
    static void peekPBO(GLuint pbo) {
        GLint prev = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prev);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

        // Wir lesen nur einen winzigen Bereich, kein Risiko für Performance.
        const GLsizeiptr N = 64; // erste 64 Bytes
        void* ptr = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, N, GL_MAP_READ_BIT);
        if (ptr) {
            const unsigned char* b = static_cast<const unsigned char*>(ptr);
            unsigned sum16 = 0;
            for (int i = 0; i < 16 && i < N; ++i) sum16 += b[i];
            // erste vier Bytes getrennt loggen
            unsigned b0=b[0], b1=b[1], b2=b[2], b3=b[3];
            LUCHS_LOG_HOST("[PBO-PEEK] pbo=%u first4=%u,%u,%u,%u sum16=%u", pbo, b0, b1, b2, b3, sum16);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        } else {
            const GLenum err = glGetError();
            LUCHS_LOG_HOST("[PBO-PEEK][WARN] map failed for pbo=%u (glError=0x%04X)", pbo, err);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prev));
    }
}

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
static void computeCudaFrame(FrameContext& fctx, RendererState& state) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] computeCudaFrame: dimensions=%dx%d, zoom=%.5f, tileSize=%d",
                       fctx.width, fctx.height, fctx.zoom, fctx.tileSize);
    }

    const int tilesX   = (fctx.width  + fctx.tileSize - 1) / fctx.tileSize;
    const int tilesY   = (fctx.height + fctx.tileSize - 1) / fctx.tileSize;
    const int numTiles = tilesX * tilesY;

    if (fctx.tileSize <= 0 || numTiles <= 0) {
        LUCHS_LOG_HOST("[FATAL] computeCudaFrame: invalid tileSize=%d or numTiles=%d",
                       fctx.tileSize, numTiles);
        return;
    }

    state.setupCudaBuffers(fctx.tileSize);

    if constexpr (Settings::debugLogging) {
        const size_t totalPixels         = size_t(fctx.width) * size_t(fctx.height);
        const size_t need_it_bytes       = totalPixels * sizeof(int);
        const size_t need_entropy_bytes  = size_t(numTiles) * sizeof(float);
        const size_t need_contrast_bytes = size_t(numTiles) * sizeof(float);
        LUCHS_LOG_HOST("[SANITY] tiles=%d (%d x %d) pixels=%zu need(it=%zu entropy=%zu contrast=%zu) alloc(it=%zu entropy=%zu contrast=%zu)",
                       numTiles, tilesX, tilesY, totalPixels,
                       need_it_bytes, need_entropy_bytes, need_contrast_bytes,
                       state.d_iterations.size(), state.d_entropy.size(), state.d_contrast.size());
    }

    // Device-Render (Iterations -> Shade) + E/C-Analyse (erzeugt Host-Arrays)
    try {
        auto t0 = Clock::now();

        float2 gpuOffset    = make_float2((float)fctx.offset.x, (float)fctx.offset.y);
        float2 gpuNewOffset = gpuOffset;

        CudaInterop::renderCudaFrame(
            state.d_iterations,
            state.d_entropy,
            state.d_contrast,
            fctx.width,
            fctx.height,
            fctx.zoom,
            gpuOffset,
            fctx.maxIterations,
            fctx.h_entropy,
            fctx.h_contrast,
            gpuNewOffset,
            fctx.shouldZoom,
            fctx.tileSize,
            state
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t1 = Clock::now();
        const double ms = std::chrono::duration_cast<msd>(t1 - t0).count();

        if constexpr (Settings::debugLogging) {
            if (state.lastTimings.valid) {
                LUCHS_LOG_HOST(
                    "[FRAME] Mandelbrot=%.3f | Launch=%.3f | Sync=%.3f | Entropy=%.3f | Contrast=%.3f | LogFlush=%.3f | PBOMap=%.3f",
                    state.lastTimings.mandelbrotTotal,
                    state.lastTimings.mandelbrotLaunch,
                    state.lastTimings.mandelbrotSync,
                    state.lastTimings.entropy,
                    state.lastTimings.contrast,
                    state.lastTimings.deviceLogFlush,
                    state.lastTimings.pboMap
                );
            } else {
                LUCHS_LOG_HOST("[TIME] CUDA kernel + sync: %.3f ms", ms);
            }
        }
    } catch (const std::exception& ex) {
        // Weiterlaufen und später trotzdem versuchen zu zeichnen (PBO-PEEK hilft)
        LUCHS_LOG_HOST("[ERROR] renderCudaFrame threw: %s", ex.what());
        LuchsLogger::flushDeviceLogToHost(0);
    }

    // Device-Logs periodisch spülen
    {
        const cudaError_t err = cudaPeekAtLastError();
        if constexpr (Settings::debugLogging) {
            if (err != cudaSuccess || (g_frame % 30 == 0)) {
                LUCHS_LOG_HOST("[PIPE] Flushing device logs (err=%d, frame=%d)", (int)err, g_frame);
                LuchsLogger::flushDeviceLogToHost(0);
            }
        } else {
            if (err != cudaSuccess) {
                LuchsLogger::flushDeviceLogToHost(0);
            }
        }
    }

    // Zoom-Analyse (nur Hostdaten h_entropy/h_contrast)
    {
        const float2 currOff = make_float2((float)fctx.offset.x, (float)fctx.offset.y);
        const float2 prevOff = currOff;

        auto zr = ZoomLogic::evaluateZoomTarget(
            fctx.h_entropy,
            fctx.h_contrast,
            tilesX, tilesY,
            fctx.width, fctx.height,
            currOff, fctx.zoom,
            prevOff,
            state.zoomV2State
        );

        if (zr.bestIndex >= 0) {
            fctx.lastEntropy  = zr.bestEntropy;
            fctx.lastContrast = zr.bestContrast;
        } else {
            fctx.lastEntropy  = 0.0f;
            fctx.lastContrast = 0.0f;
        }

        fctx.shouldZoom = zr.shouldZoom;
        if (zr.shouldZoom) {
            fctx.newOffset = { zr.newOffset.x, zr.newOffset.y };
        }

        if constexpr (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] ZOOMV2: best=%d score=%.3f accept=%d newOff=(%.6f,%.6f)",
                           zr.bestIndex, zr.bestScore, zr.shouldZoom ? 1 : 0,
                           zr.newOffset.x, zr.newOffset.y);
        }
    }
}

// ------------------------------- apply zoom step ------------------------------
static void applyZoomStep(FrameContext& fctx, CommandBus& bus) {
    if (!fctx.shouldZoom) return;

    const double2 diff = { fctx.newOffset.x - fctx.offset.x, fctx.newOffset.y - fctx.offset.y };
    const float   prevZoom = fctx.zoom;

    fctx.offset = fctx.newOffset;
    fctx.zoom  *= kZOOM_GAIN;

    ZoomCommand cmd;
    cmd.frameIndex = g_frame;
    cmd.oldOffset  = make_float2((float)(fctx.offset.x - diff.x), (float)(fctx.offset.y - diff.y));
    cmd.zoomBefore = prevZoom;
    cmd.newOffset  = make_float2((float)fctx.newOffset.x, (float)fctx.newOffset.y);
    cmd.zoomAfter  = fctx.zoom;
    cmd.entropy    = fctx.lastEntropy;
    cmd.contrast   = fctx.lastContrast;

    bus.push(cmd);
    fctx.timeSinceLastZoom = 0.0f;
}

// ------------------------------ draw (GL upload + FSQ) -----------------------
static void drawFrame(FrameContext& fctx, RendererState& state) {
    const auto t0 = Clock::now();

    // 1) Viewport sicherstellen (häufige Weiß-Bild-Ursache)
    {
        GLint vp[4] = {0,0,0,0};
        glGetIntegerv(GL_VIEWPORT, vp);
        if (vp[2] != fctx.width || vp[3] != fctx.height) {
            glViewport(0, 0, fctx.width, fctx.height);
            if constexpr (Settings::debugLogging) {
                LUCHS_LOG_HOST("[GL] viewport set -> %dx%d (was %dx%d)",
                               fctx.width, fctx.height, vp[2], vp[3]);
            }
        }
    }

    // Diagnose: Einmal vor dem Upload ins PBO schauen
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] drawFrame begin: tex=%u pbo=%u %dx%d",
                       state.tex.id(), state.pbo.id(), fctx.width, fctx.height);
        peekPBO(state.pbo.id());
    }

    // **WICHTIG**: Stabile API & richtige Reihenfolge -> PBO vor Texture
    OpenGLUtils::setGLResourceContext("draw");
    OpenGLUtils::updateTextureFromPBO(state.pbo.id(), state.tex.id(), fctx.width, fctx.height);

    // Debug-Clear (nur im Debug): dunkler Hintergrund, damit FSQ sichtbar ist
    if constexpr (Settings::debugLogging) {
        glDisable(GL_SCISSOR_TEST);
        glClearColor(0.05f, 0.06f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    RendererPipeline::drawFullscreenQuad(state.tex.id());

    const auto t1 = Clock::now();
    g_texMs = std::chrono::duration_cast<msd>(t1 - t0).count();
    state.lastTimings.uploadMs = g_texMs;

    // Overlays separat messen
    const auto tOv0 = Clock::now();

    if (fctx.overlayActive) {
        HeatmapOverlay::drawOverlay(fctx.h_entropy, fctx.h_contrast,
                                    fctx.width, fctx.height, fctx.tileSize,
                                    state.tex.id(), state);
    }

    if constexpr (Settings::warzenschweinOverlayEnabled) {
        if (!state.warzenschweinText.empty()) {
            WarzenschweinOverlay::drawOverlay(state);
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

    // View-Parameter aus State in lokalen FrameContext spiegeln
    g_ctx.width         = state.width;
    g_ctx.height        = state.height;
    g_ctx.zoom          = static_cast<float>(state.zoom);
    g_ctx.offset        = state.offset;
    g_ctx.maxIterations = state.maxIterations;
    g_ctx.tileSize      = computeTileSizeFromZoom(g_ctx.zoom);
    g_ctx.overlayActive = state.heatmapOverlayEnabled;

    computeCudaFrame(g_ctx, state);
    applyZoomStep(g_ctx, g_zoomBus);

    // Ergebnisse zurück nach State
    state.zoom   = g_ctx.zoom;
    state.offset = g_ctx.offset;

    // HUD-Text (ASCII) – nutzt MaxFPS vom letzten Frame
    state.warzenschweinText = HudText::build(g_ctx, state);
    WarzenschweinOverlay::setText(state.warzenschweinText);

    drawFrame(g_ctx, state);

    const auto tFrame1 = Clock::now();
    g_frameTotal = std::chrono::duration_cast<msd>(tFrame1 - tFrame0).count();
    state.lastTimings.frameTotalMs = g_frameTotal;

    // Exakte uncapped Framezeit -> FpsMeter (Anzeige im naechsten Frame)
    FpsMeter::updateCoreMs(g_frameTotal);

    if (perfShouldLog(g_frame)) {
        // Periodisches Device-Log im Perf-Modus
        LuchsLogger::flushDeviceLogToHost(0);

        const int    resX = g_ctx.width;
        const int    resY = g_ctx.height;
        const int    it   = g_ctx.maxIterations;
        const double fps  = (g_ctx.frameTime > 0.0f) ? (1.0 / g_ctx.frameTime) : 0.0;

        const bool   vt   = state.lastTimings.valid;
        const double mapMs  = vt ? state.lastTimings.pboMap          : 0.0;
        const double mandMs = vt ? state.lastTimings.mandelbrotTotal : 0.0;
        const double entMs  = vt ? state.lastTimings.entropy         : 0.0;
        const double conMs  = vt ? state.lastTimings.contrast        : 0.0;

        const int mallocs = 0, frees = 0, dflush = 1;
        const int maxfps  = FpsMeter::currentMaxFpsInt();

        LUCHS_LOG_HOST(
            "[PERF-A] f=%d r=%dx%d zm=%.4g it=%d fp=%.1f mx=%d ma=%d fr=%d df=%d",
            g_frame, resX, resY, (double)g_ctx.zoom, it,
            fps, maxfps, mallocs, frees, dflush
        );

        LUCHS_LOG_HOST(
            "[PERF-B] f=%d mp=%.2f md=%.2f en=%.2f ct=%.2f tx=%.2f ov=%.2f tt=%.2f e0=%.4f c0=%.4f",
            g_frame,
            mapMs, mandMs, entMs, conMs,
            g_texMs, g_ovlMs, g_frameTotal,
            (double)g_ctx.lastEntropy, (double)g_ctx.lastContrast
        );
    }
}

} // namespace FramePipeline
