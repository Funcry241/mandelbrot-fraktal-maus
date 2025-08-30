///// Otter: GL-Upload via freie Funktionen; ASCII-Logs; kein Funktionswechsel.
///// Schneefuchs: /WX-fest; Header-Sync; RAII & deterministisch.
///// Maus: Eine Quelle für Tiles/Upload; Zoom V2 außerhalb der CUDA-Interop.

// Datei: src/frame_pipeline.cpp

// Maus: Eine Quelle für Tiles pro Frame. Vor Render: Buffer-Sync via setupCudaBuffers(...).
// Otter: Sanity-Logs, deterministische Reihenfolge; Zoom V2 außerhalb der CUDA-Interop.
// Schneefuchs: Kein doppeltes Sizing, keine Alt-Settings.
// Schneefuchs: /WX-fest – keine konstanten ifs (C4127) und keine C4702 mehr; Debug/Perf via if constexpr. ASCII logs only.

#include "pch.hpp"
#include "renderer_resources.hpp"
#include "cuda_interop.hpp"
#include <vector_types.h>
#include <vector_functions.h> // make_float2
#include <chrono>   // timing
#include <cstdio>   // snprintf (ASCII only)
#include <cmath>
#include "renderer_pipeline.hpp"
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "zoom_command.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "common.hpp"
#include "zoom_logic.hpp"
#include "fps_meter.hpp"
#include "hud_text.hpp"
#include <GL/glew.h>

namespace FramePipeline {

static FrameContext g_ctx;
static CommandBus g_zoomBus;
static int globalFrameCounter = 0;

// Small local zoom gain (per accepted step)
static constexpr float kZOOM_GAIN = 1.006f;

// Otter: Local perf accumulators for this TU (ASCII-only).
namespace {
    using Clock = std::chrono::high_resolution_clock;
    using msd   = std::chrono::duration<double, std::milli>;

    // Warmup & periodic logging (only when Settings::performanceLogging == true)
    constexpr int PERF_WARMUP_FRAMES = 30;
    constexpr int PERF_LOG_EVERY     = 30;

    // Phase timings measured here (tex upload + draw, overlays, frame total).
    // Map/Kernel/Entropy/Contrast are provided by state.lastTimings (Interop/CUDA).
    double g_perfTexMs       = 0.0;
    double g_perfOverlaysMs  = 0.0;
    double g_perfFrameTotal  = 0.0;

    inline bool perfShouldLog(int frameIdx) {
        if constexpr (Settings::performanceLogging) {
            if (frameIdx <= PERF_WARMUP_FRAMES) return false;
            return (frameIdx % PERF_LOG_EVERY) == 0;
        } else {
            (void)frameIdx;
            return false;
        }
    }
}

// --------------------------------- frame begin --------------------------------
void beginFrame(FrameContext& frameCtx, RendererState& state) {
    // Schneefuchs: Host-Timings pro Frame auf Null (eine Quelle, falls genutzt)
    state.lastTimings.resetHostFrame();

    const double now = glfwGetTime();
    if constexpr (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] beginFrame: time=%.4f, totalFrames=%d", now, globalFrameCounter);

    float delta = static_cast<float>(now - frameCtx.totalTime);
    frameCtx.frameTime = (delta < 0.001f) ? 0.001f : delta;
    frameCtx.totalTime = now;
    frameCtx.timeSinceLastZoom += delta;
    frameCtx.shouldZoom = false;
    frameCtx.newOffset = frameCtx.offset;
    ++globalFrameCounter;
}

// ------------------------------- CUDA + analysis ------------------------------
void computeCudaFrame(FrameContext& frameCtx, RendererState& state) {
    if constexpr (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] computeCudaFrame: dimensions=%dx%d, zoom=%.5f, tileSize=%d",
                       frameCtx.width, frameCtx.height, frameCtx.zoom, frameCtx.tileSize);

    float2 gpuOffset     = make_float2((float)frameCtx.offset.x, (float)frameCtx.offset.y);
    float2 gpuNewOffset  = gpuOffset;

    const int tilesX   = (frameCtx.width  + frameCtx.tileSize - 1) / frameCtx.tileSize;
    const int tilesY   = (frameCtx.height + frameCtx.tileSize - 1) / frameCtx.tileSize;
    const int numTiles = tilesX * tilesY;

    if (frameCtx.tileSize <= 0 || numTiles <= 0) {
        LUCHS_LOG_HOST("[FATAL] computeCudaFrame: invalid tileSize=%d or numTiles=%d",
                       frameCtx.tileSize, numTiles);
        return;
    }

    state.setupCudaBuffers(frameCtx.tileSize);

    if constexpr (Settings::debugLogging) {
        const size_t totalPixels            = size_t(frameCtx.width) * size_t(frameCtx.height);
        const size_t need_it_bytes          = totalPixels * sizeof(int);
        const size_t need_entropy_bytes     = size_t(numTiles) * sizeof(float);
        const size_t need_contrast_bytes    = size_t(numTiles) * sizeof(float);
        LUCHS_LOG_HOST("[SANITY] tiles=%d (%d x %d) pixels=%zu need(it=%zu entropy=%zu contrast=%zu) alloc(it=%zu entropy=%zu contrast=%zu)",
                       numTiles, tilesX, tilesY, totalPixels,
                       need_it_bytes, need_entropy_bytes, need_contrast_bytes,
                       state.d_iterations.size(), state.d_entropy.size(), state.d_contrast.size());
    }

    // Host-side timing gated at compile-time.
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        auto t0 = Clock::now();

        // --- Diagnostics around kernel launch/sync
        LUCHS_LOG_HOST("[KERNEL] launch begin: w=%d h=%d zoom=%.6f tilesz=%d",
                       frameCtx.width, frameCtx.height, frameCtx.zoom, frameCtx.tileSize);

        CudaInterop::renderCudaFrame(
            state.d_iterations,
            state.d_entropy,
            state.d_contrast,
            frameCtx.width,
            frameCtx.height,
            frameCtx.zoom,
            gpuOffset,
            frameCtx.maxIterations,
            frameCtx.h_entropy,
            frameCtx.h_contrast,
            gpuNewOffset,
            frameCtx.shouldZoom,
            frameCtx.tileSize,
            state
        );

        LUCHS_LOG_HOST("[KERNEL] launch returned");

        cudaError_t syncErr = cudaDeviceSynchronize();
        LUCHS_LOG_HOST("[KERNEL] cudaDeviceSynchronize -> %d", static_cast<int>(syncErr));

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
    } else {
        // Hot path without host timing
        LUCHS_LOG_HOST("[KERNEL] launch begin (hot)");
        CudaInterop::renderCudaFrame(
            state.d_iterations,
            state.d_entropy,
            state.d_contrast,
            frameCtx.width,
            frameCtx.height,
            frameCtx.zoom,
            gpuOffset,
            frameCtx.maxIterations,
            frameCtx.h_entropy,
            frameCtx.h_contrast,
            gpuNewOffset,
            frameCtx.shouldZoom,
            frameCtx.tileSize,
            state
        );
        cudaError_t syncErr = cudaDeviceSynchronize();
        LUCHS_LOG_HOST("[KERNEL] cudaDeviceSynchronize -> %d (hot)", static_cast<int>(syncErr));
    }

    // Deterministic, modular device-log flush; immediate flush on error.
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess || (globalFrameCounter % 30 == 0)) {
        if constexpr (Settings::debugLogging)
            LUCHS_LOG_HOST("[PIPE] Flushing device logs (err=%d, frame=%d)",
                           static_cast<int>(err), globalFrameCounter);
        LuchsLogger::flushDeviceLogToHost(0);
    }

    const float2 currOff = make_float2((float)frameCtx.offset.x, (float)frameCtx.offset.y);
    const float2 prevOff = currOff;

    auto zr = ZoomLogic::evaluateZoomTarget(
        frameCtx.h_entropy,
        frameCtx.h_contrast,
        tilesX, tilesY,
        frameCtx.width, frameCtx.height,
        currOff, frameCtx.zoom,
        prevOff,
        state.zoomV2State
    );

    // Persist analysis
    if (zr.bestIndex >= 0) {
        frameCtx.lastEntropy  = zr.bestEntropy;
        frameCtx.lastContrast = zr.bestContrast;
    } else {
        frameCtx.lastEntropy  = 0.0f;
        frameCtx.lastContrast = 0.0f;
    }

    frameCtx.shouldZoom = zr.shouldZoom;
    if (zr.shouldZoom) {
        frameCtx.newOffset = { zr.newOffset.x, zr.newOffset.y };
    }

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] ZOOMV2: best=%d score=%.3f accept=%d newOff=(%.6f,%.6f)",
                       zr.bestIndex, zr.bestScore, zr.shouldZoom ? 1 : 0,
                       zr.newOffset.x, zr.newOffset.y);
    }

    if constexpr (Settings::debugLogging) {
        if (!frameCtx.h_entropy.empty() && !frameCtx.h_contrast.empty()) {
            float minE =  1e9f, maxE = -1e9f;
            float minC =  1e9f, maxC = -1e9f;
            const size_t N = std::min(frameCtx.h_entropy.size(), frameCtx.h_contrast.size());
            for (size_t i = 0; i < N; ++i) {
                const float e = frameCtx.h_entropy[i];
                const float c = frameCtx.h_contrast[i];
                minE = std::min(minE, e);
                maxE = std::max(maxE, e);
                minC = std::min(minC, c);
                maxC = std::max(maxC, c);
            }
            LUCHS_LOG_HOST("[HEAT] zoom=%.5f offset=(%.5f, %.5f) tileSize=%d",
                           frameCtx.zoom, frameCtx.offset.x, frameCtx.offset.y, frameCtx.tileSize);
            LUCHS_LOG_HOST("[HEAT] Entropy: min=%.5f  max=%.5f | Contrast: min=%.5f  max=%.5f",
                           minE, maxE, minC, maxC);
        }
    }
}

// ------------------------------- apply zoom step ------------------------------
void applyZoomLogic(FrameContext& frameCtx, CommandBus& bus, RendererState& state) {
    (void)state;
    if (!frameCtx.shouldZoom) return;

    const double2 diff = { frameCtx.newOffset.x - frameCtx.offset.x, frameCtx.newOffset.y - frameCtx.offset.y };
    frameCtx.offset = frameCtx.newOffset;
    frameCtx.zoom *= kZOOM_GAIN;

    ZoomCommand cmd;
    cmd.frameIndex = globalFrameCounter;
    cmd.oldOffset  = make_float2((float)(frameCtx.offset.x - diff.x), (float)(frameCtx.offset.y - diff.y));
    cmd.zoomBefore = (float)(frameCtx.zoom / kZOOM_GAIN);
    cmd.newOffset  = make_float2((float)frameCtx.newOffset.x, (float)frameCtx.newOffset.y);
    cmd.zoomAfter  = (float)frameCtx.zoom;
    cmd.entropy    = frameCtx.lastEntropy;
    cmd.contrast   = frameCtx.lastContrast;

    bus.push(cmd);
    frameCtx.timeSinceLastZoom = 0.0f;
}

// ------------------------------ draw (GL upload + FSQ) -----------------------
void drawFrame(FrameContext& frameCtx, GLuint tex, RendererState& state) {
    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] drawFrame begin: tex=%u pbo=%u %dx%d",
                       static_cast<unsigned>(tex),
                       static_cast<unsigned>(state.pbo.id()),
                       frameCtx.width, frameCtx.height);
    }

    // texMs measures only PBO->Texture upload + FSQ draw, separate from overlays.
    auto tTex0 = Clock::now();

    // Optional: einmalige Testfarbe, um Draw-Pfad zu verifizieren (nur Debug)
    if constexpr (Settings::debugLogging) {
        if (globalFrameCounter == 1) {
            unsigned char cc[4] = {255, 0, 255, 255}; // magenta RGBA
            glBindTexture(GL_TEXTURE_2D, tex);
            glClearTexImage(tex, 0, GL_RGBA, GL_UNSIGNED_BYTE, cc);
            LUCHS_LOG_HOST("[DBG] clearTexImage applied (magenta) to verify draw path");
        }
    }

    // Debug: PBO-Inhalt kurz peek'en (erste Bytes), um Upload-Quelle zu validieren
    if constexpr (Settings::debugLogging) {
        GLint prevPBO = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &prevPBO);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, state.pbo.id());
        const GLsizeiptr total = (GLsizeiptr)frameCtx.width * (GLsizeiptr)frameCtx.height * 4;
        GLsizeiptr sample = total < 64 ? total : 64;
        void* ptr = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, sample, GL_MAP_READ_BIT);
        if (ptr) {
            const unsigned char* u = static_cast<const unsigned char*>(ptr);
            unsigned int s0=u[0], s1=u[1], s2=u[2], s3=u[3];
            unsigned int sum = 0; for (int i=0; i<16 && i<sample; ++i) sum += u[i];
            LUCHS_LOG_HOST("[PBO-PEEK] pbo=%u first4=%u,%u,%u,%u sum16=%u",
                           static_cast<unsigned>(state.pbo.id()), s0, s1, s2, s3, sum);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        } else {
            LUCHS_LOG_HOST("[PBO-PEEK][ERR] glMapBufferRange returned null for pbo=%u",
                           static_cast<unsigned>(state.pbo.id()));
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, static_cast<GLuint>(prevPBO));
    }

    // Viewport sicherstellen (einige Plattformen starten mit 0x0)
    GLint vp[4] = {0,0,0,0};
    glGetIntegerv(GL_VIEWPORT, vp);
    if (vp[2] != frameCtx.width || vp[3] != frameCtx.height) {
        glViewport(0, 0, frameCtx.width, frameCtx.height);
        LUCHS_LOG_HOST("[VIEWPORT] set to %dx%d (was %d x %d)", frameCtx.width, frameCtx.height, vp[2], vp[3]);
    } else if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[VIEWPORT] already %d x %d", vp[2], vp[3]);
    }

    // Explizite Bind-Reihenfolge: (texture,pbo) binden und uploaden
    setGLResourceContext(tex, state.pbo.id());
    updateTextureFromPBO(tex, state.pbo.id(), frameCtx.width, frameCtx.height);

    if constexpr (Settings::debugLogging) {
        const GLenum upErr = glGetError();
        LUCHS_LOG_HOST("[PIPE][UPLOAD] tex=%u pbo=%u %dx%d glError=0x%04X",
                       static_cast<unsigned>(tex),
                       static_cast<unsigned>(state.pbo.id()),
                       frameCtx.width, frameCtx.height,
                       static_cast<unsigned>(upErr));
    }

    RendererPipeline::drawFullscreenQuad(tex);

    auto tTex1 = Clock::now();
    g_perfTexMs = std::chrono::duration_cast<msd>(tTex1 - tTex0).count();
    state.lastTimings.uploadMs = g_perfTexMs;

    // overlaysMs measures Heatmap + Warzenschwein together.
    auto tOv0 = Clock::now();

    if (frameCtx.overlayActive)
        HeatmapOverlay::drawOverlay(frameCtx.h_entropy, frameCtx.h_contrast, frameCtx.width, frameCtx.height, frameCtx.tileSize, tex, state);

    if constexpr (Settings::warzenschweinOverlayEnabled) {
        if (!state.warzenschweinText.empty())
            WarzenschweinOverlay::drawOverlay(state);
    }

    auto tOv1 = Clock::now();
    g_perfOverlaysMs = std::chrono::duration_cast<msd>(tOv1 - tOv0).count();
    state.lastTimings.overlaysMs = g_perfOverlaysMs;

    if constexpr (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] drawFrame end: texMs=%.3f ovMs=%.3f", g_perfTexMs, g_perfOverlaysMs);
    }
}

// ---------------------------------- execute ----------------------------------
void execute(RendererState& state) {
    auto tFrame0 = Clock::now();

    beginFrame(g_ctx, state);

    g_ctx.width         = state.width;
    g_ctx.height        = state.height;
    g_ctx.zoom          = static_cast<float>(state.zoom);
    g_ctx.offset        = state.offset;
    g_ctx.maxIterations = state.maxIterations;
    g_ctx.tileSize      = computeTileSizeFromZoom(g_ctx.zoom);

    computeCudaFrame(g_ctx, state);
    applyZoomLogic(g_ctx, g_zoomBus, state);

    state.zoom   = g_ctx.zoom;
    state.offset = g_ctx.offset;
    g_ctx.overlayActive = state.heatmapOverlayEnabled;

    // HUD text (ASCII, zentraler Builder) – nutzt MaxFPS vom *vorigen* Frame
    state.warzenschweinText = HudText::build(g_ctx, state);
    WarzenschweinOverlay::setText(state.warzenschweinText);

    drawFrame(g_ctx, state.tex.id(), state);

    auto tFrame1 = Clock::now();
    g_perfFrameTotal = std::chrono::duration_cast<msd>(tFrame1 - tFrame0).count();
    state.lastTimings.frameTotalMs = g_perfFrameTotal;

    // Exakte uncapped Framezeit -> FpsMeter (zeigt im nächsten Frame)
    FpsMeter::updateCoreMs(g_perfFrameTotal);

    // Compact PERF line only when enabled.
    if (perfShouldLog(globalFrameCounter)) {
        // Periodic device-log flush, only in performance mode.
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

        // Zeile A: Meta + FPS
        LUCHS_LOG_HOST(
            "[PERF-A] f=%d r=%dx%d zm=%.4g it=%d fp=%.1f mx=%d ma=%d fr=%d df=%d",
            globalFrameCounter, resX, resY, (double)g_ctx.zoom, it,
            fps, maxfps, mallocs, frees, dflush
        );

        // Zeile B: Zeiten + erste Metrikwerte
        LUCHS_LOG_HOST(
            "[PERF-B] f=%d mp=%.2f md=%.2f en=%.2f ct=%.2f tx=%.2f ov=%.2f tt=%.2f e0=%.4f c0=%.4f",
            globalFrameCounter,
            mapMs, mandMs, entMs, conMs, g_perfTexMs, g_perfOverlaysMs, g_perfFrameTotal,
            (double)g_ctx.lastEntropy, (double)g_ctx.lastContrast
        );
    }
}

} // namespace FramePipeline
