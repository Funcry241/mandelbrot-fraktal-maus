// Datei: src/frame_pipeline.cpp
// 🐭 Maus: Eine Quelle für Tiles pro Frame. Vor Render: Buffer-Sync via setupCudaBuffers(...).
// 🦦 Otter: Sanity-Logs, deterministische Reihenfolge; Zoom V2 außerhalb der CUDA-Interop. (Bezug zu Otter)
// 🐑 Schneefuchs: Kein doppeltes Sizing, keine Alt-Settings. (Bezug zu Schneefuchs)

#include "pch.hpp"
#include <vector_types.h>
#include <chrono>  // Zeitmessung
#include "cuda_interop.hpp"
#include "renderer_pipeline.hpp"
#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "zoom_command.hpp"
#include "heatmap_overlay.hpp"
#include "warzenschwein_overlay.hpp"
#include "settings.hpp"
#include "luchs_log_host.hpp"
#include "luchs_cuda_log_buffer.hpp"
#include "common.hpp"         // computeTileSizeFromZoom
#include "zoom_logic.hpp"     // Zoom V2 API

namespace FramePipeline {

static FrameContext g_ctx;
static CommandBus g_zoomBus;
static int globalFrameCounter = 0;

// Kleiner, lokaler Zoom-Gain (pro akzeptiertem Schritt). Bewusst NICHT in settings.hpp.
static constexpr float kZOOM_GAIN = 1.006f;

void beginFrame(FrameContext& frameCtx) {
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] beginFrame: time=%.4f, totalFrames=%d", glfwGetTime(), globalFrameCounter);
    float delta = static_cast<float>(glfwGetTime() - frameCtx.totalTime);
    frameCtx.frameTime = (delta < 0.001f) ? 0.001f : delta;
    frameCtx.totalTime += delta;
    frameCtx.timeSinceLastZoom += delta;
    frameCtx.shouldZoom = false;
    frameCtx.newOffset = frameCtx.offset;
    ++globalFrameCounter;
}

void computeCudaFrame(FrameContext& frameCtx, RendererState& state) {
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] computeCudaFrame: dimensions=%dx%d, zoom=%.5f, tileSize=%d",
                       frameCtx.width, frameCtx.height, frameCtx.zoom, frameCtx.tileSize);

    float2 gpuOffset     = make_float2((float)frameCtx.offset.x, (float)frameCtx.offset.y);
    float2 gpuNewOffset  = gpuOffset;

    // Lokale Tile-Geometrie (nur Laufzeit-Indexing / Sanity)
    const int tilesX   = (frameCtx.width  + frameCtx.tileSize - 1) / frameCtx.tileSize;
    const int tilesY   = (frameCtx.height + frameCtx.tileSize - 1) / frameCtx.tileSize;
    const int numTiles = tilesX * tilesY;

    if (frameCtx.tileSize <= 0 || numTiles <= 0) {
        LUCHS_LOG_HOST("[FATAL] computeCudaFrame: invalid tileSize=%d or numTiles=%d",
                       frameCtx.tileSize, numTiles);
        return;
    }

    // *** Zentrale Stelle: Device-Buffer-Sync für aktuelles Tile-Layout ***
    // Idempotent: allocate() in RendererState vergrößert nur bei Bedarf
    state.setupCudaBuffers(frameCtx.tileSize);

    // Sanity-Log: erwartete vs. allozierte Größen (ASCII)
    if (Settings::debugLogging) {
        const int totalPixels = frameCtx.width * frameCtx.height;
        const size_t need_it_bytes       = static_cast<size_t>(totalPixels) * sizeof(int);
        const size_t need_entropy_bytes  = static_cast<size_t>(numTiles)    * sizeof(float);
        const size_t need_contrast_bytes = static_cast<size_t>(numTiles)    * sizeof(float);
        LUCHS_LOG_HOST("[SANITY] tiles=%d (%d x %d) pixels=%d need(it=%zu entropy=%zu contrast=%zu) alloc(it=%zu entropy=%zu contrast=%zu)",
                       numTiles, tilesX, tilesY, totalPixels,
                       need_it_bytes, need_entropy_bytes, need_contrast_bytes,
                       state.d_iterations.size(), state.d_entropy.size(), state.d_contrast.size());
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Calling CudaInterop::renderCudaFrame");

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
        gpuNewOffset,            // wird von Interop nicht mehr gesetzt – wir setzen danach selbst
        frameCtx.shouldZoom,     // wird von Interop nicht mehr gesetzt – wir setzen danach selbst
        frameCtx.tileSize,
        state
    );

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Returned from renderCudaFrame");

    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (Settings::debugLogging && state.lastTimings.valid) {
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
    } else if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[TIME] CUDA kernel + sync: %.3f ms", ms);
    }

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess || (globalFrameCounter % 30 == 0)) {
        if (Settings::debugLogging) {
            LUCHS_LOG_HOST("[PIPE] Flushing device logs (err=%d, frame=%d)",
                           static_cast<int>(err), globalFrameCounter);
        }
        LuchsLogger::flushDeviceLogToHost(0);
    }

    // --- Zoom V2 Entscheidung NACH dem Render-Frame treffen ---
    // Tiles bereits oben berechnet (eine Quelle).
    const float2 currOff = make_float2((float)frameCtx.offset.x, (float)frameCtx.offset.y);
    const float2 prevOff = currOff; // alternativ: separaten historischen Offset führen

    auto zr = ZoomLogic::evaluateZoomTarget(
        frameCtx.h_entropy,
        frameCtx.h_contrast,
        tilesX, tilesY,
        frameCtx.width, frameCtx.height,
        currOff, frameCtx.zoom,
        prevOff,
        state.zoomV2State
    );

    frameCtx.shouldZoom = zr.shouldZoom;
    if (zr.shouldZoom) {
        frameCtx.newOffset = { zr.newOffset.x, zr.newOffset.y };
    }

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[PIPE] ZOOMV2: best=%d score=%.3f accept=%d newOff=(%.6f,%.6f)",
                       zr.bestIndex, zr.bestScore, zr.shouldZoom ? 1 : 0,
                       zr.newOffset.x, zr.newOffset.y);
    }

    // Für HUD/Diagnose einfache Statistiken
    if (Settings::debugLogging && !frameCtx.h_entropy.empty()) {
        float minE =  1e9f, maxE = -1e9f;
        float minC =  1e9f, maxC = -1e9f;
        for (std::size_t i = 0; i < frameCtx.h_entropy.size(); ++i) {
            float e = frameCtx.h_entropy[i];
            float c = frameCtx.h_contrast[i];
            minE = std::min(minE, e); maxE = std::max(maxE, e);
            minC = std::min(minC, c); maxC = std::max(maxC, c);
        }
        LUCHS_LOG_HOST("[HEAT] zoom=%.5f offset=(%.5f, %.5f) tileSize=%d",
                       frameCtx.zoom, frameCtx.offset.x, frameCtx.offset.y, frameCtx.tileSize);
        LUCHS_LOG_HOST("[HEAT] Entropy: min=%.5f  max=%.5f | Contrast: min=%.5f  max=%.5f",
                       minE, maxE, minC, maxC);
    }
}

void applyZoomLogic(FrameContext& frameCtx, CommandBus& bus, RendererState& state) {
    // Neue Logik: newOffset wurde bereits durch Zoom V2 geglättet.
    (void)state; // ASCII: suppress unused-parameter warning (MSVC /WX)
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[Logic] Start | shouldZoom=%d | Zoom=%.5f",
                       frameCtx.shouldZoom ? 1 : 0, frameCtx.zoom);

    if (!frameCtx.shouldZoom) return;

    double2 diff = {
        frameCtx.newOffset.x - frameCtx.offset.x,
        frameCtx.newOffset.y - frameCtx.offset.y
    };
    double dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);

    // Offset direkt übernehmen (V2 liefert bereits smoothed target)
    frameCtx.offset = frameCtx.newOffset;

    // Zoom leicht anheben, um „Reinzoomen“ zu signalisieren
    frameCtx.zoom *= kZOOM_GAIN;

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[Logic] Applied | dist=%.4e | newZoom=%.5f | newOffset=(%.6f,%.6f)",
                       dist, frameCtx.zoom, frameCtx.offset.x, frameCtx.offset.y);

    // Command protokollieren
    ZoomCommand cmd;
    cmd.frameIndex = globalFrameCounter;
    cmd.oldOffset  = make_float2((float)(frameCtx.offset.x - diff.x), (float)(frameCtx.offset.y - diff.y));
    cmd.zoomBefore = (float)(frameCtx.zoom / kZOOM_GAIN);
    cmd.newOffset  = make_float2((float)frameCtx.newOffset.x, (float)frameCtx.newOffset.y);
    cmd.zoomAfter  = (float)frameCtx.zoom;
    // Für HUD: einfache Werte; falls detailliert gewünscht, aus evaluateZoomTarget herausreichen
    cmd.entropy    = 0.0f;
    cmd.contrast   = 0.0f;

    bus.push(cmd);
    frameCtx.timeSinceLastZoom = 0.0f;
}

void drawFrame(FrameContext& frameCtx, GLuint tex, RendererState& state) {
    if (Settings::debugLogging) {
        LUCHS_LOG_HOST(
            "[PIPE] drawFrame: overlay=%d warzenschwein=%d entropy=%zu contrast=%zu tex=%u",
            frameCtx.overlayActive ? 1 : 0,
            Settings::warzenschweinOverlayEnabled ? 1 : 0,
            frameCtx.h_entropy.size(),
            frameCtx.h_contrast.size(),
            tex
        );
    }

    OpenGLUtils::setGLResourceContext("frame");
    OpenGLUtils::updateTextureFromPBO(state.pbo.id(), tex, frameCtx.width, frameCtx.height);

    RendererPipeline::drawFullscreenQuad(tex);

    if (frameCtx.overlayActive)
        HeatmapOverlay::drawOverlay(
            frameCtx.h_entropy,
            frameCtx.h_contrast,
            frameCtx.width,
            frameCtx.height,
            frameCtx.tileSize,
            tex,
            state
        );

    if (Settings::warzenschweinOverlayEnabled && !state.warzenschweinText.empty())
        WarzenschweinOverlay::drawOverlay(state);
}

void execute(RendererState& state) {
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] execute start");

    beginFrame(g_ctx);

    g_ctx.width         = state.width;
    g_ctx.height        = state.height;
    g_ctx.zoom          = static_cast<float>(state.zoom);
    g_ctx.offset        = state.offset;
    g_ctx.maxIterations = state.maxIterations;

    // Ameise: tileSize genau einmal pro Frame bestimmen
    g_ctx.tileSize      = computeTileSizeFromZoom(g_ctx.zoom);

    // Der eigentliche CUDA-Frame (inkl. Buffer-Sync & Sanity-Logs)
    computeCudaFrame(g_ctx, state);

    // Zoom-Entscheidung anwenden
    applyZoomLogic(g_ctx, g_zoomBus, state);

    // Zustand zurückspiegeln
    state.zoom   = g_ctx.zoom;
    state.offset = g_ctx.offset;
    g_ctx.overlayActive = state.heatmapOverlayEnabled;

    // HUD-Text
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "Zoom:    " << g_ctx.zoom << "\n";
    oss << "Offset:  (" << g_ctx.offset.x << ", " << g_ctx.offset.y << ")\n";

    if (!g_ctx.h_entropy.empty()) {
        oss << std::setprecision(3);
        oss << "Entropy: " << g_ctx.h_entropy[0] << "\n";
    }

    float fps = static_cast<float>(1.0 / g_ctx.frameTime);
    oss << std::setprecision(1);
    oss << "FPS:     " << fps << "\n";

    state.warzenschweinText = oss.str();
    WarzenschweinOverlay::setText(state.warzenschweinText);

    drawFrame(g_ctx, state.tex.id(), state);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] execute complete");
}

} // namespace FramePipeline
