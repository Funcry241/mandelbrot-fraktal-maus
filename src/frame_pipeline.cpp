// Datei: src/frame_pipeline.cpp
// 🐭 Maus-Kommentar: Alpha 80 – Device-Log jetzt fehlertolerant: sofort bei Fehlern, sonst modulo-basiert. Klarer Datenfluss bleibt erhalten.
// 🦦 Otter: flushDeviceLogToHost abhängig von cudaPeekAtLastError – keine redundanten Fluten mehr. CPU-Zeitmessung nun pro CUDA-Frame aktiv.
// 🐑 Schneefuchs: performante Logik, deterministisch, ohne Nebeneffekte.

#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <vector_types.h>
#include <sstream>
#include <iomanip>
#include <chrono>  // für Zeitmessung
#include "pch.hpp"
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

namespace FramePipeline {

static FrameContext g_ctx;
static CommandBus g_zoomBus;
static int globalFrameCounter = 0;

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
        LUCHS_LOG_HOST("[PIPE] computeCudaFrame: dimensions=%dx%d, zoom=%.2f, tileSize=%d", frameCtx.width, frameCtx.height, frameCtx.zoom, frameCtx.tileSize);

    float2 gpuOffset     = make_float2((float)frameCtx.offset.x, (float)frameCtx.offset.y);
    float2 gpuNewOffset  = gpuOffset;

    const int tilesX = (frameCtx.width + frameCtx.tileSize - 1) / frameCtx.tileSize;
    const int tilesY = (frameCtx.height + frameCtx.tileSize - 1) / frameCtx.tileSize;
    const int numTiles = tilesX * tilesY;

    if (frameCtx.tileSize <= 0 || numTiles <= 0) {
        LUCHS_LOG_HOST("[FATAL] computeCudaFrame: Invalid tileSize (%d) or numTiles (%d)", frameCtx.tileSize, numTiles);
        return;
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
        gpuNewOffset,
        frameCtx.shouldZoom,
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
            LUCHS_LOG_HOST("[PIPE] Flushing device logs (err=%d, frame=%d)", static_cast<int>(err), globalFrameCounter);
        }
        LuchsLogger::flushDeviceLogToHost(0);
    }

    if (frameCtx.shouldZoom) {
        frameCtx.newOffset = { gpuNewOffset.x, gpuNewOffset.y };
    }

    if (Settings::debugLogging && !frameCtx.h_entropy.empty()) {
        LUCHS_LOG_HOST("[PIPE] Heatmap sample: Entropy[0]=%.4f Contrast[0]=%.4f", frameCtx.h_entropy[0], frameCtx.h_contrast[0]);
    }

    frameCtx.lastEntropy  = state.zoomResult.bestEntropy;
    frameCtx.lastContrast = state.zoomResult.bestContrast;
}

void applyZoomLogic(FrameContext& frameCtx, CommandBus& bus) {
    double2 diff = {
        frameCtx.newOffset.x - frameCtx.offset.x,
        frameCtx.newOffset.y - frameCtx.offset.y
    };
    double dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[Logic] Start | shouldZoom=%d | Zoom=%.2f | dO=%.4e", frameCtx.shouldZoom ? 1 : 0, frameCtx.zoom, dist);
    if (!frameCtx.shouldZoom) return;
    if (dist < Settings::DEADZONE) {
        if (Settings::debugLogging)
            LUCHS_LOG_HOST("[Logic] Offset in DEADZONE (%.4e) -> no movement", dist);
        return;
    }

    double stepScale = std::tanh(Settings::OFFSET_TANH_SCALE * dist);
    double2 step = {
        diff.x * stepScale * Settings::MAX_OFFSET_FRACTION,
        diff.y * stepScale * Settings::MAX_OFFSET_FRACTION
    };

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[Logic] Step len=%.4e | Zoom += %.5f", std::sqrt(step.x * step.x + step.y * step.y), Settings::AUTOZOOM_SPEED);

    ZoomCommand cmd;
    cmd.frameIndex = globalFrameCounter;
    cmd.oldOffset  = make_float2((float)frameCtx.offset.x, (float)frameCtx.offset.y);
    cmd.zoomBefore = (float)frameCtx.zoom;

    frameCtx.offset.x += static_cast<float>(step.x);
    frameCtx.offset.y += static_cast<float>(step.y);
    frameCtx.zoom *= Settings::AUTOZOOM_SPEED;

    cmd.newOffset  = make_float2((float)frameCtx.newOffset.x, (float)frameCtx.newOffset.y);
    cmd.zoomAfter  = (float)frameCtx.zoom;
    cmd.entropy    = frameCtx.lastEntropy;
    cmd.contrast   = frameCtx.lastContrast;

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
    g_ctx.tileSize      = state.lastTileSize;
    g_ctx.zoom          = static_cast<float>(state.zoom);
    g_ctx.offset        = state.offset;
    g_ctx.maxIterations = state.maxIterations;

    computeCudaFrame(g_ctx, state);
    applyZoomLogic(g_ctx, g_zoomBus);

    if (g_ctx.shouldZoom) {
        g_ctx.tileSize = computeTileSizeFromZoom(g_ctx.zoom);
        LUCHS_LOG_HOST("[ZoomLog] Updated tileSize after zoom: %.5f -> tileSize=%d", g_ctx.zoom, g_ctx.tileSize);
    }

    state.zoom = g_ctx.zoom;
    state.offset = g_ctx.offset;
    g_ctx.overlayActive = state.heatmapOverlayEnabled;

    std::ostringstream oss;
    oss << "Zoom: " << std::fixed << std::setprecision(4) << g_ctx.zoom << "\n";
    oss << "Offset: (" << g_ctx.offset.x << ", " << g_ctx.offset.y << ")\n";
    if (!g_ctx.h_entropy.empty())
        oss << "Entropy[0]: " << std::setprecision(3) << g_ctx.h_entropy[0] << "\n";
    float fps = static_cast<float>(1.0 / g_ctx.frameTime);
    oss << "FPS: " << std::setprecision(1) << fps;
    state.warzenschweinText = oss.str();
    WarzenschweinOverlay::setText(state.warzenschweinText);

    drawFrame(g_ctx, state.tex.id(), state);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] execute complete");
}

} // namespace FramePipeline
