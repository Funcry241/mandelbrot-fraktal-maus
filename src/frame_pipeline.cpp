// Datei: src/frame_pipeline.cpp
// 🐭 Maus-Kommentar: Alpha 49f - Supersampling restlos entfernt. `computeCudaFrame()` ohne `supersampling`, `d_tileSupersampling` oder `h_tileSupersampling`. Alles stabil, nichts vergessen. Otter: deterministisch. Schneefuchs: präzise.
// 🐭 Maus-Kommentar: Alpha 63b - Setzt FrameContext-Dimensionen explizit aus RendererState - kein implizites GLFW nötig.
// 🦦 Otter: Klare Datenflussregel: RendererState .> FrameContext . CUDA. Kein Kontext-Zugriff im Pipeline-Code.
// 🐑 Schneefuchs: Trennung von Plattformdetails und Logik ist jetzt durchgezogen.

#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <vector_types.h>
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
    float delta = static_cast<float>(glfwGetTime() - frameCtx.totalTime);
    frameCtx.frameTime = (delta < 0.001f) ? 0.001f : delta;
    frameCtx.totalTime += delta;
    frameCtx.timeSinceLastZoom += delta;
    frameCtx.shouldZoom = false;
    frameCtx.newOffset = frameCtx.offset;
    ++globalFrameCounter;
}

void computeCudaFrame(FrameContext& frameCtx, RendererState& state) {
    float2 gpuOffset     = make_float2((float)frameCtx.offset.x, (float)frameCtx.offset.y);
    float2 gpuNewOffset  = gpuOffset;

    if (Settings::debugLogging) {
        LUCHS_LOG_HOST("[DEBUG] Mandelbrot-Kernel Call: width=%d, height=%d, maxIter=%d, zoom=%.2f, offset=(%.10f, %.10f), tileSize=%d",
            frameCtx.width, frameCtx.height, frameCtx.maxIterations, frameCtx.zoom,
            frameCtx.offset.x, frameCtx.offset.y, frameCtx.tileSize);
    }

    const int tilesX = (frameCtx.width + frameCtx.tileSize - 1) / frameCtx.tileSize;
    const int tilesY = (frameCtx.height + frameCtx.tileSize - 1) / frameCtx.tileSize;
    const int numTiles = tilesX * tilesY;

    if (frameCtx.tileSize <= 0 || numTiles <= 0) {
        LUCHS_LOG_HOST("[FATAL] computeCudaFrame: Invalid tileSize (%d) or numTiles (%d)", frameCtx.tileSize, numTiles);
        return;
    }

    // 🦦 Otter-Fix: Verwende gültige CUDA-Pointer aus RendererState
    CudaInterop::renderCudaFrame(
        state.d_iterations,
        state.d_entropy,
        state.d_contrast,
        frameCtx.width,
        frameCtx.height,
        (float)frameCtx.zoom,
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
        LUCHS_LOG_HOST("[KERNEL] Mandelbrot kernel launched - synchronizing");
    CUDA_CHECK(cudaDeviceSynchronize());
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[KERNEL] Mandelbrot kernel completed");

    if (frameCtx.shouldZoom) {
        frameCtx.newOffset = { gpuNewOffset.x, gpuNewOffset.y };
    }

    if (Settings::debugLogging && !frameCtx.h_entropy.empty()) {
        LUCHS_LOG_HOST("[CUDA] Input: offset=(%.10f, %.10f) | zoom=%.2f", frameCtx.offset.x, frameCtx.offset.y, frameCtx.zoom);
        LUCHS_LOG_HOST("[Heatmap] Entropy[0]=%.4f Contrast[0]=%.4f", frameCtx.h_entropy[0], frameCtx.h_contrast[0]);
    }
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
    if (frameCtx.overlayActive)
        HeatmapOverlay::drawOverlay(frameCtx.h_entropy, frameCtx.h_contrast, frameCtx.width, frameCtx.height, frameCtx.tileSize, tex, state);
    if (Settings::warzenschweinOverlayEnabled)
        WarzenschweinOverlay::drawOverlay(state);

    RendererPipeline::drawFullscreenQuad(tex);
}

void execute(RendererState& state) {
    beginFrame(g_ctx);
    g_ctx.width  = state.width;
    g_ctx.height = state.height;
    g_ctx.tileSize = state.lastTileSize; // <--- Fix: tileSize synchronisieren

    computeCudaFrame(g_ctx, state);
    applyZoomLogic(g_ctx, g_zoomBus);
    drawFrame(g_ctx, state.tex, state);
}

} // namespace FramePipeline
