// Datei: src/frame_pipeline.cpp
// Zeilen: 120
// 🐭 Maus-Kommentar: Kiwi – Kein Shadowing mehr: globale Variablen heißen jetzt g_ctx, g_zoomBus. Alles warnungsfrei, Otter und Schneefuchs lächeln zufrieden.

#include "pch.hpp"
#include "cuda_interop.hpp"
#include "renderer_pipeline.hpp"
#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <vector_types.h>

static FrameContext g_ctx;
static CommandBus g_zoomBus;
static int globalFrameCounter = 0;

void beginFrame(FrameContext& frameCtx) {
    float delta = static_cast<float>(glfwGetTime() - frameCtx.totalTime);
    frameCtx.frameTime = delta;
    frameCtx.totalTime += delta;
    frameCtx.timeSinceLastZoom += delta;
    frameCtx.shouldZoom = false;
    frameCtx.newOffset = frameCtx.offset;
    if (frameCtx.frameTime < 0.001f) frameCtx.frameTime = 0.001f;
    ++globalFrameCounter;
}

void computeCudaFrame(FrameContext& frameCtx, RendererState& state) {
    float2 gpuOffset = make_float2(static_cast<float>(frameCtx.offset.x), static_cast<float>(frameCtx.offset.y));
    float2 gpuNewOffset = gpuOffset;

    if (Settings::debugLogging) {
        std::printf("[DEBUG] Mandelbrot-Kernel Call: width=%d, height=%d, maxIter=%d, zoom=%.2f, offset=(%.10f, %.10f), tileSize=%d, supersampling=%d\n",
            frameCtx.width, frameCtx.height, frameCtx.maxIterations, frameCtx.zoom,
            frameCtx.offset.x, frameCtx.offset.y, frameCtx.tileSize, frameCtx.supersampling);
    }

    CudaInterop::renderCudaFrame(
        frameCtx.d_iterations,
        frameCtx.d_entropy,
        frameCtx.d_contrast,
        frameCtx.width,
        frameCtx.height,
        static_cast<float>(frameCtx.zoom),
        gpuOffset,
        frameCtx.maxIterations,
        frameCtx.h_entropy,
        frameCtx.h_contrast,
        gpuNewOffset,
        frameCtx.shouldZoom,
        frameCtx.tileSize,
        frameCtx.supersampling,
        state,
        state.d_tileSupersampling,
        state.h_tileSupersampling
    );
    if (frameCtx.shouldZoom) {
        frameCtx.newOffset.x = gpuNewOffset.x;
        frameCtx.newOffset.y = gpuNewOffset.y;
    }
    if (Settings::debugLogging) {
        std::printf("[CUDA] Input: offset=(%.10f, %.10f) | zoom=%.2f\n", frameCtx.offset.x, frameCtx.offset.y, frameCtx.zoom);
        if (!frameCtx.h_entropy.empty())
            std::printf("[Heatmap] Entropy[0]=%.4f Contrast[0]=%.4f\n", frameCtx.h_entropy[0], frameCtx.h_contrast[0]);
    }
}

void applyZoomLogic(FrameContext& frameCtx, CommandBus& bus) {
    double2 diff = { frameCtx.newOffset.x - frameCtx.offset.x, frameCtx.newOffset.y - frameCtx.offset.y };
    double dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);

    if (Settings::debugLogging) {
        std::printf("[Logic] Start | shouldZoom=%d | Zoom=%.2f | dO=%.4e\n", frameCtx.shouldZoom ? 1 : 0, frameCtx.zoom, dist);
    }
    if (!frameCtx.shouldZoom) return;
    if (dist < Settings::DEADZONE) {
        if (Settings::debugLogging)
            std::printf("[Logic] Offset in DEADZONE (%.4e) → no movement\n", dist);
        return;
    }
    double stepScale = std::tanh(Settings::OFFSET_TANH_SCALE * dist);
    double2 step = { diff.x * stepScale * Settings::MAX_OFFSET_FRACTION, diff.y * stepScale * Settings::MAX_OFFSET_FRACTION };
    if (Settings::debugLogging) {
        double stepLen = std::sqrt(step.x * step.x + step.y * step.y);
        std::printf("[Logic] Step len=%.4e | Zoom += %.5f\n", stepLen, Settings::AUTOZOOM_SPEED);
    }
    frameCtx.offset.x = static_cast<float>(frameCtx.offset.x + step.x);
    frameCtx.offset.y = static_cast<float>(frameCtx.offset.y + step.y);
    frameCtx.zoom *= Settings::AUTOZOOM_SPEED;

    ZoomCommand cmd;
    cmd.frameIndex = globalFrameCounter;
    cmd.oldOffset = make_float2(static_cast<float>(frameCtx.offset.x), static_cast<float>(frameCtx.offset.y));
    cmd.newOffset = make_float2(static_cast<float>(frameCtx.newOffset.x), static_cast<float>(frameCtx.newOffset.y));
    cmd.zoomBefore = static_cast<float>(frameCtx.zoom / Settings::AUTOZOOM_SPEED);
    cmd.zoomAfter = static_cast<float>(frameCtx.zoom);
    cmd.entropy = frameCtx.lastEntropy;
    cmd.contrast = frameCtx.lastContrast;
    cmd.tileIndex = frameCtx.lastTileIndex;
    bus.push(cmd);
    frameCtx.timeSinceLastZoom = 0.0f;
}

void drawFrame(FrameContext& frameCtx, GLuint tex, RendererState& state) {
    if (frameCtx.overlayActive) {
        HeatmapOverlay::drawOverlay(
            frameCtx.h_entropy,
            frameCtx.h_contrast,
            frameCtx.width,
            frameCtx.height,
            frameCtx.tileSize,
            tex,
            state
        );
    }
    RendererPipeline::drawFullscreenQuad(tex);
}

// ---- Zentrale Frame-Pipeline ----
void framePipeline(RendererState& state) {
    beginFrame(g_ctx);
    computeCudaFrame(g_ctx, state);
    applyZoomLogic(g_ctx, g_zoomBus);
    drawFrame(g_ctx, state.tex, state);
}
