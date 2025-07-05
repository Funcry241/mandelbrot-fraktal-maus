// Datei: src/frame_pipeline.cpp
// Zeilen: 121
// 🐭 Maus-Kommentar: Kiwi – Heatmap-Logik nur nach Render, alle Daten aktuell. Keine DrawOverlay-Altpfade mehr. Schneefuchs validiert: Framedaten, Overlay und CUDA sauber synchron.

#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "cuda_interop.hpp"
#include "renderer_pipeline.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector_types.h>
#include <vector>

static int globalFrameCounter = 0;

void beginFrame(FrameContext& ctx) {
    ctx.frameTime = glfwGetTime();
    ctx.frameTime -= ctx.totalTime;
    ctx.totalTime += ctx.frameTime;
    ctx.timeSinceLastZoom += ctx.frameTime;
    ctx.shouldZoom = false;
    ctx.newOffset = ctx.offset;
    ctx.frameTime = std::max(0.001, ctx.frameTime);
    ++globalFrameCounter;
}

void computeCudaFrame(FrameContext& ctx, RendererState& state) {
    // 🥝 Kiwi: Zuerst Mandelbrot-Kernel, dann Entropie/Kontrast-Analyse auf aktuellem Bild
    float2 gpuOffset = make_float2(static_cast<float>(ctx.offset.x), static_cast<float>(ctx.offset.y));
    float2 gpuNewOffset = gpuOffset;

    // 1. Starte Mandelbrot-Kernel, liefert Iterationsbuffer für Heatmap-Analyse
    CudaInterop::renderCudaFrame(
        ctx.d_iterations,
        ctx.d_entropy,
        ctx.d_contrast,
        ctx.width,
        ctx.height,
        static_cast<float>(ctx.zoom),
        gpuOffset,
        ctx.maxIterations,
        ctx.h_entropy,
        ctx.h_contrast,
        gpuNewOffset,
        ctx.shouldZoom,
        ctx.tileSize,
        ctx.supersampling,
        state,
        state.d_tileSupersampling,
        state.h_tileSupersampling
    );
    // 2. Offset-Update nach erfolgreichem Zoom
    if (ctx.shouldZoom) {
        ctx.newOffset.x = gpuNewOffset.x;
        ctx.newOffset.y = gpuNewOffset.y;
    }

    if (Settings::debugLogging) {
        std::printf("[CUDA] Input: offset=(%.10f, %.10f) | zoom=%.2f\n",
            ctx.offset.x, ctx.offset.y, ctx.zoom);
    }    
}

void applyZoomLogic(FrameContext& ctx, CommandBus& zoomBus) {
    double2 diff = { ctx.newOffset.x - ctx.offset.x, ctx.newOffset.y - ctx.offset.y };
    double dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);

    if (Settings::debugLogging) {
        std::printf("[Logic] Start | shouldZoom=%d | Zoom=%.2f | dO=%.4e\n",
               ctx.shouldZoom ? 1 : 0, ctx.zoom, dist);
    }

    if (!ctx.shouldZoom) return;

    if (dist < Settings::DEADZONE) {
        if (Settings::debugLogging) {
            std::printf("[Logic] Offset in DEADZONE (%.4e) → no movement\n", dist);
        }
        return;
    }

    double stepScale = std::tanh(Settings::OFFSET_TANH_SCALE * dist);
    double2 step = { diff.x * stepScale * Settings::MAX_OFFSET_FRACTION,
                     diff.y * stepScale * Settings::MAX_OFFSET_FRACTION };

    if (Settings::debugLogging) {
        double stepLen = std::sqrt(step.x * step.x + step.y * step.y);
        std::printf("[Logic] Step len=%.4e | Zoom += %.5f\n",
               stepLen, Settings::AUTOZOOM_SPEED);
    }

    ctx.offset.x += step.x;
    ctx.offset.y += step.y;
    ctx.zoom *= Settings::AUTOZOOM_SPEED;

    ZoomCommand cmd;
    cmd.frameIndex = globalFrameCounter;
    cmd.oldOffset = make_float2(static_cast<float>(ctx.offset.x), static_cast<float>(ctx.offset.y));
    cmd.newOffset = make_float2(static_cast<float>(ctx.newOffset.x), static_cast<float>(ctx.newOffset.y));
    cmd.zoomBefore = static_cast<float>(ctx.zoom / Settings::AUTOZOOM_SPEED);
    cmd.zoomAfter = static_cast<float>(ctx.zoom);
    cmd.entropy = ctx.lastEntropy;
    cmd.contrast = ctx.lastContrast;
    cmd.tileIndex = ctx.lastTileIndex;

    zoomBus.push(cmd);
    ctx.timeSinceLastZoom = 0.0;
}

void drawFrame(FrameContext& ctx, GLuint tex, RendererState& state) {
    // 🥝 Kiwi: Overlay wird **nur** hier aufgerufen, basierend auf aktuellem Buffer nach Render
    if (ctx.overlayActive) {
        HeatmapOverlay::drawOverlay(
            ctx.h_entropy,
            ctx.h_contrast,
            ctx.width,
            ctx.height,
            ctx.tileSize,
            tex,
            state
        );
    }
    RendererPipeline::drawFullscreenQuad(tex);
}
