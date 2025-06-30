// Datei: src/frame_pipeline.cpp
// Zeilen: 90
/* 🐭 interner Maus-Kommentar:
   Diese Datei definiert die logische Render-Pipeline:
   - Klar getrennte Schritte (beginFrame, computeCudaFrame, applyZoomLogic, drawFrame)
   - Alle verwenden `FrameContext& ctx`
   - Ziel: deterministisch, testbar, modular – Grundlage für Replay & Analyse.
   - Kein OpenGL/GUI-Code direkt hier – nur Logik!
   - FIX: renderCudaFrame nun explizit mit Namespace (Schneefuchs-Fund)
   - FIX: zoomFactor ersetzt durch AUTOZOOM_SPEED (Schneefuchs-Empfehlung)
   - FIX: HeatmapOverlay::drawOverlay korrekt eingebunden (kein drawOverlayTexture mehr)
   - FIX: float2 durch double2 ersetzt – volle Präzision laut frame_context.hpp
   - FIX: RendererState& state als finaler Parameter eingeführt (wegen Schneefuchs' Analyse)
*/

#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "cuda_interop.hpp"
#include "renderer_pipeline.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

static int globalFrameCounter = 0;

void beginFrame(FrameContext& ctx) {
    ctx.frameTime = glfwGetTime();
    ctx.frameTime -= ctx.totalTime;
    ctx.totalTime += ctx.frameTime;
    ctx.timeSinceLastZoom += ctx.frameTime;
    ctx.shouldZoom = false;
    ctx.newOffset = ctx.offset;
    ctx.frameTime = std::max(0.001, ctx.frameTime); // keine 0-Div
    ++globalFrameCounter;
}

void computeCudaFrame(FrameContext& ctx, RendererState& state) {
    CudaInterop::renderCudaFrame(
        ctx.d_iterations,
        ctx.d_entropy,
        ctx.width,
        ctx.height,
        ctx.zoom,
        ctx.offset,
        ctx.maxIterations,
        ctx.h_entropy,
        ctx.newOffset,
        ctx.shouldZoom,
        ctx.tileSize,
        ctx.supersampling,
        state // ✅ notwendig für Zielwahl und Zoomlogik
    );
}

void applyZoomLogic(FrameContext& ctx, CommandBus& zoomBus) {
    if (!ctx.shouldZoom) return;

    ZoomCommand cmd;
    cmd.frameIndex = globalFrameCounter;
    cmd.oldOffset = make_float2(static_cast<float>(ctx.offset.x), static_cast<float>(ctx.offset.y));
    cmd.newOffset = make_float2(static_cast<float>(ctx.newOffset.x), static_cast<float>(ctx.newOffset.y));
    cmd.zoomBefore = static_cast<float>(ctx.zoom);
    cmd.zoomAfter = static_cast<float>(ctx.zoom * Settings::AUTOZOOM_SPEED);
    cmd.entropy = ctx.lastEntropy;
    cmd.contrast = ctx.lastContrast;
    cmd.tileIndex = ctx.lastTileIndex;

    zoomBus.push(cmd);
    ctx.offset = ctx.newOffset;
    ctx.zoom = cmd.zoomAfter;
    ctx.timeSinceLastZoom = 0.0;
}

void drawFrame(FrameContext& ctx, GLuint tex) {
    if (ctx.overlayActive) {
        HeatmapOverlay::drawOverlay(
            ctx.h_entropy,
            ctx.h_contrast,
            ctx.width,
            ctx.height,
            ctx.tileSize,
            tex
        );
    }

    RendererPipeline::drawFullscreenQuad(tex);
}
