// Datei: src/frame_pipeline.cpp
// Zeilen: 99
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
   - FIX: RendererPipeline::updateTexture eingefügt nach CUDA-Kernel (Otter-Fund)
   - FIX: <cmath> eingebunden, manuelle Operatorfunktionen entfernt (Otter)
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
#include <cmath>  // ✅ ersetzt manuelle sqrt/tanh-Berechnungen

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
        state
    );

    if (Settings::debugLogging) {
        printf("[CUDA] Input: offset=(%.10f, %.10f) | zoom=%.2f\n",
               ctx.offset.x, ctx.offset.y, ctx.zoom);
    }

    RendererPipeline::updateTexture(state.pbo, state.tex, ctx.width, ctx.height);
}

void applyZoomLogic(FrameContext& ctx, CommandBus& zoomBus) {
    double2 diff = { ctx.newOffset.x - ctx.offset.x, ctx.newOffset.y - ctx.offset.y };
    double dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);

    if (Settings::debugLogging) {
        printf("[Logic] Start | shouldZoom=%d | Zoom=%.2f | dO=%.4e\n",
               ctx.shouldZoom, ctx.zoom, dist);
    }

    if (!ctx.shouldZoom) return;

    if (dist < Settings::DEADZONE) {
        if (Settings::debugLogging) {
            printf("[Logic] Offset in DEADZONE (%.4e) → no movement\n", dist);
        }
        return;
    }

    double stepScale = std::tanh(Settings::OFFSET_TANH_SCALE * dist);
    double2 step = { diff.x * stepScale * Settings::MAX_OFFSET_FRACTION,
                     diff.y * stepScale * Settings::MAX_OFFSET_FRACTION };

    if (Settings::debugLogging) {
        double stepLen = std::sqrt(step.x * step.x + step.y * step.y);
        printf("[Logic] Step len=%.4e | Zoom += %.5f\n",
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
