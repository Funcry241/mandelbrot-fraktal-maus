// Datei: src/frame_pipeline.cpp
// Zeilen: 127
// 🐭 Maus-Kommentar: Kiwi – Heatmap-Logik nur nach Render, alle Daten aktuell. Keine DrawOverlay-Altpfade mehr. Schneefuchs validiert: Framedaten, Overlay und CUDA sauber synchron.
// Alpha 42: Explizite Includes für Namespace-Auflösung (Otter-Fix für VSCode-IntelliSense).
// Otter-Sonderlog: Host-seitige Kernel-Parameter vor jedem Render für Debug-Analyse
#include "pch.hpp"
#include "cuda_interop.hpp"
#include "renderer_pipeline.hpp"

#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "heatmap_overlay.hpp"
#include "settings.hpp"
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector_types.h>
#include <vector>

static int globalFrameCounter = 0;

void beginFrame(FrameContext& ctx) {
    // Fix für C4244: explizite float-Casts auf alle double -> float Umwandlungen
    float delta = static_cast<float>(glfwGetTime() - ctx.totalTime);
    ctx.frameTime = delta;
    ctx.totalTime += delta;
    ctx.timeSinceLastZoom += delta;
    ctx.shouldZoom = false;
    ctx.newOffset = ctx.offset;
    if (ctx.frameTime < 0.001f) ctx.frameTime = 0.001f;
    ++globalFrameCounter;
}

void computeCudaFrame(FrameContext& ctx, RendererState& state) {
    // 🥝 Kiwi: Zuerst Mandelbrot-Kernel, dann Entropie/Kontrast-Analyse auf aktuellem Bild
    float2 gpuOffset = make_float2(static_cast<float>(ctx.offset.x), static_cast<float>(ctx.offset.y));
    float2 gpuNewOffset = gpuOffset;

    // Otter-Log: Host-seitig, ALLE Kernel-Parameter ausgeben
    if (Settings::debugLogging) {
        std::printf("[DEBUG] Mandelbrot-Kernel Call: width=%d, height=%d, maxIter=%d, zoom=%.2f, offset=(%.10f, %.10f), tileSize=%d, supersampling=%d\n",
            ctx.width, ctx.height, ctx.maxIterations, ctx.zoom,
            ctx.offset.x, ctx.offset.y, ctx.tileSize, ctx.supersampling);
    }

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

    // Host-Debug nach CUDA-Render
    if (Settings::debugLogging) {
        std::printf("[CUDA] Input: offset=(%.10f, %.10f) | zoom=%.2f\n",
            ctx.offset.x, ctx.offset.y, ctx.zoom);
        if (!ctx.h_entropy.empty()) {
            std::printf("[Heatmap] Entropy[0]=%.4f Contrast[0]=%.4f\n",
                ctx.h_entropy[0], ctx.h_contrast[0]);
        }
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

    // Explizite Casts um C4244 zu vermeiden (double->float)
    ctx.offset.x = static_cast<float>(ctx.offset.x + step.x);
    ctx.offset.y = static_cast<float>(ctx.offset.y + step.y);
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
    ctx.timeSinceLastZoom = 0.0f;
}

void drawFrame(FrameContext& ctx, GLuint tex, RendererState& state) {
    // 🥝 Kiwi: Overlay wird nur hier aufgerufen, basierend auf aktuellem Buffer nach Render
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
