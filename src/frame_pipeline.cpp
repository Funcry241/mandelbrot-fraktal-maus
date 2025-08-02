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

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Calling CudaInterop::renderCudaFrame");
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
        LUCHS_LOG_HOST("[PIPE] Returned from renderCudaFrame");
    CUDA_CHECK(cudaDeviceSynchronize());

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Cuda frame compute completed, flushing device logs");
    LuchsLogger::flushDeviceLogToHost(0);

    if (frameCtx.shouldZoom) {
        frameCtx.newOffset = { gpuNewOffset.x, gpuNewOffset.y };
    }

    if (Settings::debugLogging && !frameCtx.h_entropy.empty()) {
        LUCHS_LOG_HOST("[PIPE] Heatmap sample: Entropy[0]=%.4f Contrast[0]=%.4f", frameCtx.h_entropy[0], frameCtx.h_contrast[0]);
    }
}

void applyZoomLogic(FrameContext& frameCtx, CommandBus& bus) {
    // unchanged
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

// 🦦 Otter: Klarer Datenfluss. Textur wird sauber aus dem PBO aktualisiert. Alle Overlays sind korrekt integriert.
// 🐭 Maus: drawFrame ist zentral – hier entscheidet sich, was sichtbar ist. Kein PBO-Upload = schwarze Fläche.
// 🦊 Schneefuchs: Reihenfolge beachtet: PBO → Textur → Overlay → Draw. Nichts implizit, alles kontrolliert.
void drawFrame(FrameContext& frameCtx, GLuint tex, RendererState& state) {
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] drawFrame: overlayActive=%d, texID=%u", frameCtx.overlayActive ? 1 : 0, tex);

    // 1️⃣ CUDA hat vorher in den OpenGL-PBO geschrieben.
    //     Jetzt übertragen wir den Inhalt explizit in die OpenGL-Textur (GL_TEXTURE_2D),
    //     da OpenGL nicht direkt aus dem PBO rendert.
    OpenGLUtils::setGLResourceContext("frame");
    OpenGLUtils::updateTextureFromPBO(state.pbo.id(), tex, frameCtx.width, frameCtx.height);

    // 2️⃣ Optionales Overlay: Heatmap basierend auf Entropie-/Kontrastdaten.
    //     Wird auf Basis der aktualisierten Textur gezeichnet.
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

    // 3️⃣ Optionales Overlay: Warzenschwein (z.B. Debug- oder HUD-Anzeige).
    //     Kommt immer obendrauf, unabhängig von der Heatmap.
    if (Settings::warzenschweinOverlayEnabled)
        WarzenschweinOverlay::drawOverlay(state);

    // 4️⃣ Jetzt wird das finale Bild (inkl. Overlays) auf den Screen gezeichnet.
    //     Das Fullscreen-Quad nutzt die zuvor aktualisierte Textur.
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Calling RendererPipeline::drawFullscreenQuad");
    RendererPipeline::drawFullscreenQuad(tex);
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] Completed drawFullscreenQuad");
}

void execute(RendererState& state) {
    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] execute start");
    beginFrame(g_ctx);

    // Schwarze Ameise: Explizite Synchronisierung aller relevanten FrameContext-Parameter aus RendererState
    g_ctx.width        = state.width;
    g_ctx.height       = state.height;
    g_ctx.tileSize     = state.lastTileSize;
    g_ctx.zoom         = static_cast<float>(state.zoom);
    g_ctx.offset       = state.offset;
    g_ctx.maxIterations = state.maxIterations;

    computeCudaFrame(g_ctx, state);
    applyZoomLogic(g_ctx, g_zoomBus);
    drawFrame(g_ctx, state.tex.id(), state);

    if (Settings::debugLogging)
        LUCHS_LOG_HOST("[PIPE] execute complete");
}

} // namespace FramePipeline
