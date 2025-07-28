// Datei: src/frame_pipeline.hpp
/* 🐭 interner Maus-Kommentar:
Schnittstelle für die modulare Frame-Pipeline.
Deklariert klar getrennte Schritte:

    Frame-Beginn (Zeit, Init)
    CUDA-Rendering
    ZoomLogik (mit CommandBus)
    Bildausgabe (Heatmap & Fraktal)

    -> Alles basiert auf FrameContext, keine globalen Zustände.
    ❤️ FIX: computeCudaFrame explizit mit RendererState - Maus liebt Präzision, Schneefuchs liebt Klarheit.
    ❤️ FIX: drawFrame braucht jetzt RendererState für HeatmapOverlay (neuer Parameter).
*/

#pragma once
#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "renderer_state.hpp"

namespace FramePipeline {

// 🔁 Frame-Start (Zeit, Init)
void beginFrame(FrameContext& ctx);

// ⚙️ CUDA Rendering
void computeCudaFrame(FrameContext& ctx, RendererState& state);

// 🌀 Zoom-Verarbeitung + Zustand aktualisieren
void applyZoomLogic(FrameContext& ctx, CommandBus& zoomBus, RendererState& state);

// 🎨 Ausgabe: Fraktal + Heatmap + Overlays
void drawFrame(FrameContext& ctx, GLuint tex, RendererState& state);

// 🚀 Komplettes Frame ausführen (alle Schritte, konsistent)
void execute(RendererState& state);

} // namespace FramePipeline
