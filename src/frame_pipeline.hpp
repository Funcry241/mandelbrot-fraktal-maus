/* Datei: src/frame_pipeline.hpp
   🐭 Maus: Klare Pipeline-Schnittstelle, kompatibel zum Zoom V2.
   🦦 Otter: computeCudaFrame liefert Entscheidungsgrundlage, applyZoomLogic führt sie aus. (Bezug zu Otter)
   🐑 Schneefuchs: Keine versteckten Abhängigkeiten – Tiles werden pro Frame einmal berechnet. (Bezug zu Schneefuchs)
*/
#pragma once
#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "renderer_state.hpp"
#include <GL/glew.h> // GLuint

namespace FramePipeline {

// 🔁 Frame-Start (Zeit, Init)
void beginFrame(FrameContext& ctx, RendererState& state);

// ⚙️ CUDA Rendering + Heatmap
void computeCudaFrame(FrameContext& ctx, RendererState& state);

// 🌀 Zoom-Verarbeitung + Zustand aktualisieren (Zoom V2)
void applyZoomLogic(FrameContext& ctx, CommandBus& zoomBus, RendererState& state);

// 🎨 Ausgabe: Fraktal + Heatmap + Overlays
void drawFrame(FrameContext& ctx, GLuint tex, RendererState& state);

// 🚀 Komplettes Frame ausführen (alle Schritte, konsistent)
void execute(RendererState& state);

} // namespace FramePipeline
