// Datei: src/frame_pipeline.hpp
// Zeilen: 43
/* 🐭 interner Maus-Kommentar:
   Schnittstelle für die modulare Frame-Pipeline.
   Deklariert klar getrennte Schritte:
   - Frame-Beginn (Zeit, Init)
   - CUDA-Rendering
   - ZoomLogik (mit CommandBus)
   - Bildausgabe (Heatmap & Fraktal)
   → Alles basiert auf FrameContext, keine globalen Zustände.
   ❤️ FIX: computeCudaFrame explizit mit RendererState – Maus liebt Präzision, Schneefuchs liebt Klarheit.
   ❤️ FIX: drawFrame braucht jetzt RendererState für HeatmapOverlay (neuer Parameter).
*/

#pragma once
#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "renderer_state.hpp"   // ✅ Für computeCudaFrame & drawFrame – explizit nötig

void beginFrame(FrameContext& ctx);

// ✅ FIX: vollständige Signatur mit RendererState – sonst erkennt .cpp den Aufruf nicht
void computeCudaFrame(FrameContext& ctx, RendererState& state);

// Wendet Zoomentscheidung an, erstellt ZoomCommand und aktualisiert Zustand.
void applyZoomLogic(FrameContext& ctx, CommandBus& zoomBus);

// Zeichnet das Bild auf den Bildschirm (inkl. Heatmap bei Bedarf).
void drawFrame(FrameContext& ctx, GLuint tex, RendererState& state);
