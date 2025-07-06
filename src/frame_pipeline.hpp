// Datei: src/frame_pipeline.hpp
// Zeilen: 43
/* 🐭 interner Maus-Kommentar:
Schnittstelle für die modulare Frame-Pipeline.
Deklariert klar getrennte Schritte:

    Frame-Beginn (Zeit, Init)

    CUDA-Rendering

    ZoomLogik (mit CommandBus)

    Bildausgabe (Heatmap & Fraktal)
    → Alles basiert auf FrameContext, keine globalen Zustände.
    ❤️ FIX: computeCudaFrame explizit mit RendererState – Maus liebt Präzision, Schneefuchs liebt Klarheit.
    ❤️ FIX: drawFrame braucht jetzt RendererState für HeatmapOverlay (neuer Parameter).
    */

#pragma once
#include "frame_context.hpp"
#include "zoom_command.hpp"
#include "renderer_state.hpp"

void beginFrame(FrameContext& ctx);

// Volle Signatur für CUDA-Frame – braucht explizit RendererState!
void computeCudaFrame(FrameContext& ctx, RendererState& state);

// Wendet Zoom-Logik an, erstellt ZoomCommand, aktualisiert Zustand
void applyZoomLogic(FrameContext& ctx, CommandBus& zoomBus);

// Bild + Overlay ausgeben (immer RendererState mitgeben)
void drawFrame(FrameContext& ctx, GLuint tex, RendererState& state);
