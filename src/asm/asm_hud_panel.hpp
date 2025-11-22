///// Otter: ASM HUD panel – bottom-right mini-fractal grid; uses precomputed ASM tiles.
///// Schneefuchs: Header leicht – keine GL-Includes, kein ASM-Compute; klarer Draw-Einstieg.
///// Maus: Panel-API: draw(state, width, height); Grid via RendererState::asmHudGrid; ASCII-only.
///// Datei: src/asm/asm_hud_panel.hpp
#pragma once

struct RendererState;

namespace AsmHudPanel
{
    // Zeichnet das ASM-Mini-Panel unten rechts.
    //
    // Erwartet:
    //  - state.asmHudTilesX/Y > 0
    //  - state.asmHudGrid.size() == asmHudTilesX * asmHudTilesY
    //
    // viewportWidth/viewportHeight sind die aktuelle Frame-Auflösung (Framebuffer).
    void draw(const RendererState& state, int viewportWidth, int viewportHeight);
} // namespace AsmHudPanel
