///// Otter: ASM HUD overlay – draws tiny tilesX×tilesY Mandelbrot grid (ASM-based) as mini-panel.
/// /// Schneefuchs: Kapselt nur Draw/GL; liest Grid aus RendererState; keine ASM-/CUDA-Abhängigkeiten.
/// /// Maus: Rechts unten im Screen; ASCII-only Logs; tut nichts, wenn Grid leer/inkonsistent.
/// /// Datei: src/asm/asm_hud_overlay.hpp

#pragma once

struct RendererState;
struct FrameContext;

namespace asm_hud_overlay
{
    /// \brief Zeichnet das ASM-HUD-Mini-Panel rechts unten.
    ///
    /// Erwartet:
    /// - RendererState.asmHudGrid (size = asmHudTilesX * asmHudTilesY)
    /// - RendererState.asmHudTilesX, asmHudTilesY > 0
    ///
    /// Verhalten:
    /// - Wenn Grid leer oder inkonsistent ist, passiert nichts.
    /// - Nutzt eine eigene kleine 2D-Textur mit tilesX×tilesY Auflösung.
    /// - Panel wird in Pixelkoordinaten unten rechts positioniert.
    ///
    /// Nebenwirkungen:
    /// - OpenGL-State wird lokal angepasst und am Ende weitgehend restauriert
    ///   (Program, VAO, Array-Buffer, Blend-State).
    void draw(const RendererState& state, const FrameContext& ctx);
} // namespace asm_hud_overlay
