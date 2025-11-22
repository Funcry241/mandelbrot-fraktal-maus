///// Otter: ASM HUD probe – tiny tilesX×tilesY Mandelbrot grid, separate from CUDA main render.
/// // Schneefuchs: Kapselt Mapping + ASM-Iterationen; keine OpenGL-/HUD-Abhängigkeiten hier.
/// // Maus: Liefert nur ein std::vector<float> (size = tilesX * tilesY); ASCII-only, no logs.
/// // Datei: src/asm/asm_hud_probe.hpp

#pragma once

#include <vector>

// Forward declarations to avoid schwere Includes hier.
struct RendererState;
struct FrameContext;

namespace asm_hud_probe
{
    /// \brief Berechnet ein ASM-basiertes Mini-Fraktal-Grid für das HUD.
    ///
    /// Contract:
    /// - Rechnet ein eigenes tilesX × tilesY Grid.
    /// - Für jedes Tile wird genau ein Sample (Mittenpixel) genommen.
    /// - Pixel→Complex-Mapping wird aus RendererState + FrameContext abgeleitet.
    /// - Intern wird (in der .cpp) mandelbrotIter_asm(x0, y0, maxIterHud) verwendet.
    /// - Kein OpenGL, keine Textur-Objekte, kein HUD-Zeichnen – nur Daten.
    ///
    /// Resultat:
    /// - Rückgabe-Vector hat exakt tilesX * tilesY Einträge.
    /// - Layout ist row-major: index = y * tilesX + x.
    ///
    /// Fehler/Robustheit:
    /// - Bei tilesX <= 0 oder tilesY <= 0 oder maxIterHud <= 0 wird ein leerer Vector zurückgegeben.
    ///
    /// Hinweis:
    /// - Logging (falls nötig) passiert ausschließlich in der .cpp (ASCII only).
    [[nodiscard]]
    std::vector<float> buildHudProbeGrid(
        const RendererState& rendererState,
        const FrameContext&  frameCtx,
        int                  tilesX,
        int                  tilesY,
        int                  maxIterHud);
} // namespace asm_hud_probe
