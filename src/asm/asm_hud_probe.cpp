///// Otter: ASM HUD probe – tiny Mandelbrot tilesX×tilesY grid, separate from CUDA main render.
///// Schneefuchs: Kapselt ASM-Iterationen in eigenem Modul; nutzt Capybara-Mapping (cx/stepX/stepY) hostseitig.
///// Maus: Nutzt mandelbrotIter_asm(...) pro Tile; liefert nur std::vector<float>; ASCII-only.
/// /// Datei: src/asm/asm_hud_probe.cpp

#include "pch.hpp"
#include "asm_hud_probe.hpp"
#include "renderer_state.hpp"
#include "frame_context.hpp"
#include "capybara_mapping.cuh"

#include <cassert>

namespace
{
    // Otter: Externe ASM-Funktion – Signatur ggf. an dein iter.asm anpassen.
    extern "C" int mandelbrotIter_asm(double x0, double y0, int maxIter);

    struct ComplexCoord
    {
        double x;
        double y;
    };

    // Maus: Host-Mapping für das ASM-Mini-Grid.
    // Nutzt exakt dieselben Parameter wie der Capybara-Renderpfad:
    //  - cx, cy      aus RendererState::center
    //  - zoom        aus RendererState::zoom
    //  - pixelScale  aus RendererState::pixelScale
    //  - capy_pixel_steps_from_zoom_scale(...) + capy_map_pixel_double(...)
    ComplexCoord mapHudTileToComplex(
        const RendererState& state,
        const FrameContext&  frame,
        int                  tilesX,
        int                  tilesY,
        int                  tx,
        int                  ty)
    {
        ComplexCoord out{0.0, 0.0};

        const int w = frame.width;
        const int h = frame.height;
        if (w <= 0 || h <= 0 || tilesX <= 0 || tilesY <= 0) {
            return out;
        }

        // Index-Guard – lieber defensiv, damit kein UB entsteht.
        if (tx < 0 || tx >= tilesX || ty < 0 || ty >= tilesY) {
            return out;
        }

        // Otter: Kachelgröße in Pixeln über den gesamten Bildschirm.
        const double tileW = static_cast<double>(w) / static_cast<double>(tilesX);
        const double tileH = static_cast<double>(h) / static_cast<double>(tilesY);

        // Kachelmitte im Screenraum (Pixelkoordinate, 0..w/h).
        const double centerPx = (static_cast<double>(tx) + 0.5) * tileW;
        const double centerPy = (static_cast<double>(ty) + 0.5) * tileH;

        // Integer-Pixel wählen – capy_map_pixel_double nutzt selbst +0.5 (Pixelzentrum),
        // daher hier KEIN weiteres +0.5, nur Clamping.
        int px = static_cast<int>(centerPx);
        int py = static_cast<int>(centerPy);

        if (px < 0)      px = 0;
        if (py < 0)      py = 0;
        if (px >= w)     px = w - 1;
        if (py >= h)     py = h - 1;

        // Schneefuchs: Schrittweiten exakt wie im CUDA-Renderpfad (render_to_pbo_core).
        const double sx   = static_cast<double>(state.pixelScale.x);
        const double sy   = static_cast<double>(state.pixelScale.y);
        const double zoom = state.zoom;

        double stepX = 0.0;
        double stepY = 0.0;
        capy_pixel_steps_from_zoom_scale(sx, sy, w, zoom, stepX, stepY);

        const double cx = state.center.x;
        const double cy = state.center.y;

        const double2 c = capy_map_pixel_double(cx, cy, stepX, stepY, px, py, w, h);
        out.x = c.x;
        out.y = c.y;
        return out;
    }

} // namespace

namespace asm_hud_probe
{
    std::vector<float> buildHudProbeGrid(
        const RendererState& rendererState,
        const FrameContext&  frameCtx,
        int                  tilesX,
        int                  tilesY,
        int                  maxIterHud)
    {
        // Basic guard: ungültige Parameter -> leeres Grid, kein Crash.
        if (tilesX <= 0 || tilesY <= 0 || maxIterHud <= 0) {
            return {};
        }

        const int totalTiles = tilesX * tilesY;
        std::vector<float> result;
        result.resize(static_cast<std::size_t>(totalTiles));

        for (int ty = 0; ty < tilesY; ++ty)
        {
            for (int tx = 0; tx < tilesX; ++tx)
            {
                const int index = ty * tilesX + tx;
                assert(index >= 0 && index < totalTiles);

                // Maus: Complex-Koordinate für die Kachelmitte holen (Capybara-konform).
                const ComplexCoord cc = mapHudTileToComplex(
                    rendererState,
                    frameCtx,
                    tilesX,
                    tilesY,
                    tx,
                    ty);

                // Otter: ASM-Iteration einmal pro Tile.
                const int iter = mandelbrotIter_asm(cc.x, cc.y, maxIterHud);

                // Schneefuchs: Clamp gegen kaputte Rückgaben aus ASM.
                const int clampedIter =
                    (iter < 0) ? 0 :
                    (iter > maxIterHud) ? maxIterHud :
                    iter;

                result[static_cast<std::size_t>(index)] =
                    static_cast<float>(clampedIter);
            }
        }

        return result;
    }

} // namespace asm_hud_probe
