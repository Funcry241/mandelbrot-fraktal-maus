///// Otter: HUD-Text – kompakte ASCII-Anzeige (zoom, offset, fps, tiles); robust gegen edge cases.
///// Schneefuchs: Header mit vollständigen Typdefinitionen; deterministisch; keine iostreams.
///// Maus: Nur LUCHS_LOG_* im Projekt; hier keine Logs, reine Textformatierung.

#include "hud_text.hpp"
#include <algorithm>
#include <cstdio>
#include <string>

#include "fps_meter.hpp"
#include "frame_context.hpp"   // ← WICHTIG: liefert die *Definition* von FrameContext
#include "renderer_state.hpp"  // ← dito für RendererState (auch wenn aktuell ungenutzt)

namespace HudText {

static inline void appendKV(std::string& buf, const char* label, const char* value) {
    char line[96];
    const int n = std::snprintf(line, sizeof(line), "%10s  %-18s\n", label, value);
    if (n > 0) buf.append(line, static_cast<size_t>(std::min(n, static_cast<int>(sizeof(line)))));
}

std::string build(const FrameContext& ctx, const RendererState& state) {
    (void)state; // reserviert für spätere Erweiterungen
    std::string hud;
    hud.reserve(256);

    { char v[64]; std::snprintf(v, sizeof(v), "%.6e", static_cast<double>(ctx.zoom)); appendKV(hud, "zoom", v); }
    { char v[64]; std::snprintf(v, sizeof(v), "%.4f, %.4f", static_cast<double>(ctx.offset.x), static_cast<double>(ctx.offset.y)); appendKV(hud, "offset", v); }

    // fps actual (max) – Schutz gegen dt <= 0; max aus FpsMeter
    {
        const double dt  = (ctx.frameTime > 0.0) ? static_cast<double>(ctx.frameTime) : 1.0;
        const double fps = 1.0 / dt;
        const int    maxFpsInt = FpsMeter::currentMaxFpsInt();
        char v[64];
        std::snprintf(v, sizeof(v), "%.1f (%d)", fps, maxFpsInt);
        appendKV(hud, "fps", v);
    }

    // Tiles aus width/height + tileSize; robust gegen tileSize<=0
    {
        char v[64];
        if (ctx.tileSize > 0) {
            const int tilesX   = (ctx.width  + ctx.tileSize - 1) / ctx.tileSize;
            const int tilesY   = (ctx.height + ctx.tileSize - 1) / ctx.tileSize;
            const int numTiles = tilesX * tilesY;
            std::snprintf(v, sizeof(v), "%d x %d (%d)", tilesX, tilesY, numTiles);
        } else {
            std::snprintf(v, sizeof(v), "n/a");
        }
        appendKV(hud, "tiles", v);
    }

    return hud;
}

} // namespace HudText
