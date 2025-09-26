///// Otter: HUD text builder – zoom, offset, FPS (with capped max), and tile stats.
///// Schneefuchs: MAUS header; pch first; ASCII-only; safe snprintf append without NUL.
///// Maus: Keep API stable; compute tiles robustly for tileSize<=0; small fixed buffer lines.
///// Datei: src/hud_text.cpp

#include "pch.hpp"
#include "hud_text.hpp"
#include <algorithm>
#include <cstdio>
#include <string>

#include "fps_meter.hpp"
#include "frame_context.hpp"   // WICHTIG: liefert die Definition von FrameContext
#include "renderer_state.hpp"  // dito fuer RendererState

namespace HudText {

static inline void appendKV(std::string& buf, const char* label, const char* value) {
    char line[96];
    const int n = std::snprintf(line, sizeof(line), "%10s  %-18s\n", label, value);
    if (n > 0) {
        // Append at most sizeof(line)-1 to avoid embedding the terminating NUL
        const size_t toAppend = static_cast<size_t>(std::min(n, static_cast<int>(sizeof(line) - 1)));
        buf.append(line, toAppend);
    }
}

std::string build(const FrameContext& ctx, const RendererState& state) {
    std::string hud;
    hud.reserve(256);

    { char v[64]; std::snprintf(v, sizeof(v), "%.6e", static_cast<double>(ctx.zoom)); appendKV(hud, "zoom", v); }
    { char v[64]; std::snprintf(v, sizeof(v), "%.4f, %.4f", static_cast<double>(ctx.offset.x), static_cast<double>(ctx.offset.y)); appendKV(hud, "offset", v); }

    // fps actual (max) – aus Host-Framezeit in ms (state.lastTimings.frameTotalMs)
    {
        const double dtMs = (state.lastTimings.frameTotalMs > 0.0) ? state.lastTimings.frameTotalMs : 1000.0; // Fallback 1s
        const double fps  = 1000.0 / dtMs;
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
