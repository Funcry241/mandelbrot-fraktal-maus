#include "hud_text.hpp"
#include <algorithm>
#include <cstdio>
#include "fps_meter.hpp"

namespace HudText {

static inline void appendKV(std::string& buf, const char* label, const char* value) {
    char line[96];
    const int n = std::snprintf(line, sizeof(line), "%10s  %-18s\n", label, value);
    if (n > 0) buf.append(line, (size_t)std::min(n, (int)sizeof(line)));
}

std::string build(const FrameContext& ctx, const RendererState& state) {
    (void)state; // aktuell nicht benötigt, reserviert für spätere Erweiterungen
    std::string hud;
    hud.reserve(256);

    { char v[64]; std::snprintf(v, sizeof(v), "%.6e", (double)ctx.zoom); appendKV(hud, "zoom", v); }
    { char v[64]; std::snprintf(v, sizeof(v), "%.4f, %.4f", (double)ctx.offset.x, (double)ctx.offset.y); appendKV(hud, "offset", v); }
    // { char v[64]; std::snprintf(v, sizeof(v), "%.3f", (double)ctx.lastEntropy); appendKV(hud, "entropy", v); }
    // { char v[64]; std::snprintf(v, sizeof(v), "%.3f", (double)ctx.lastContrast); appendKV(hud, "contrast", v); }

    // fps actual (max) – Klammern sicher vorhanden (Font ergänzt)
    {
        const double fps = (ctx.frameTime > 0.0f) ? (1.0 / ctx.frameTime) : 0.0;
        const int    maxFpsInt = FpsMeter::currentMaxFpsInt();
        char v[64];
        std::snprintf(v, sizeof(v), "%.1f (%d)", fps, maxFpsInt);
        appendKV(hud, "fps", v);
    }

    // Tiles aus EINER Quelle (ctx.width/height + ctx.tileSize)
    {
        const int tilesX   = (ctx.width  + ctx.tileSize - 1) / ctx.tileSize;
        const int tilesY   = (ctx.height + ctx.tileSize - 1) / ctx.tileSize;
        const int numTiles = tilesX * tilesY;
        char v[64];
        std::snprintf(v, sizeof(v), "%d x %d (%d)", tilesX, tilesY, numTiles);
        appendKV(hud, "tiles", v);
    }

    return hud;
}

} // namespace HudText
