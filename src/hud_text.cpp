///// Otter: HUD-Text – kompakte Center-Statistik (3 Zeilen), deterministisch formatiert.
///// Schneefuchs: ASCII-only; pch zuerst; keine GL- oder Device-Abhängigkeiten; /WX clean.
///// Maus: Feste Präzision (cx/cy 9, z 6, fps 1); fallback auf dt für FPS falls nötig.
///// Datei: src/hud_text.cpp

#include "pch.hpp"
#include "hud_text.hpp"

#include "frame_context.hpp"
#include "renderer_state.hpp"
#include "settings.hpp"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <string>

namespace HudText {

static inline double safe_fps_from_ms(double ms) noexcept {
    return (ms > 1e-9) ? (1000.0 / ms) : 0.0;
}

std::string build(const FrameContext& fctx, const RendererState& state) {
    // Daten einsammeln (nur Host-Seite; keine GL/CUDA-Aufrufe)
    const double cx   = static_cast<double>(state.center.x);
    const double cy   = static_cast<double>(state.center.y);
    const double zoom = static_cast<double>(fctx.zoom);
    const int    it   = fctx.maxIterations;
    const int    tile = std::max(1, fctx.tileSize);
    const int    w    = fctx.width;
    const int    h    = fctx.height;

    // FPS primär aus gemessener Framezeit, sonst aus dt schätzen
    double fps = safe_fps_from_ms(state.lastTimings.frameTotalMs);
    if (fps <= 0.0 && fctx.deltaSeconds > 0.0f) {
        fps = 1.0 / static_cast<double>(fctx.deltaSeconds);
    }

    // Kompakt & stabil formatiert (ASCII)
    char line1[96], line2[96], line3[96];
    std::snprintf(line1, sizeof(line1), "cx=%.9f cy=%.9f", cx, cy);
    std::snprintf(line2, sizeof(line2), "z=%.6f it=%d tile=%d", zoom, it, tile);
    std::snprintf(line3, sizeof(line3), "res=%dx%d fps=%.1f", w, h, fps);

    std::string out;
    out.reserve(96 + 96 + 96);
    out.append(line1);
    out.push_back('\n');
    out.append(line2);
    out.push_back('\n');
    out.append(line3);
    return out;
}

} // namespace HudText
