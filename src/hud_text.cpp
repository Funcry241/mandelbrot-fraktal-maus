///// Otter: HUD-Text – kompakte Center-Statistik (3 Zeilen), deterministisch formatiert.
///// Schneefuchs: ASCII-only; pch zuerst; keine GL- oder Device-Abhängigkeiten; /WX clean; C-Locale wird vorausgesetzt.
///// Maus: Feste Präzision (cx/cy 9, z wissenschaftlich %.3e, fps 1); Fallback auf dt für FPS falls nötig.
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

namespace AOP_Telemetry {
    // Phase-1 Telemetrie (extern definiert im AOP-Controller / Warzenschwein-Overlay)
    extern float            g_ai_last_delta;
    extern int              g_ai_ov_valid;
    extern unsigned long long g_ai_frame_id;
}

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
    const int    tile = std::max(1, static_cast<int>(fctx.tileSize));
    const int    w    = fctx.width;
    const int    h    = fctx.height;

    // Heatmap-Infos (spiegeln die [PERF]-Zeile)
    const size_t hmN     = state.h_entropy.size();
    const int    statsPx = std::max(1, fctx.statsTileSize);

    // ROI/Policy-Delta (P1 mini – nur Anzeige)
    const int   roiValid = AOP_Telemetry::g_ai_ov_valid;
    const float delta    = AOP_Telemetry::g_ai_last_delta;

    // FPS primär aus gemessener Framezeit, sonst aus dt schätzen
    double fps = safe_fps_from_ms(state.lastTimings.frameTotalMs);
    if (fps <= 0.0 && fctx.deltaSeconds > 0.0f) {
        fps = 1.0 / static_cast<double>(fctx.deltaSeconds);
    }

    // Kompakt & stabil formatiert (ASCII; C-Locale erwartet)
    char line1[96], line2[128], line3[128];

    // Zeile 1: Center
    std::snprintf(line1, sizeof(line1), "cx=%.9f cy=%.9f", cx, cy);

    // Zeile 2: Zoom/Iter/Tile + hmN/statsPx (nur Zoom dezent selbsterklärend)
    std::snprintf(line2, sizeof(line2), "zoom=%.3e x it=%d tile=%d hmN=%zu statsPx=%d",
                  zoom, it, tile, hmN, statsPx);

    // Zeile 3: Auflösung/FPS + ROI/Delta (Delta nur bei gültigem ROI – sonst "--")
    char dstr[16];
    if (roiValid) {
        std::snprintf(dstr, sizeof(dstr), "%.3f", static_cast<double>(delta));
    } else {
        std::snprintf(dstr, sizeof(dstr), "--");
    }
    // FPS jetzt feste Breite: %6.1f (z.B. "  94.6", " 999.9")
    std::snprintf(line3, sizeof(line3), "res=%dx%d fps=%6.1f ROI=%d d=%s",
                  w, h, fps, roiValid, dstr);

    std::string out;
    out.reserve(sizeof(line1) + sizeof(line2) + sizeof(line3));
    out.append(line1);
    out.push_back('\n');
    out.append(line2);
    out.push_back('\n');
    out.append(line3);
    return out;
}

} // namespace HudText
