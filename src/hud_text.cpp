///// Otter: HUD-Text – kompakte Center-Statistik (3 Zeilen), deterministisch formatiert.
///// Schneefuchs: ASCII-only; pch zuerst; keine GL-/Device-Abhängigkeiten; /WX clean; C-Locale nicht vorausgesetzt (ASCII-Dezimalpunkt erzwungen).
///// Maus: Narrow-Classic Layout – Zeile1: Zoom/FPS/ROI/d | Zeile2: Iter/Tile/Grid/Tiles | Zeile3: Center/Res; feste Feldbreiten; FPS-Fallback über dt.
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
    extern float              g_ai_last_delta;
    extern int                g_ai_ov_valid;
    extern unsigned long long g_ai_frame_id;
}

namespace HudText {

static inline double safe_fps_from_ms(double ms) noexcept {
    return (ms > 1e-9) ? (1000.0 / ms) : 0.0;
}

// Erzwingt ASCII-Dezimalpunkt, falls die C-Locale Kommas liefert.
static inline void enforce_ascii_decimal(char* s) noexcept {
    if (!s) return;
    for (char* p = s; *p; ++p) {
        if (*p == ',') *p = '.';
    }
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

    // Kompakt & stabil formatiert (ASCII; feste Feldbreiten)
    char line1[160], line2[160], line3[160];

    // ── Zeile 1: Zoom / FPS / ROI / d ───────────────────────────────────────
    // Zoom: %11.3e (feste Breite), FPS: %6.1f, ROI: %d, d: %5.3f (oder "--")
    char dstr[16];
    if (roiValid) {
        std::snprintf(dstr, sizeof(dstr), "%5.3f", static_cast<double>(delta));
    } else {
        std::snprintf(dstr, sizeof(dstr), "--");
    }
    std::snprintf(line1, sizeof(line1),
                  "Zoom  %11.3e x | FPS %6.1f | ROI %d | d %s",
                  zoom, fps, roiValid, dstr);
    enforce_ascii_decimal(line1);

    // ── Zeile 2: Iter / Tile / Grid / Tiles ─────────────────────────────────
    // Iter: %5d, Tile: %3dpx, Grid: %3d, Tiles: %5zu
    std::snprintf(line2, sizeof(line2),
                  "Iter %5d | Tile %3dpx | Grid %3d | Tiles %5zu",
                  it, tile, statsPx, hmN);
    enforce_ascii_decimal(line2);

    // ── Zeile 3: Center / Res ───────────────────────────────────────────────
    // Center mit 9 Nachkommastellen, Res als W×H (ASCII 'x')
    std::snprintf(line3, sizeof(line3),
                  "Center  %.9f, %.9f | Res %dx%d",
                  cx, cy, w, h);
    enforce_ascii_decimal(line3);

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
