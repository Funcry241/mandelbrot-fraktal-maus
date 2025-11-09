///// Otter: Dachs-HUD -- tri-pane top layout (Left | Center | Right) with centered help; persistent hint-badge state.
///// Schneefuchs: ASCII-only; pixel-snapped; DPI-scaled; no GL deps; zero hot-path allocs.
///// Maus: implements set_text/visible/style, toggle_help; adds help-hint controls (enable_help_hint/set_help_hint_text).
///// Datei: src/dachs_hud.cpp

#include "dachs_hud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace DachsHUD {

namespace {
    // local state (single-threaded usage)
    Style g_style{};
    std::array<bool,3> g_visible{ true, false, true }; // center off by default
    std::array<std::string,3> g_text{
        std::string{}, std::string{}, std::string{}
    };
    bool g_help{false};

    // Hint-Badge unten links
    bool        g_help_hint_enabled = true;
    std::string g_help_hint_text    = "F1 - Help";

    // Empfohlener vertikaler Sicherheitsabstand für das F1-Hint unten links,
    // damit es nicht vom Eye/Glow überlappt wird (Pixel vor DPI).
    constexpr int kHelpHintYOffsetPx = 96;

    inline int iround(float v) {
        return static_cast<int>(std::lround(v));
    }

    inline Align default_align_for(Pane p) {
        switch (p) {
            case Pane::Left:   return Align::Left;
            case Pane::Center: return Align::Center;
            case Pane::Right:  return Align::Right;
        }
        return Align::Left;
    }
} // namespace

// visibility / text / style ---------------------------------------------------

void set_visible(Pane p, bool v) {
    g_visible[static_cast<int>(p)] = v;
}

bool is_visible(Pane p) {
    return g_visible[static_cast<int>(p)];
}

void set_text(Pane p, std::string_view s) {
    g_text[static_cast<int>(p)].assign(s.begin(), s.end());
}

std::string get_text(Pane p) {
    return g_text[static_cast<int>(p)];
}

void set_style(const Style& s) { g_style = s; }
Style get_style() { return g_style; }

// help toggle/state -----------------------------------------------------------

bool toggle_help() {
    g_help = !g_help;
    set_visible(Pane::Center, g_help);
    if (g_help && g_text[static_cast<int>(Pane::Center)].empty()) {
        g_text[static_cast<int>(Pane::Center)] = build_help_text();
    }
    return g_help;
}

bool help_enabled() { return g_help; }

std::string build_help_text() {
    // 7 Zeilen, ASCII-only
    std::string s;
    s  = "DACHS-HUD\n";
    s += "F1       : Overlay on/off\n";
    s += "WASD/Arrows : Pan (<50 ms, deadzone)\n";
    s += "R        : Reset view\n";
    s += "Space    : Pause/Resume\n";
    s += "Strg+C   : Copy State (cx,cy,zoom)\n";
    s += "Esc      : Quit";
    return s;
}

// layout ----------------------------------------------------------------------

std::array<Rect,3> compute_tri_columns(int vpW, int /*vpH*/, float dpiScale) {
    const int top  = iround(static_cast<float>(g_style.topMarginPx) * dpiScale);
    const int colW = std::max(1, vpW / 3);

    std::array<Rect,3> out{};

    // left
    out[0].x = 0;
    out[0].y = top;
    out[0].w = colW;
    out[0].h = 0;

    // center
    out[1].x = colW;
    out[1].y = top;
    out[1].w = colW;
    out[1].h = 0;

    // right (eat remainder from integer division)
    out[2].x = 2 * colW;
    out[2].y = top;
    out[2].w = vpW - out[2].x;
    out[2].h = 0;

    // optional clamp for extremely wide windows (aesthetic)
    if (g_style.clampCenterWidth) {
        const int maxW = g_style.maxCenterWidthPx > 0
                       ? iround(static_cast<float>(g_style.maxCenterWidthPx) * dpiScale)
                       : colW;
        const int desiredW = std::min(out[1].w, maxW);
        const int cx = out[1].x + (out[1].w - desiredW) / 2;
        out[1].x = cx;
        out[1].w = desiredW;
    }

    return out;
}

RenderModel build_render_model(int vpW, int vpH, float dpiScale) {
    RenderModel model{};
    model.style = g_style;

    const auto cols = compute_tri_columns(vpW, vpH, dpiScale);

    for (int i = 0; i < 3; ++i) {
        const Pane p = static_cast<Pane>(i);
        auto& it = model.items[i];
        it.pane    = p;
        it.align   = default_align_for(p);
        it.rect    = cols[i];
        it.visible = g_visible[i] && !g_text[i].empty();
        it.text    = g_text[i];
    }

    // height is left at 0; the concrete text renderer should measure text
    // (lines * lineHeight + padY*2) and draw equal-height boxes if requested.

    return model;
}

// --- Hint-Badge-API ----------------------------------------------------------

void enable_help_hint(bool on) noexcept { g_help_hint_enabled = on; }
bool help_hint_enabled() noexcept { return g_help_hint_enabled; }

void set_help_hint_text(std::string s) {
    if (s.empty()) g_help_hint_text = "F1 - Help";
    else           g_help_hint_text = std::move(s);
}

const std::string& help_hint_text() { return g_help_hint_text; }

// Empfohlene Y-Verschiebung (Pixel) für das F1-Hint unten links.
// Konsumiere im Renderer z.B. so:  hintPanelY = vpH - badgeH - DachsHUD::help_hint_offset_px(dpi);
int help_hint_offset_px(float dpiScale){
    return iround(static_cast<float>(kHelpHintYOffsetPx) * dpiScale);
}

} // namespace DachsHUD
