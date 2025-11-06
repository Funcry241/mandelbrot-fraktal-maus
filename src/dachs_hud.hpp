///// Otter: Dachs-HUD -- tri-pane top layout (Left | Center | Right) with centered "Dachs-HUD" help + tiny bottom-left "F1 - Help" hint API.
///// Schneefuchs: ASCII-only; pixel-snapped; DPI-scaled metrics; no GL deps; zero hot-path allocs.
///// Maus: state API incl. set_text/visible/style, toggle_help, enable_help_hint/set_help_hint_text; renderer builds boxes from model.
///// Ziel: symmetrische Top-HUDs; Mitte priorisiert; 6–8 lines onboarding; stable across resizes.
///// Perf: layout math is O(1); text measuring left to existing renderer; equal-height optional.
///// Log: [HELP] only on toggle; no spam; header belongs to caller.
///// Datei: src/dachs_hud.hpp

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace DachsHUD {

// logical panes across the top row
enum class Pane : int { Left = 0, Center = 1, Right = 2 };

// text alignment hint for the renderer
enum class Align : int { Left = 0, Center = 1, Right = 2 };

// pixel rect (target box for background + padded text)
struct Rect {
    int x{0}, y{0}, w{0}, h{0};
};

// visual and metric style; renderer interprets these fields
struct Style {
    // metrics (pre-DPI; multiplied by dpiScale)
    int   topMarginPx     = 8;
    int   padX            = 8;
    int   padY            = 6;
    float fontPx          = 14.0f;
    // colors (packed RGBA 0xRRGGBBAA)
    uint32_t bgRGBA       = 0x2A2A2ACC;
    uint32_t fgRGBA       = 0xFFFFFFFF;
    // layout tweaks
    bool  equalHeightBoxes = true;
    bool  clampCenterWidth = false;
    int   maxCenterWidthPx = 720;
};

// one render item per pane
struct Item {
    Pane pane{Pane::Left};
    Align align{Align::Left};
    Rect rect{};
    bool visible{false};
    std::string text;
};

// full model for a frame
struct RenderModel {
    Style style;
    std::array<Item, 3> items;
};

// visibility per pane
void set_visible(Pane p, bool v);
bool is_visible(Pane p);

// text content per pane
void set_text(Pane p, std::string_view s);
std::string get_text(Pane p);

// style access
void set_style(const Style& s);
Style get_style();

// help toggle/state (center pane)
bool toggle_help();   // returns new state (true = visible)
bool help_enabled();

// compact onboarding help text (6-8 lines, ASCII-only)
std::string build_help_text();

// compute symmetric three columns across the top (pixel-snapped)
std::array<Rect,3> compute_tri_columns(int vpW, int vpH, float dpiScale);

// build full render model for current state/style
RenderModel build_render_model(int vpW, int vpH, float dpiScale);

// --- kleines Hint-Badge unten links ("F1 - Help") ---
void enable_help_hint(bool on) noexcept;
bool help_hint_enabled() noexcept;
void set_help_hint_text(std::string s);
const std::string& help_hint_text();

} // namespace DachsHUD
