///// Otter: Projekt 'Pfau' – identical UI params/alpha; anchor Top-Right; header cleaned.
///// Schneefuchs: MAUS header (#62) strict; no GL includes in header; ASCII-only docs.
///// Maus: Pfau final; keep function signatures unchanged so build stays stable until .cpp update.
///// Datei: src/heatmap_overlay.hpp

#pragma once

#include <vector>
#include "renderer_state.hpp"

// No GL headers here to avoid include-order constraints (GLEW before GL).
using GLuint = unsigned int;

namespace HeatmapOverlay {

// -----------------------------------------------------------------------------
// Pfau theme: unified UI parameters (no alternates, hard-wired)
// -----------------------------------------------------------------------------
namespace Pfau {
inline constexpr float UI_MARGIN   = 16.0f;   // outer margin to window edge (px)
inline constexpr float UI_PADDING  = 12.0f;   // inner padding content->panel (px)
inline constexpr float UI_RADIUS   = 12.0f;   // panel corner radius (px)
inline constexpr float UI_BORDER   = 1.5f;    // panel border thickness (px)
inline constexpr float PANEL_ALPHA = 0.84f;   // panel opacity (0..1)
inline constexpr const char* LOG_PREFIX = "[UI/Pfau]"; // ASCII log prefix

// Anchor for mini heatmap: Top-Right (Warzenschwein uses Top-Left)
enum class Anchor { TopRight };
inline constexpr Anchor ANCHOR = Anchor::TopRight;
} // namespace Pfau

// Pixel snapping helper (for crisp edges)
inline int snapToPixel(float v) { return static_cast<int>(v + 0.5f); }

// Toggle overlay flag in RendererState (no hidden globals)
void toggle(RendererState& ctx);

// Release all GL resources created by the overlay (idempotent)
void cleanup();

// Draw the mini heatmap (Pfau material: panel + texture-driven heatmap with
// FS glow/alpha). Expects one value per tile in 'entropy' and 'contrast'
// (size tilesX*tilesY). width/height = framebuffer, tileSize = tile size in
// pixels. textureId is optional (0 allowed).
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height,
                 int tileSize,
                 GLuint textureId,
                 RendererState& ctx);

} // namespace HeatmapOverlay
