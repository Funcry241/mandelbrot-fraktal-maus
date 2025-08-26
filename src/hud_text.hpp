#pragma once
// 🐭 Maus: Zentraler Builder für den HUD-Text (ASCII-only, allokationsarm).
// 🦦 Otter: Zeigt "fps actual (max)" – max aus Core-GPU-Zeit. (Bezug zu Otter)
// 🐑 Schneefuchs: Eine Quelle für Tiles (ctx.width/height + ctx.tileSize), deterministische Formatierung. (Bezug zu Schneefuchs)

#include <string>
#include "frame_context.hpp"
#include "renderer_state.hpp"

namespace HudText {

// Erzeugt den HUD-Text. Keine Nebenwirkungen, ASCII-only.
std::string build(const FrameContext& ctx, const RendererState& state);

} // namespace HudText
