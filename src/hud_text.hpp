///// Otter: Zentraler HUD-Builder – ASCII-only; Forward-Decls halten den Header schlank.
///// Schneefuchs: Header/Source synchron; keine schweren Includes; /WX-fest.
///// Maus: Eine Quelle für Tiles; deterministische Formatierung; keine Seiteneffekte.
///// Datei: src/hud_text.hpp

#pragma once

#include <string>

// Schlank: nur Forward-Declarations statt schwerer Includes.
struct FrameContext;
class RendererState;

namespace HudText {

// Erzeugt den HUD-Text. Keine Nebenwirkungen, ASCII-only.
[[nodiscard]] std::string build(const FrameContext& ctx, const RendererState& state);

} // namespace HudText
