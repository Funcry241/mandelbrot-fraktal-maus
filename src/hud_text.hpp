///// Otter: HUD-Text Header - schlanke öffentliche API für kompakte Center-Statistik.
///// Schneefuchs: Einheitliche Forward-Decls (FrameContext=struct, RendererState=struct); ASCII-only; /WX clean.
///// Maus: Nur Signatur; Implementierung in hud_text.cpp.
///// Datei: src/hud_text.hpp

#pragma once
#include <string>

// Schlanke Forward-Decls
struct FrameContext;   // struct
struct RendererState;  // struct

namespace HudText {

// Baut die 3-zeilige Statistik (cx/cy, z/it/tile, res/fps)
std::string build(const FrameContext& fctx, const RendererState& state);

} // namespace HudText
