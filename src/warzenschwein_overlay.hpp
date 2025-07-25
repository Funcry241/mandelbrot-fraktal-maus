/*
🐭 Maus-Kommentar: Warzenschwein ist der moderne Gegenentwurf zu `stb_easy_font`. Statt veralteter Fixed-Function nutzt es OpenGL 4.3, Shader und VAO. Jeder Text besteht aus individuellen Quads mit Per-Vertex-Position und Grauwert für Textfarbe. Otter-approved.
*/

#pragma once
#include <string>
#include <vector>
#include "renderer_state.hpp"

namespace WarzenschweinOverlay {

// Aktiviert oder deaktiviert das Text-Overlay im HUD
void toggle(RendererState& ctx);

// Gibt alle GPU-Ressourcen (VAO, VBO, Shader) frei
void cleanup();

// Setzt den darzustellenden Text (wird intern in Quads umgewandelt)
// Position ist oben links in Pixelkoordinaten
void setText(const std::string& text, int x, int y);

// Zeichnet den vorbereiteten Text (nur wenn `ctx.warzenschweinOverlayEnabled == true`)
void drawOverlay(RendererState& ctx);

} // namespace Warzenschwein
