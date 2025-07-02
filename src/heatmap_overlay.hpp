// Zeilen: 32
// Datei: src/heatmap_overlay.hpp
/*
Maus-Kommentar 🐭: Overlay ist jetzt vollständig zustandslos – kein interner bool mehr. Alle Kontrollfunktionen arbeiten direkt mit RendererState&. drawOverlay-API akzeptiert ctx. Schneefuchs: „Kein Schatten, nur Klarheit.“
*/

#pragma once
#include <vector>
#include <GL/glew.h>

struct RendererState;

namespace HeatmapOverlay {

// Overlay ein-/ausblenden via Tastendruck (setzt ctx.overlayEnabled um)
void toggle(RendererState& ctx);

// Gibt GPU-Ressourcen (VAO, VBO, Shader) frei
void cleanup();

// Zeichnet das Debug-Overlay über dem Fraktalbild.
// entropy + contrast: Tile-Daten (gleiche Länge)
// width, height: Bildgröße in Pixel
// tileSize: Größe eines Tiles in Pixeln
// textureId: Fraktal-Textur (optional, wird ignoriert)
// ctx: Zustandsobjekt mit overlayEnabled-Flag
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width,
                 int height,
                 int tileSize,
                 GLuint textureId,
                 RendererState& ctx);

// Convenience-Version ohne textureId
void drawOverlayTexture(const std::vector<float>& entropy,
                        const std::vector<float>& contrast,
                        int width,
                        int height,
                        int tileSize,
                        RendererState& ctx);

} // namespace HeatmapOverlay
