///// Otter: Heatmap overlay API - draw path + pure compute ROI setter (no GL).
///// Schneefuchs: Leichter Header; keine schweren Includes; ASCII-only; signaturstabil.
///// Maus: updateInterestFromGrid setzt ctx.interest früh im Frame; drawOverlay rendert optional.
///// Datei: src/heatmap_overlay.hpp

#pragma once

#include <vector>

// Vorwärtsdeklaration, um den Header schlank zu halten
struct RendererState;

namespace HeatmapOverlay {

// Sichtbarkeit toggeln (UI)
void toggle(RendererState& ctx);

// GL-Ressourcen freigeben (Overlay-seitig)
void cleanup();

/**
 * Berechnet den ROI aus Entropie/Kontrast und schreibt ihn nach RendererState::interest.
 * Keine GL-Aufrufe, keine Nebenwirkungen außerhalb von ctx.interest.*
 *
 * @param entropy   Heatmap Entropie pro Tile (size = tilesX * tilesY)
 * @param contrast  Heatmap Kontrast  pro Tile (size = tilesX * tilesY)
 * @param width     Framebreite in Pixeln
 * @param height    Framehöhe in Pixeln
 * @param tileSize  Tile-Kantenlänge in Pixeln (Compute-Grid)
 * @param zoom      aktueller Zoom (double)
 * @param ctx       RendererState; schreibt ctx.interest.{ndcX,ndcY,radiusNdc,strength,valid}
 * @return          true bei Erfolg, false wenn Eingaben ungültig/unvollständig
 */
bool updateInterestFromGrid(const std::vector<float>& entropy,
                            const std::vector<float>& contrast,
                            int width, int height, int tileSize,
                            double zoom,
                            RendererState& ctx) noexcept;

/**
 * Zeichnet das Overlay (Panel + Heatmap + Marker). Setzt zuvor immer den ROI
 * via updateInterestFromGrid(...). Wenn ctx.heatmapOverlayEnabled == false,
 * wird nur der ROI gesetzt und der Draw-Teil übersprungen.
 */
void drawOverlay(const std::vector<float>& entropy,
                 const std::vector<float>& contrast,
                 int width, int height, int tileSize,
                 unsigned int textureId,
                 RendererState& ctx);

} // namespace HeatmapOverlay
