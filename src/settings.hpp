// Datei: src/settings.hpp
// 🐅 Maus-Kommentar: Minimalistische Steuerungszentrale – nur aktive, genutzte Konstanten.
// Keine verwaisten Auto-Zoom-Parameter, alles klar und kompakt.

#pragma once

namespace Settings {

// 🔍 Debug-Modus: aktiviert Konsolenausgaben (z. B. für CUDA/Overlay-Diagnose)
constexpr bool debugLogging  = true;

// 🔥 Sichtbarkeit des Heatmap-Overlays beim Programmstart
constexpr bool heatmapOverlayEnabled = true; 

// 🐷 Sichtbarkeit des WarzenschweinOverlays (HUD) beim Programmstart
constexpr bool warzenschweinOverlayEnabled = true; 

// HUD-Textgröße (in NDC-Einheiten pro Pixelquadrat)
inline constexpr float hudPixelSize = 0.0025f;

// 💥 Fensterkonfiguration
constexpr int width       = 1024;
constexpr int height      = 768;
constexpr int windowPosX  = 100;
constexpr int windowPosY  = 100;

// 🔭 Initialer Fraktal-Ausschnitt
constexpr float initialZoom    = 1.5f;
constexpr float initialOffsetX = -0.5f;
constexpr float initialOffsetY = 0.0f;

// ♻️ Iterationssteuerung
constexpr int INITIAL_ITERATIONS = 100;
constexpr int MAX_ITERATIONS_CAP = 50000;

// 💚 CUDA Tile-Größen
constexpr int BASE_TILE_SIZE = 32;
constexpr int MIN_TILE_SIZE  = 8;
constexpr int MAX_TILE_SIZE  = 64;

} // namespace Settings
