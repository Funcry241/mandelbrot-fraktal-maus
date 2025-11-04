///// Otter: Axolotel-Coupler API — exposes activityEnergy() for Zoom boost; clean minimal surface.
///// Schneefuchs: Header-only declarations; ASCII-only; no GL deps here; stable ABI.
///// Maus: init/shutdown/noteKeyPress/draw + energy tap; optional configure & setColors preserved.
///// Datei: src/axolotel_hud.hpp

#pragma once

namespace AxolotelHUD {

// Lifetime
void init();
void shutdown();

// Global toggle
void setEnabled(bool enabled);
bool isEnabled();

// Events + draw
void noteKeyPress(int key, int mods);
void draw(int viewportWidth, int viewportHeight, double timeSeconds);

// Coupler: returns current activity energy E in [0,1] based on live pulses (time = glfwGetTime()).
// Cheap O(pulses), does NOT mutate internal buffers.
float activityEnergy();

// Optional styling/config (kept for compatibility)
void setColors(float nr,float ng,float nb,
               float or_,float og,float ob,
               float sr,float sg,float sb,
               float tr,float tg,float tb);

void configure(float pulseMs, float breathAmp, float breathHz,
               int maxPulsesClamped, bool perfLog);

} // namespace AxolotelHUD
