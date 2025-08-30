// 🐭 Maus-Kommentar: Render-Loop, FPS, HUD, Auto-Zoom und Eingaben sauber gekapselt.
// 🦦 Otter: minimale, stabile API; kein beginFrame-Export nötig. (Bezug zu Otter)
// 🦊 Schneefuchs: Header schlank – keine schweren Includes, Forward-Decl für GLFWwindow. (Bezug zu Schneefuchs)

#pragma once

// Forward declaration statt schwerem GLFW-Header
struct GLFWwindow;

#include "renderer_state.hpp"

namespace RendererLoop {

// 🎬 Kompletter Frame: CUDA, Auto-Zoom, Textur, HUD, Overlay
void renderFrame_impl(RendererState& state);

// 🎹 Tastatur-Callback für GLFW (Overlay, Pause, Zoom etc.)
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

} // namespace RendererLoop
