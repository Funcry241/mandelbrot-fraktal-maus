///// Otter: Render-Loop API – minimal & stable; kein beginFrame-Export.
///// Schneefuchs: Schlanker Header; GLFWwindow vorwärts deklariert; ASCII-only.
///// Maus: FPS/HUD/Auto-Zoom/Eingaben sauber gekapselt; Source definiert Verhalten.

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
