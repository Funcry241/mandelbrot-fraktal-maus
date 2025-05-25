// src/gui.hpp

#pragma once

#include <GLFW/glfw3.h>
#include "settings.hpp"
#include "metrics.hpp"

// Erzeugt das GLFW-Fenster (öffnet das Fenster und setzt Kontext)
GLFWwindow* init_window();

// Initialisiert ImGui für das gegebene Fenster
void init_gui(GLFWwindow* window);

// Zeichnet das HUD (FPS, Zoom, etc.)
void render_gui(const Settings& S, const Metrics& M);

// Räumt ImGui wieder auf
void shutdown_gui();
