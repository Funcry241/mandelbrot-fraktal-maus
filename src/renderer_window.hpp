// Datei: src/renderer_window.hpp
// Zeilen: 31
// 🐭 Maus-Kommentar: Header für Fenster- und Kontextverwaltung im Renderer. Initialisiert GLFW-Fenster, Kontextversion und registriert Events (Resize, Key). Schneefuchs hätte gesagt: "Ohne Fenster kein Blick in die Unendlichkeit."

#pragma once

#include "pch.hpp"               // 🧩 PCH bringt GLFW, GLEW, Windows & Standard – zentrale Verwaltung
#include "renderer_core.hpp"     // 🔁 Für vollständige Renderer-Definition

class Renderer;

namespace RendererWindow {

GLFWwindow* createWindow(int width, int height, Renderer* instance);
bool shouldClose(GLFWwindow* window);
void swapAndPoll(GLFWwindow* window);
void setResizeCallback(GLFWwindow* window, Renderer* instance);
void setKeyCallback(GLFWwindow* window);

} // namespace RendererWindow
