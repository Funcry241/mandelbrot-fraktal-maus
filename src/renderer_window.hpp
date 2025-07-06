// Datei: src/renderer_window.hpp
// Zeilen: 23
// 🐭 Maus-Kommentar: GLFW-Fensterverwaltung als zentrale API – Callbacks nur noch über createWindow(). Keine Mehrdeutigkeit. Schneefuchs-konform.

#pragma once

#include "pch.hpp" // 🧩 PCH: enthält GLFW, GLEW etc.
#include "renderer_core.hpp" // 🔁 Für Renderer-Definition

class Renderer;

namespace RendererWindow {

// 🟢 Erstellt Fenster und registriert alle Callbacks (Größe, Tasten, etc.)
GLFWwindow* createWindow(int width, int height, Renderer* instance);

// Fragt Fenster-Schließwunsch ab
bool shouldClose(GLFWwindow* window);

// Gibt Fenster und Ressourcen frei
void destroyWindow(GLFWwindow* window);

} // namespace RendererWindow
