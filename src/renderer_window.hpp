// Datei: src/renderer_window.hpp
// Zeilen: 32
// 🐭 Maus-Kommentar: Header für Fenster- und Kontextverwaltung im Renderer. Neu: `destroyWindow()` schließt GLFW sauber. Schneefuchs: „Ein Fenster, das nicht schließt, lässt den Otter erfrieren.“

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
void destroyWindow(GLFWwindow* window);  // 🆕 Fenster korrekt schließen

} // namespace RendererWindow
