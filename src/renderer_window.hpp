// Datei: src/renderer_window.hpp
// Zeilen: 31
// 🐭 Maus-Kommentar: Header für Fenster- und Kontextverwaltung im Renderer. Jetzt mit getrennter Callback-Registrierung – kein Überschreiben mehr. Schneefuchs: „Ein Ereignis, ein Handler – so bleibt das Rudel stabil.“

#pragma once

#include "pch.hpp"               // 🧩 PCH bringt GLFW, GLEW, Windows & Standard – zentrale Verwaltung
#include "renderer_core.hpp"     // 🔁 Für vollständige Renderer-Definition

class Renderer;

namespace RendererWindow {

GLFWwindow* createWindow(int width, int height, Renderer* instance);
bool shouldClose(GLFWwindow* window);

// 🔁 Neu: Callback-Registrierung klar getrennt
void setResizeCallback(GLFWwindow* window, Renderer* instance);
void setKeyCallback(GLFWwindow* window);

void destroyWindow(GLFWwindow* window);  // 🆕 Fenster korrekt schließen

} // namespace RendererWindow
