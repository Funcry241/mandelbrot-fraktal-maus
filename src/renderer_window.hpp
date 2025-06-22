// Datei: src/renderer_window.hpp
// Zeilen: 27
// 🐭 Maus-Kommentar: Header für GLFW-Fensterverwaltung – Callback-Registrierung jetzt ausschließlich über `createWindow(...)`. Keine Mehrdeutigkeit, keine Überschreibgefahr. Schneefuchs: „Ein Fenster, eine Regel – keine wilden Handler mehr.“

#pragma once

#include "pch.hpp"               // 🧩 PCH bringt GLFW, GLEW, Windows & Standard – zentrale Verwaltung
#include "renderer_core.hpp"     // 🔁 Für vollständige Renderer-Definition

class Renderer;

namespace RendererWindow {

GLFWwindow* createWindow(int width, int height, Renderer* instance);  // 🟢 Erstellt Fenster und konfiguriert alle Callbacks
bool shouldClose(GLFWwindow* window);

void destroyWindow(GLFWwindow* window);  // 🧼 Ressourcen korrekt freigeben

// 🧹 Entfernt: setResizeCallback(...)
// 🧹 Entfernt: setKeyCallback(...)

} // namespace RendererWindow
