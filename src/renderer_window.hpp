// Datei: src/renderer_window.hpp
// 🐭 Maus-Kommentar: GLFW-Fensterverwaltung als zentrale API - Callbacks nur noch über createWindow(). Keine Mehrdeutigkeit. Schneefuchs-konform.
// 🦦 Otter: Schlanker Header ohne Mass-Includes; klare Forward-Declarations. (Bezug zu Otter)
// 🦊 Schneefuchs: Header/Source synchron, keine versteckten Abhängigkeiten. (Bezug zu Schneefuchs)

#pragma once

// Schlank: keine PCH/GLFW-Header hier
struct GLFWwindow;   // forward decl
class Renderer;      // forward decl

namespace RendererWindow {

// 🟢 Erstellt Fenster und registriert alle Callbacks (Größe, Tasten, etc.)
[[nodiscard]] GLFWwindow* createWindow(int width, int height, Renderer* instance);

// Fragt Fenster-Schließwunsch ab
[[nodiscard]] bool shouldClose(GLFWwindow* window);

// Gibt Fenster und Ressourcen frei
void destroyWindow(GLFWwindow* window);

} // namespace RendererWindow
