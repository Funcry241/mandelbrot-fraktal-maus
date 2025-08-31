///// Otter: Schlanker Header ohne Mass-Includes; klare Forward-Declarations.
///// Schneefuchs: Header/Source synchron; keine versteckten Abhängigkeiten; /WX-fest.
///// Maus: GLFW-Fensterverwaltung als zentrale API; Callbacks nur via createWindow().
///// Datei: src/renderer_window.hpp

#pragma once

// Forward declarations – kein schwerer GLFW-Header
struct GLFWwindow;
class Renderer;

namespace RendererWindow {

// Erstellt Fenster und registriert alle Callbacks (Größe, Tasten, etc.)
[[nodiscard]] GLFWwindow* createWindow(int width, int height, Renderer* instance);

// Abfrage, ob das Fenster geschlossen werden soll
[[nodiscard]] bool shouldClose(GLFWwindow* window);

// Gibt das Fenster frei (glfwTerminate erfolgt zentral im Renderer)
void destroyWindow(GLFWwindow* window);

} // namespace RendererWindow
