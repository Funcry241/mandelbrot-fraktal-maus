#pragma once
#include <GLFW/glfw3.h>

// Erstellt ein GLFW-Fenster (Core-Profile 4.3), setzt V-Sync (1).
// Gibt das Window-Handle zurück oder nullptr bei Fehler.
GLFWwindow* createGLFWWindow(int width, int height, const char* title);

// Checkt, ob das Fenster geschlossen wurde.
bool windowShouldClose(GLFWwindow* window);

// Tauscht Buffer und polled Events für das Fenster
void swapAndPoll(GLFWwindow* window);
