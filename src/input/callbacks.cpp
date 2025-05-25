// src/input/callbacks.cpp

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "input/callbacks.hpp"

// Registriere die Callbacks
void init_callbacks(GLFWwindow* window) {
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
}

// Definitionen der Callback-Funktionen
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // Passe die Viewport-Größe an
    glViewport(0, 0, width, height);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    // Aktualisiere den Drag-State oder andere Logik
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
    // Berechne Panning-Verschiebung
}
