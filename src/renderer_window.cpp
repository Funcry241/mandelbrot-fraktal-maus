// Datei: src/renderer_window.cpp
// 🐭 Maus-Kommentar: Fixed-Function raus, moderner Kontext rein - für Warzenschwein wird OpenGL 4.3 erzwungen. Keine Kompromisse mehr, Otter-Style.

#include "pch.hpp"
#include "renderer_window.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp" // für RendererLoop::keyCallback
#include <cstdio>

namespace RendererInternals {

// Erstellt GLFW-Fenster, setzt OpenGL-Kontext & VSync, Position aus Settings
GLFWwindow* createGLFWWindow(int width, int height) {
    if (!glfwInit()) {
        LUCHS_LOG( "[ERROR] GLFW init failed\n");
        return nullptr;
    }

    // OpenGL 4.3 Core - nötig für Shader-Text (Warzenschwein!)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(width, height, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        LUCHS_LOG( "[ERROR] Window creation failed\n");
        glfwTerminate();
        return nullptr;
    }

    glfwSetWindowPos(window, Settings::windowPosX, Settings::windowPosY);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSync an
    return window;
}

// Registriert Callbacks (Größe, Taste) am Fenster - UserPointer ist Renderer*
void configureWindowCallbacks(GLFWwindow* window, void* userPointer) {
    glfwSetWindowUserPointer(window, userPointer);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* win, int newW, int newH) {
        if (auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(win))) {
            self->resize(newW, newH);
        }
    });

    // KeyCallback aus RendererLoop - so hat man Zugriff auf RendererState
    glfwSetKeyCallback(window, RendererLoop::keyCallback);
}

} // namespace RendererInternals

namespace RendererWindow {

// Erstellt Fenster und konfiguriert alle Callbacks
GLFWwindow* createWindow(int width, int height, Renderer* instance) {
    GLFWwindow* window = RendererInternals::createGLFWWindow(width, height);
    if (!window) return nullptr;
    RendererInternals::configureWindowCallbacks(window, instance);
    return window;
}

bool shouldClose(GLFWwindow* window) {
    return glfwWindowShouldClose(window);
}

void destroyWindow(GLFWwindow* window) {
    if (window) {
        glfwDestroyWindow(window);
    }
}

} // namespace RendererWindow
