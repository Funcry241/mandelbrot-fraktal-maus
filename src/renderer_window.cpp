// Datei: src/renderer_window.cpp
// Zeilen: 71
// 🐭 Maus-Kommentar: Fenster-Erstellung ist jetzt fehlerbehandelbar – kein `std::exit` mehr, sondern nullptr-Rückgabe bei Misserfolg. Aufrufende Instanzen (z. B. Renderer) können reagieren. Schneefuchs: „Nicht jedes Scheitern ist fatal – außer du beendest dich selbst.“

#include "pch.hpp"
#include "renderer_window.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"

namespace RendererInternals {

GLFWwindow* createGLFWWindow(int width, int height) {
    if (!glfwInit()) {
        std::cerr << "[ERROR] GLFW init failed\n";
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        std::cerr << "[ERROR] Window creation failed\n";
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync

    return window;
}

// ✅ Zentrale Callback-Registrierung – wird **nur** über createWindow(...) aufgerufen
void configureWindowCallbacks(GLFWwindow* window, void* userPointer) {
    glfwSetWindowUserPointer(window, userPointer);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* win, int newW, int newH) {
        if (auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(win))) {
            self->resize(newW, newH);
        }
    });

    glfwSetKeyCallback(window, CudaInterop::keyCallback);
}

} // namespace RendererInternals

namespace RendererWindow {

// 🟢 Einzige öffentliche Schnittstelle: Erzeugt Fenster und konfiguriert Callbacks
GLFWwindow* createWindow(int width, int height, Renderer* instance) {
    GLFWwindow* window = RendererInternals::createGLFWWindow(width, height);
    if (!window) return nullptr;
    RendererInternals::configureWindowCallbacks(window, instance);
    return window;
}

bool shouldClose(GLFWwindow* window) {
    return glfwWindowShouldClose(window);
}

// 🧹 Entfernt: setResizeCallback()
// 🧹 Entfernt: setKeyCallback()

void destroyWindow(GLFWwindow* window) {
    if (window) {
        glfwDestroyWindow(window);
    }
}

} // namespace RendererWindow
