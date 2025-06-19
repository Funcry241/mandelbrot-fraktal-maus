// Datei: src/renderer_window.cpp
// Zeilen: 85
// 🐭 Maus-Kommentar: Initialisierung und Verwaltung des GLFW-Fensters. Korrigiert: Öffentliche API in RendererWindow, interne Helfer in RendererInternals. Schneefuchs streicht sich zufrieden durchs Fell.

#include "pch.hpp"
#include "renderer_window.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "cuda_interop.hpp"

namespace RendererInternals {

GLFWwindow* createGLFWWindow(int width, int height) {
    if (!glfwInit()) {
        std::cerr << "[ERROR] GLFW init failed\n";
        std::exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        std::cerr << "[ERROR] Window creation failed\n";
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync

    return window;
}

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

GLFWwindow* createWindow(int width, int height, Renderer* instance) {
    GLFWwindow* window = RendererInternals::createGLFWWindow(width, height);
    RendererInternals::configureWindowCallbacks(window, instance);
    return window;
}

bool shouldClose(GLFWwindow* window) {
    return glfwWindowShouldClose(window);
}

void swapAndPoll(GLFWwindow* window) {
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void setResizeCallback(GLFWwindow* window, Renderer* instance) {
    RendererInternals::configureWindowCallbacks(window, instance); // 🟡 Duplikat, aber korrekt
}

void setKeyCallback(GLFWwindow* window) {
    glfwSetKeyCallback(window, CudaInterop::keyCallback);
}

void destroyWindow(GLFWwindow* window) {
    if (window) {
        glfwDestroyWindow(window);
    }
}

} // namespace RendererWindow
