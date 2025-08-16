// MAUS:
// Datei: src/renderer_window.cpp
// 🐭 Maus-Kommentar: Fixed-Function raus, moderner Kontext rein - für Warzenschwein wird OpenGL 4.3 erzwungen. Keine Kompromisse mehr, Otter-Style.
// 🦦 Otter: LUCHS_LOG_HOST für GLFW-Fehler. Schneefuchs: Klarer Host-Kontext.
// 🐑 Schneefuchs: Debug-Kontext optional (nur bei Debug/Perf), VSync abhängig vom Perf-Modus, zentrierte Fensterposition fallback.

#include "pch.hpp"
#include "renderer_window.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp" // für RendererLoop::keyCallback
#include "luchs_log_host.hpp"
#include <GLFW/glfw3.h>
#include <cstdio>

namespace RendererInternals {

// 🔔 Otter: GLFW-Error-Callback für robustes Fehler-Logging (ASCII)
static void glfwErrorCallback(int code, const char* description) {
    LUCHS_LOG_HOST("[GLFW-ERROR] Code=%d | %s", code, description ? description : "(null)");
}

// 🐑 Schneefuchs: Fallback – Fenster zentrieren, wenn Settings-Position invalid (<0)
// NOTE: if constexpr entfernt MSVC C4127 (constant conditional).
static void centerWindowIfRequested(GLFWwindow* window, int w, int h) {
    if (!window) return;
    if constexpr ((Settings::windowPosX >= 0) && (Settings::windowPosY >= 0)) {
        glfwSetWindowPos(window, Settings::windowPosX, Settings::windowPosY);
        return;
    } else {
        GLFWmonitor* mon = glfwGetPrimaryMonitor();
        const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;
        if (vm) {
            const int x = (vm->width  - w) / 2;
            const int y = (vm->height - h) / 2;
            glfwSetWindowPos(window, x, y);
        }
    }
}

// Erstellt GLFW-Fenster, setzt OpenGL-Kontext & VSync, Position aus Settings
GLFWwindow* createGLFWWindow(int width, int height) {
    // Setze Error-Callback bevor glfwInit()
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        LUCHS_LOG_HOST("[ERROR] GLFW init failed");
        return nullptr;
    }

    // OpenGL 4.3 Core - nötig für Debug/KHR und unsere Pipelines
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // 🐑 Schneefuchs: Debug-Kontext nur, wenn wir Logs messen (kein Overhead im Release)
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    } else {
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_FALSE);
    }

#ifdef GLFW_SRGB_CAPABLE
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(width, height, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        LUCHS_LOG_HOST("[ERROR] Window creation failed");
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    // 🐑 Schneefuchs: VSync aus im Perf-Modus, sonst an (stabilere Frametime im Normalbetrieb)
    glfwSwapInterval(Settings::performanceLogging ? 0 : 1);

    // 🐭 Maus: Position setzen/zentrieren
    centerWindowIfRequested(window, width, height);

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
    // glfwTerminate() erfolgt im Renderer-Cleanup (zentral), nicht hier.
}

} // namespace RendererWindow
