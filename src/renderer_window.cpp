///// Otter: Moderner GL-Kontext (4.3 Core), deterministischer Setup-Flow; frühes Fehler-Logging – aber erst NACH glfwInit().
///// Schneefuchs: ASCII-Logs; kein glfwGetWindowAttrib(GLFW_SRGB_CAPABLE); sRGB-Check erfolgt per GL in renderer_core.
///// Maus: State clean; zentrieren/Position deterministisch; Header/Source synchron; kein Pre-Init-GLFW in main.
///// Datei: src/renderer_window.cpp

#include "pch.hpp"
#include "renderer_window.hpp"
#include "renderer_core.hpp"
#include "settings.hpp"
#include "renderer_loop.hpp"   // RendererLoop::keyCallback
#include "luchs_log_host.hpp"

#include <GLFW/glfw3.h>

namespace RendererInternals {

// GLFW-Error-Callback (ASCII only)
static void glfwErrorCallback(int code, const char* description) {
    LUCHS_LOG_HOST("[GLFW-ERROR] code=%d desc=%s", code, description ? description : "(null)");
}

// Fenster zentrieren ODER feste Position setzen (compile-time Branch, um C4127 zu vermeiden).
static void centerWindowIfRequested(GLFWwindow* window, int w, int h) {
    if (!window) return;

    constexpr bool kHasFixedPos =
        (Settings::windowPosX >= 0) && (Settings::windowPosY >= 0);

    if constexpr (kHasFixedPos) {
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

// Erstellt Fenster, setzt GL-Kontext & VSync
static GLFWwindow* createGLFWWindow(int width, int height) {
    // (2) Korrekte Reihenfolge: erst glfwInit(), DANN Error-Callback setzen
    if (!glfwInit()) {
        LUCHS_LOG_HOST("[ERROR] GLFW init failed");
        return nullptr;
    }
    glfwSetErrorCallback(glfwErrorCallback);

    glfwDefaultWindowHints();

    // OpenGL 4.3 Core (Forward-Compatible nur wo nötig)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if defined(__APPLE__)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Debug-Kontext nur, wenn wir tatsächlich debuggen/profilen (compile-time)
    if constexpr (Settings::debugLogging || Settings::performanceLogging) {
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    } else {
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_FALSE);
    }

#ifdef GLFW_SRGB_CAPABLE
    // Hint ist ok – ABER nicht per glfwGetWindowAttrib abfragen (siehe Fix 3).
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
#endif

    // Farbtiefen/Buffer explizit setzen
    glfwWindowHint(GLFW_RED_BITS,     8);
    glfwWindowHint(GLFW_GREEN_BITS,   8);
    glfwWindowHint(GLFW_BLUE_BITS,    8);
    glfwWindowHint(GLFW_ALPHA_BITS,   8);
    glfwWindowHint(GLFW_DEPTH_BITS,  24);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE,    GLFW_TRUE);
    glfwWindowHint(GLFW_VISIBLE,      GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(width, height, "OtterDream Mandelbrot", nullptr, nullptr);
    if (!window) {
        LUCHS_LOG_HOST("[ERROR] Window creation failed w=%d h=%d", width, height);
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    // (3) KEIN glfwGetWindowAttrib(window, GLFW_SRGB_CAPABLE) hier!
    //     Der korrekte sRGB-Check erfolgt nach GLEW-Init im renderer_core via GL:
    //     glGetFramebufferAttachmentParameteriv(..., GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, ...)
    //     -> dort wird geloggt "[GL] Default FB sRGB capable: <0|1>"

    // VSync: im Perf-Logging-Modus aus, sonst an
    glfwSwapInterval(Settings::performanceLogging ? 0 : 1);

    centerWindowIfRequested(window, width, height);
    return window;
}

// Callbacks registrieren (Größe, Tasten)
static void configureWindowCallbacks(GLFWwindow* window, void* userPointer) {
    glfwSetWindowUserPointer(window, userPointer);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* win, int newW, int newH) {
        if (auto* self = static_cast<Renderer*>(glfwGetWindowUserPointer(win))) {
            self->resize(newW, newH);
        }
    });

    glfwSetKeyCallback(window, RendererLoop::keyCallback);
}

} // namespace RendererInternals

namespace RendererWindow {

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
    // glfwTerminate() erfolgt zentral beim Programmende (Renderer::cleanup()).
}

} // namespace RendererWindow
